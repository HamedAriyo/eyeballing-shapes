# eyeballing_shapes_app.py
# Local, no-cloud shape “eyeballing” with live calibration
# Hamed-friendly: deterministic, adjustable, and explainable.

import io
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import cv2
import streamlit as st
from skimage.measure import label, regionprops

# ---------- Config & helpers ----------
CONFIG_PATH = "eyeballing_config.json"

@dataclass
class EyeballConfig:
    blur_kernel: int = 5                 # odd
    thresh_mode: str = "adaptive"        # "adaptive" or "otsu"
    adaptive_block: int = 35             # odd
    adaptive_C: int = 5
    morph_open_iter: int = 1
    morph_close_iter: int = 1
    min_area: int = 100                  # px
    approx_eps_pct: float = 0.02         # fraction of perimeter
    circularity_circle: float = 0.80     # >= => circle-ish
    square_ar_tol: float = 0.15          # aspect ratio tolerance for square
    rect_angle_tol_deg: float = 15.0     # max deviation from 90 deg for rectangles
    star_defects_ratio: float = 0.035    # convexity defects heuristic

def load_config() -> EyeballConfig:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return EyeballConfig(**data)
    except Exception:
        return EyeballConfig()

def save_config(cfg: EyeballConfig):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

def angle_between(p1, p2, p3) -> float:
    """Return angle at p2 in degrees for triangle p1-p2-p3."""
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cosang = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)
    cosang = np.clip(cosang, -1, 1)
    return np.degrees(np.arccos(cosang))

def contour_circularity(cnt) -> float:
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if perim <= 1e-6:
        return 0.0
    return (4.0 * np.pi * area) / (perim * perim)

def convexity_defects_ratio(cnt) -> float:
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0.0
    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        return 0.0
    # sum of defect depths / perimeter
    depths = defects[:, 0, 3] if defects.ndim == 3 else defects[:, 3]
    perim = cv2.arcLength(cnt, True) + 1e-6
    return float(np.sum(depths) / perim)

def classify_polygon(approx, cnt, cfg: EyeballConfig) -> str:
    sides = len(approx)

    # Basic shapes by vertex count
    if sides == 3:
        return "triangle"
    if sides == 4:
        # Square vs rectangle by AR and right angles
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        ar_ok = abs(ar - 1.0) <= cfg.square_ar_tol

        # angle sanity: expect ~90 deg at all corners
        angles = []
        for i in range(4):
            p1 = tuple(approx[(i - 1) % 4][0])
            p2 = tuple(approx[i][0])
            p3 = tuple(approx[(i + 1) % 4][0])
            angles.append(angle_between(p1, p2, p3))
        rightish = all(abs(a - 90.0) <= cfg.rect_angle_tol_deg for a in angles)

        if rightish and ar_ok:
            return "square"
        if rightish:
            return "rectangle"
        return "quadrilateral"

    if sides == 5:
        return "pentagon"
    if sides == 6:
        return "hexagon"

    # Many sides: circle/ellipse/star-ish
    circ = contour_circularity(cnt)
    defects_r = convexity_defects_ratio(cnt)
    if circ >= cfg.circularity_circle:
        return "circle"
    # spiky shapes tend to have higher defects
    if defects_r >= cfg.star_defects_ratio:
        return "star-ish"
    return "ellipse/round-ish"

def classify_contour(cnt, cfg: EyeballConfig) -> str:
    perim = cv2.arcLength(cnt, True)
    eps = cfg.approx_eps_pct * perim
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) < 3:
        return "unknown"
    return classify_polygon(approx, cnt, cfg)

def preprocess(img_bgr: np.ndarray, cfg: EyeballConfig) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    k = max(3, cfg.blur_kernel | 1)  # force odd
    gray = cv2.GaussianBlur(gray, (k, k), 0)

    if cfg.thresh_mode == "adaptive":
        block = max(3, cfg.adaptive_block | 1)
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, cfg.adaptive_C
        )
    else:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    if cfg.morph_open_iter > 0:
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_open_iter)
    if cfg.morph_close_iter > 0:
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=cfg.morph_close_iter)

    return gray, th

def find_contours(bin_img: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_annotations(img_bgr: np.ndarray, contours: List[np.ndarray], labels: List[str]) -> np.ndarray:
    out = img_bgr.copy()
    for cnt, lbl in zip(contours, labels):
        if cv2.contourArea(cnt) <= 1:
            continue
        color = (0, 255, 0)
        cv2.drawContours(out, [cnt], -1, color, 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(out, lbl, (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 255), 2, cv2.LINE_AA)
    return out

def count_by_label(labels: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

# ---------- UI ----------
st.set_page_config(page_title="Eyeballing Shapes", layout="wide")
st.title("Eyeballing: Local geometric shape detector (calibration-first)")

cfg = load_config()

with st.sidebar:
    st.header("Calibration")
    cfg.blur_kernel = st.slider("Blur kernel (odd)", 3, 25, cfg.blur_kernel, step=2)
    cfg.thresh_mode = st.selectbox("Threshold mode", ["adaptive", "otsu"], index=0 if cfg.thresh_mode=="adaptive" else 1)
    cfg.adaptive_block = st.slider("Adaptive block size (odd)", 11, 101, cfg.adaptive_block, step=2)
    cfg.adaptive_C = st.slider("Adaptive C", -20, 20, cfg.adaptive_C, step=1)
    cfg.morph_open_iter = st.slider("Morph OPEN iters", 0, 5, cfg.morph_open_iter)
    cfg.morph_close_iter = st.slider("Morph CLOSE iters", 0, 5, cfg.morph_close_iter)
    cfg.min_area = st.slider("Min area (px)", 10, 5000, cfg.min_area, step=10)
    cfg.approx_eps_pct = st.slider("Approx ε (% perimeter)", 0.005, 0.08, cfg.approx_eps_pct, step=0.005)
    cfg.circularity_circle = st.slider("Circle circularity ≥", 0.50, 0.98, cfg.circularity_circle, step=0.01)
    cfg.square_ar_tol = st.slider("Square AR tolerance ±", 0.02, 0.40, cfg.square_ar_tol, step=0.01)
    cfg.rect_angle_tol_deg = st.slider("Rect angle tol (°)", 5.0, 30.0, cfg.rect_angle_tol_deg, step=1.0)
    cfg.star_defects_ratio = st.slider("Star defects ratio ≥", 0.0, 0.2, cfg.star_defects_ratio, step=0.005)

    col_s1, col_s2 = st.columns(2)
    if col_s1.button("Save config"):
        save_config(cfg)
        st.success("Config saved.")
    if col_s2.button("Reset to defaults"):
        cfg = EyeballConfig()
        st.experimental_rerun()

st.subheader("Input")
use_webcam = st.toggle("Use webcam")
uploaded = None
frame = None

if use_webcam:
    cam = st.camera_input("Capture a frame")
    if cam is not None:
        uploaded = cam
else:
    uploaded = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    gray, th = preprocess(img_bgr, cfg)
    contours = [c for c in find_contours(th) if cv2.contourArea(c) >= cfg.min_area]
    labels = [classify_contour(c, cfg) for c in contours]
    annotated = draw_annotations(img_bgr, contours, labels)
    counts = count_by_label(labels)

    c1, c2 = st.columns(2)
    with c1:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated")
    with c2:
        st.image(th, clamp=True, caption="Binary mask (post-morphology)")

    st.write("### Counts")
    st.json(counts)

    # Export results
    ok, buf = cv2.imencode(".png", annotated)
    if ok:
        st.download_button("Download annotated PNG", data=buf.tobytes(), file_name="annotated.png", mime="image/png")

    # Simple explainability per contour
    with st.expander("Per-object debug table"):
        rows = []
        for i, c in enumerate(contours):
            perim = cv2.arcLength(c, True)
            area = cv2.contourArea(c)
            circ = contour_circularity(c)
            defects_r = convexity_defects_ratio(c)
            rows.append({
                "id": i,
                "label": labels[i],
                "area": round(float(area), 2),
                "perimeter": round(float(perim), 2),
                "circularity": round(float(circ), 3),
                "defects_ratio": round(float(defects_r), 4),
                "sides(approx)": int(len(cv2.approxPolyDP(c, cfg.approx_eps_pct * perim, True)))
            })
        st.dataframe(rows, use_container_width=True)
else:
    st.info("Upload an image or turn on the webcam.")

st.caption("Heuristics: polygon sides via approxPolyDP, circularity=4πA/P², rectangles by ~90° corners and AR, stars by convexity defects. Tune and save—this is your eyeballing calibration.")
