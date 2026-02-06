# -*- coding: utf-8 -*-
"""
Refer and Pipeline Testing (VS Code / local Python version)

This script is a refactor of the original Colab notebook export:
- Removes google.colab drive mounting
- Removes notebook shell commands (e.g., !pip install ...)
- Uses local paths (relative to this file) by default
- Adds a proper main() entrypoint so it can run from VS Code/terminal

Requirements (install once):
    pip install opencv-python tensorflow torch tqdm
    pip install git+https://github.com/facebookresearch/segment-anything.git

Recommended folder structure:
project/
├─ refer_and_pipeline_testing_vscode.py
├─ models/
│  ├─ leaf_classifier.h5
│  └─ class_names.json
├─ sam_models/
│  └─ sam_vit_b_01ec64.pth
├─ data/
│  └─ test_cam/            # from google drive (.jpg/.png/.jpeg)
└─ outputs/             # sent to google drive
"""

from __future__ import annotations

import os
import json
import time
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import cv2 as cv
from tqdm import tqdm

import torch
from tensorflow.keras.models import load_model
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import argparse
from datetime import datetime

# ----------------------------
# Configuration
# ----------------------------

@dataclass
class Paths:
    base_dir: str
    model_path: str
    class_json: str
    sam_ckpt: str
    test_dir: str
    out_base: str


def default_paths(test_dir_override: Optional[str] = None) -> Paths:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # default local folder (fallback)
    local_default = os.path.join(base_dir, "data", "test_images")

    # allow override from:
    # 1) function argument
    # 2) env var LEAF_TEST_DIR
    # 3) fallback local_default
    test_dir = test_dir_override or os.environ.get("LEAF_TEST_DIR") or local_default

    return Paths(
        base_dir=base_dir,
        model_path=os.path.join(base_dir, "models", "leaf_classifier.h5"),
        class_json=os.path.join(base_dir, "models", "class_names.json"),
        sam_ckpt=os.path.join(base_dir, "sam_models", "sam_vit_b_01ec64.pth"),
        test_dir=test_dir,
        out_base=r"G:\My Drive\outputs",
    )



# ----------------------------
# Utilities
# ----------------------------
import re
from pathlib import Path

def parse_cam_date_time(fname: str):
    stem = Path(fname).stem

    # cam1_20260127_015350_764310
    m = re.match(r"(?i)^(cam\d+)[ _-]?(\d{8})[ _-]?(\d{6})", stem)
    if m:
        cam = m.group(1).lower()
        yyyymmdd = m.group(2)
        hhmmss = m.group(3)
        dt = datetime.strptime(yyyymmdd + hhmmss, "%Y%m%d%H%M%S")
        return cam, dt.strftime("%Y-%m-%d"), dt.strftime("%H%M")

    # fallback: cam1_29122026 1207 (DDMMYYYY + HHMM)
    m = re.match(r"(?i)^(cam\d+)[ _-]?(\d{8})(?:[ _-]?(\d{4}))?", stem)
    if m:
        cam = m.group(1).lower()
        ddmmyyyy = m.group(2)
        hhmm = m.group(3) or "0000"
        dt = datetime.strptime(ddmmyyyy + hhmm, "%d%m%Y%H%M")
        return cam, dt.strftime("%Y-%m-%d"), hhmm

    return "unknown", "unknown-date", None


def safe_move_file(src: str, dst_dir: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(src)
    dst = os.path.join(dst_dir, base)

    if os.path.exists(dst):
        name, ext = os.path.splitext(base)
        k = 1
        while True:
            candidate = os.path.join(dst_dir, f"{name}_dup{k}{ext}")
            if not os.path.exists(candidate):
                dst = candidate
                break
            k += 1

    shutil.move(src, dst)
    return dst


def imread_unicode(path: str) -> Optional[np.ndarray]:
    """Read image from unicode path on Windows reliably."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv.imdecode(data, cv.IMREAD_COLOR)
        return img
    except Exception:
        return None

def load_class_names(class_json: str) -> List[str]:
    with open(class_json, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # Support either dict {"0":"good","1":"hole",...} or list ["good","hole",...]
    if isinstance(class_names, dict):
        # sort by numeric key if possible
        def _key(x):
            k, _v = x
            try:
                return int(k)
            except Exception:
                return str(k)
        classes = [v for _, v in sorted(class_names.items(), key=_key)]
    elif isinstance(class_names, list):
        classes = list(class_names)
    else:
        raise ValueError(f"Unsupported class_names format in {class_json}: {type(class_names)}")

    # normalize to strings
    classes = [str(c) for c in classes]
    return classes


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clear_dir(folder: str) -> None:
    if not os.path.isdir(folder):
        return
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        try:
            if os.path.isfile(p) or os.path.islink(p):
                os.unlink(p)
            else:
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


def class_from_filename(fname: str, classes: List[str]) -> str:
    """
    Original logic from notebook:
    expected crop filename pattern: leaf_{i}_{pred}_{conf}.png
    and parse the {pred} part back to the class folder.
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if len(parts) < 4:
        return "reject"

    cls_raw = "_".join(parts[2:-1]).replace("_", " ").strip().lower()
    for c in classes:
        if c.strip().lower() == cls_raw:
            return c
    return "reject"


# ----------------------------
# Core pipeline (SAM + classify)
# ----------------------------

def make_mask_generator(
    sam_ckpt: str,
    model_type: str = "vit_b",
    points_per_side: int = 16,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.92,
    min_mask_region_area: int = 1500,
) -> Tuple[SamAutomaticMaskGenerator, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_ckpt).to(device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    return mask_generator, device


def apply_bg(crop_rgb: np.ndarray, leaf_mask: np.ndarray, bg_mode: str = "white") -> Optional[np.ndarray]:
    out = crop_rgb.copy()
    if bg_mode == "black":
        out[leaf_mask == 0] = 0
    elif bg_mode == "white":
        out[leaf_mask == 0] = 255
    elif bg_mode == "mean":
        px = crop_rgb[leaf_mask == 1]
        if px.size == 0:
            return None
        out[leaf_mask == 0] = px.mean(axis=0).astype(np.uint8)
    else:
        raise ValueError("bg_mode must be black/white/mean")
    return out


def segment_and_classify_leaves(
    img_bgr: np.ndarray,
    model,
    class_names: List[str],
    mask_generator: SamAutomaticMaskGenerator,
    out_crop_dir: Optional[str] = None,

    # default threshold (for all classes except overrides)
    conf_threshold: float = 30.0,

    # per-class threshold overrides (e.g. oxy higher)
    conf_threshold_by_class: Optional[Dict[str, float]] = None,

    max_masks: int = 15,
    min_area: int = 200,
    min_fill: float = 0.25,
    max_bbox_ratio: float = 0.25,
    pad: int = 8,
    min_leaf_like: float = 0.08,
    bg_mode: str = "white",
    debug: bool = False
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Full version with OverflowError fix (kept from notebook):
    - Force bbox/area/H/W into Python int before doing w*h and comparisons
    - Use float for bbox_area + fill to avoid dtype overflow
    - Clamp bbox to image bounds (safer)
    """

    if conf_threshold_by_class is None:
        conf_threshold_by_class = {"oxy": 50.0}  # example override (same as original)

    def get_thr(pred_name: str) -> float:
        p = str(pred_name).lower().strip()
        return float(conf_threshold_by_class.get(p, conf_threshold))

    # Keras input size
    img_size = model.input_shape[1:3]
    assert len(class_names) == model.output_shape[-1], (class_names, model.output_shape)

    def classify_crop_rgb(crop_rgb: np.ndarray):
        img = cv.resize(crop_rgb, img_size).astype(np.float32)
        x = np.expand_dims(img, axis=0)
        probs = model.predict(x, verbose=0)[0].astype(float)
        idx = int(np.argmax(probs))
        pred = class_names[idx]
        conf = float(probs[idx]) * 100.0
        return pred, conf, probs

    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    H, W = int(H), int(W)

    masks = mask_generator.generate(img_rgb)
    masks = sorted(masks, key=lambda m: int(m.get("area", 0)), reverse=True)[:max_masks]

    if out_crop_dir:
        ensure_dir(out_crop_dir)

    vis = img_bgr.copy()
    results: List[Dict[str, Any]] = []
    i = 0

    for m in masks:
        seg = m["segmentation"].astype(np.uint8)

        area = int(m.get("area", 0))
        x, y, w, h = m["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)

        # basic filters
        if area < int(min_area) or w < 30 or h < 30:
            continue

        bbox_area = float(w * h) + 1e-6
        fill = float(area) / bbox_area
        if fill < float(min_fill):
            continue

        if (w * h) > float(max_bbox_ratio) * float(H * W):
            continue

        # bbox clamp
        x1, y1, x2, y2 = x, y, x + w, y + h
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W))
        y2 = max(0, min(y2, H))
        if x2 <= x1 or y2 <= y1:
            continue

        crop_rgb = img_rgb[y1:y2, x1:x2].copy()
        crop_seg = seg[y1:y2, x1:x2]

        ys, xs = np.where(crop_seg > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        xmin = max(0, int(xs.min()) - int(pad))
        xmax = min(crop_rgb.shape[1] - 1, int(xs.max()) + int(pad))
        ymin = max(0, int(ys.min()) - int(pad))
        ymax = min(crop_rgb.shape[0] - 1, int(ys.max()) + int(pad))
        if xmax <= xmin or ymax <= ymin:
            continue

        crop_rgb = crop_rgb[ymin:ymax + 1, xmin:xmax + 1]
        crop_seg = crop_seg[ymin:ymax + 1, xmin:xmax + 1]
        leaf_mask = (crop_seg > 0).astype(np.uint8)

        masked = apply_bg(crop_rgb, leaf_mask, bg_mode=bg_mode)
        if masked is None:
            continue

        # leaf-likeness
        idx_leaf = (leaf_mask == 1)
        if idx_leaf.sum() < 50:
            continue

        bgr = cv.cvtColor(masked, cv.COLOR_RGB2BGR)
        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)

        Hh, Ss, Vv = cv.split(hsv)
        B, G, R = cv.split(bgr)

        green = cv.inRange(hsv, (25, 30, 30), (95, 255, 255))
        yellow_hsv = cv.inRange(hsv, (15, 50, 50), (45, 255, 255))
        not_too_green = (G.astype("int16") - R.astype("int16")) < 20
        not_overbright = Vv < 245

        yellow_bool = (yellow_hsv > 0) & not_too_green & not_overbright
        yellow = (yellow_bool.astype("uint8") * 255)

        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        green = cv.morphologyEx(green, cv.MORPH_OPEN, k, iterations=1)
        yellow = cv.morphologyEx(yellow, cv.MORPH_OPEN, k, iterations=1)

        leaf_like_ratio = (((green > 0) | (yellow > 0))[idx_leaf]).mean()
        if float(leaf_like_ratio) < float(min_leaf_like):
            continue

        # classify
        pred, conf, probs = classify_crop_rgb(masked)
        thr = float(get_thr(pred))
        pred_show = pred if float(conf) >= thr else "unknown"

        if debug:
            prob_dict = {class_names[j]: float(probs[j]) for j in range(len(class_names))}
            print(f"[{i}] {pred_show} {float(conf):.1f}% (thr={thr:.1f}%) | raw={pred} | {prob_dict}")

        # draw
        cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.putText(
            vis,
            f"{pred_show} ({float(conf):.1f}%)",
            (x1, max(20, y1 - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        # save crop
        if out_crop_dir:
            fn = f"leaf_{i}_{pred_show}_{float(conf):.1f}.png"
            cv.imwrite(os.path.join(out_crop_dir, fn), cv.cvtColor(masked, cv.COLOR_RGB2BGR))

        results.append({
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "pred_raw": str(pred),
            "pred": str(pred_show),
            "conf": float(conf),
            "thr": float(thr),
            "leaf_like_ratio": float(leaf_like_ratio),
            "prob": {class_names[j]: float(probs[j]) for j in range(len(class_names))}
        })
        i += 1

    return vis, results


# ----------------------------
# Batch processing (same behavior as notebook end section)
# ----------------------------

def run_batch_split_crops_by_filename(
    img_dir: str,
    out_base: str,
    model,
    class_names: List[str],
    mask_generator: SamAutomaticMaskGenerator,
    conf_th: float = 30.0,
    debug: bool = False,
    processed_root: Optional[str] = None,
) -> None:
    ensure_dir(out_base)

    # output folders
    #for c in class_names + ["reject"]:
    #    ensure_dir(os.path.join(out_base, c))

    tmp_dir = os.path.join(out_base, "_tmp")
    ensure_dir(tmp_dir)

    # gather images
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"TEST_DIR not found or not a folder: {img_dir}")

    all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    total = len(all_imgs)
    print("sample:", total)
    if total == 0:
        print("No images found. Put images in:", img_dir)
        return

    t0 = time.time()

    for idx, fname in enumerate(tqdm(all_imgs, total=total, unit="img"), start=1):
        cam, date_str, hhmm = parse_cam_date_time(fname) # parse cam / date / time from source firle
        img_path = os.path.join(img_dir, fname) 
        img = imread_unicode(img_path)
        if img is None:
            continue

        clear_dir(tmp_dir)

        _vis, _results = segment_and_classify_leaves(
            img_bgr=img,
            model=model,
            class_names=class_names,
            mask_generator=mask_generator,
            out_crop_dir=tmp_dir,
            conf_threshold=conf_th,
            debug=debug,
        )

        # move crops to: outputs/<date>/<cam>/<class>/
        crops = [f for f in os.listdir(tmp_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for cf in crops:
            src = os.path.join(tmp_dir, cf)
            cls = class_from_filename(cf, class_names)
            dst_dir = os.path.join(out_base, date_str, cam, cls)
            ensure_dir(dst_dir)
            if hhmm:
                new_name = f"{cam}_{date_str}_{hhmm}__{cf}"
            else:
                new_name = f"{cam}_{date_str}__{cf}"

            dst = os.path.join(dst_dir, new_name)
            if os.path.exists(dst):
                base, ext = os.path.splitext(new_name)
                dst = os.path.join(dst_dir, f"{base}_dup{idx}{ext}")

            shutil.move(src, dst)
        
            # after success: move original to processed_root (avoid re-processing next run)
            if processed_root:
                dst_dir = os.path.join(processed_root, date_str, cam)
                moved = safe_move_file(img_path, dst_dir)
                if debug:
                    print("moved original ->", moved)


        # ETA every 10 images (same spirit as notebook)
        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - t0
            avg_per_img = elapsed / idx
            remaining = (total - idx) * avg_per_img

            def fmt(sec: float) -> str:
                sec = int(sec)
                h = sec // 3600
                m = (sec % 3600) // 60
                s = sec % 60
                return f"{h:02d}:{m:02d}:{s:02d}"

            print(f"✔ {idx}/{total} ({idx/total*100:.1f}%) | elapsed {fmt(elapsed)} | ETA {fmt(remaining)} | {avg_per_img:.2f}s/img")

    print("DONE: split crops by filename class into:", out_base)


# ----------------------------
# Entrypoint
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Folder containing test images (.jpg/.png). "
             "Can be Google Drive for Desktop path, e.g. G:\\My Drive\\test_cam"
    )

    ap.add_argument(
        "--processed_dir",
        type=str,
        default=None,
        help="(Optional) Folder to move original images after processing "
             "to avoid re-processing. e.g. G:\\My Drive\\test_cam_processed"
    )

    args = ap.parse_args()

    p = default_paths(test_dir_override=args.test_dir)

    print("BASE_DIR :", p.base_dir)
    print("MODEL    :", p.model_path)
    print("CLASSES  :", p.class_json)
    print("SAM_CKPT :", p.sam_ckpt)
    print("TEST_DIR :", p.test_dir)
    print("OUT_BASE :", p.out_base)

    # quick checks
    for label, path in [
        ("MODEL_PATH", p.model_path),
        ("CLASS_JSON", p.class_json),
        ("SAM_CKPT", p.sam_ckpt),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # test dir check (friendly message)
    if not os.path.isdir(p.test_dir):
        raise FileNotFoundError(
            "TEST_DIR not found or not a folder:\n"
            f"  {p.test_dir}\n\n"
            "Fix:\n"
            "1) If images are on Google Drive, use Google Drive for Desktop and pass a real local path, e.g.\n"
            '   python refer_and_pipeline_testing_vscode.py --test_dir "G:\\My Drive\\<your_folder>"\n'
            "2) Or copy images to local folder:\n"
            f"   {os.path.join(p.base_dir, 'data', 'test_images')}\n"
        )

    class_names = load_class_names(p.class_json)
    model = load_model(p.model_path)

    print("class_names:", class_names)
    print("model.input_shape:", model.input_shape)
    print("model.output_shape:", model.output_shape)

    mask_generator, device = make_mask_generator(p.sam_ckpt, model_type="vit_b")
    print("SAM ready on", device)

    run_batch_split_crops_by_filename(
        img_dir=p.test_dir,
        out_base=p.out_base,
        model=model,
        class_names=class_names,
        mask_generator=mask_generator,
        conf_th=30.0,
        debug=False,
        processed_root=args.processed_dir
    )


if __name__ == "__main__":
    main()
