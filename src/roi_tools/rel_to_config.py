#!/usr/bin/env python3
"""
rel_to_config.py
Creates config JSON stubs from RELATIVE zones (fractions). Edit RELATIVE_ZONES inside this file
to match approximate areas for your sequences, then run to produce pixel-based configs in configs/.
"""
import os, json
from PIL import Image

# Edit relative polygons here [x_frac, y_frac] in 0..1
RELATIVE_ZONES = {
  "MVI_20011": {
    "A": [[0.03,0.78],[0.35,0.78],[0.35,0.60],[0.03,0.60]],
    "B": [[0.36,0.78],[0.65,0.78],[0.65,0.60],[0.36,0.60]],
    "C": [[0.66,0.78],[0.97,0.78],[0.97,0.60],[0.66,0.60]],
    "entry_lines": [[0.035,0.75,0.34,0.75]],
    "expected_mapping": {"A->C":"normal", "C->A":"wrong_way"}
  },
  # Add other sequences similarly...
}

FRAME_ROOT = "data/subset"
OUT_DIR = "configs"

def rel2pix(poly,w,h):
    return [[int(x*w), int(y*h)] for x,y in poly]

os.makedirs(OUT_DIR, exist_ok=True)
for seq, cfg in RELATIVE_ZONES.items():
    seq_dir = os.path.join(FRAME_ROOT, seq)
    if not os.path.isdir(seq_dir):
        print("Skip (no frames):", seq); continue
    # pick the first image in the seq folder
    files = [f for f in os.listdir(seq_dir) if f.lower().endswith(('.jpg','.png'))]
    if not files:
        print("No files in", seq); continue
    img_path = os.path.join(seq_dir, sorted(files)[0])
    w,h = Image.open(img_path).size
    zones = {}
    for zname, poly in cfg.items():
        if zname == "entry_lines" or zname == "expected_mapping": continue
        zones[zname] = {"type":"polygon", "points": rel2pix(poly, w, h)}
    lines = {}
    for i, ln in enumerate(cfg.get("entry_lines", [])):
        x1,y1,x2,y2 = ln
        lines[f"line_{i+1}"] = {"p1":[int(x1*w),int(y1*h)], "p2":[int(x2*w),int(y2*h)]}
    out = {
        "sequence_id": seq,
        "frame_pattern": img_path,
        "fps": 25,
        "image_width": w,
        "image_height": h,
        "homography": None,
        "coordinate_space": "image",
        "zones": zones,
        "entry_lines": lines,
        "expected_mapping": cfg.get("expected_mapping", {"A->C":"normal","C->A":"wrong_way"})
    }
    out_path = os.path.join(OUT_DIR, f"{seq}_config.json")
    with open(out_path,"w") as f:
        json.dump(out, f, indent=2)
    print("Wrote stub:", out_path)
