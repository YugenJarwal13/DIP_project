#!/usr/bin/env python3
"""
visualize_zones.py
Draws zones & lines from a saved config JSON onto a frame for quick verification.

Usage:
    python src/roi_tools/visualize_zones.py --config configs/MVI_20011_config.json --frame data/subset/MVI_20011/img00563.jpg
"""
import cv2, json, argparse
import os

def draw_from_config(img, cfg):
    # draw polygons
    for k,v in cfg.get("zones", {}).items():
        pts = v["points"]
        for i in range(len(pts)):
            p1 = tuple(pts[i]); p2 = tuple(pts[(i+1) % len(pts)]) if i+1 < len(pts) else None
            if p2: cv2.line(img, p1, p2, (0,255,0), 2)
        if pts: cv2.putText(img, k, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    # draw lines
    for k,v in cfg.get("entry_lines", {}).items():
        cv2.line(img, tuple(v["p1"]), tuple(v["p2"]), (0,0,255), 2)
        cv2.putText(img, k, tuple(v["p1"]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    return img

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frame", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    img = cv2.imread(args.frame)
    if img is None: raise SystemExit("Cannot open frame")
    out = draw_from_config(img, cfg)
    cv2.imshow("zones", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
