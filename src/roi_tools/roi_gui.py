#!/usr/bin/env python3
"""
roi_gui.py

Interactive ROI editor to draw Zones (A,B,C) and entry/exit lines on a representative frame.

Usage (examples):
    python src/roi_tools/roi_gui.py --frame data/subset/MVI_20011/img00563.jpg --out configs/MVI_20011_config.json

Controls while window is focused:
    - Left-click: add a point to current polygon or line
    - n : finish current polygon (it will be saved as next Zone letter A, B, C...)
    - l : toggle line mode (click exactly 2 points to create a line)
    - r : reset current polygon or current line
    - u : undo last polygon or last line
    - s : save config (zones + lines) to JSON and exit
    - q or ESC : quit without saving
Notes:
  - Draw polygons in order: A (entry), B (middle), C (exit). You can add more polygons but keep same convention.
  - After saving, open JSON and update "expected_mapping" if needed.
"""
import cv2, json, argparse, os
from copy import deepcopy

WINDOW = "ROI Editor"

def save_config(out_path, seq_id, img_path, image_w, image_h, zones, lines, expected_mapping=None):
    conf = {
        "sequence_id": seq_id,
        "frame_pattern": img_path,
        "fps": 25,
        "image_width": image_w,
        "image_height": image_h,
        "homography": None,
        "coordinate_space": "image",
        "zones": zones,
        "entry_lines": lines,
        "expected_mapping": expected_mapping or {"A->C":"normal", "C->A":"wrong_way"},
        "notes": ""
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(conf, f, indent=2)
    print("Saved config:", out_path)

def draw_overlay(base_img, zones, lines, current_poly, current_line):
    img = base_img.copy()
    # draw polygons
    for k, v in zones.items():
        pts = v["points"]
        if len(pts) >= 2:
            pts_np = cv2.UMat(cv2.array(pts)).get() if False else None  # placeholder
        # draw poly lines
        for i in range(len(pts)):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i+1)%len(pts)]) if i+1 < len(pts) else None
            if p2:
                cv2.line(img, p1, p2, (0,255,0), 2)
        # label
        if pts:
            cv2.putText(img, k, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    # current polygon in progress
    if current_poly:
        for i in range(len(current_poly)-1):
            cv2.line(img, tuple(current_poly[i]), tuple(current_poly[i+1]), (0,200,255), 2)
        for p in current_poly:
            cv2.circle(img, tuple(p), 4, (0,200,255), -1)

    # draw lines
    for k,v in lines.items():
        cv2.line(img, tuple(v["p1"]), tuple(v["p2"]), (0,0,255), 2)
        cv2.putText(img, k, tuple(v["p1"]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # current line in progress
    if current_line:
        for p in current_line:
            cv2.circle(img, tuple(p), 4, (255,0,0), -1)
        if len(current_line) == 2:
            cv2.line(img, tuple(current_line[0]), tuple(current_line[1]), (255,0,0), 2)

    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="Path to representative frame image")
    ap.add_argument("--out", required=True, help="Output JSON config path")
    args = ap.parse_args()

    img_path = args.frame
    out_path = args.out
    seq_id = os.path.splitext(os.path.basename(out_path))[0].replace("_config","")
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit("Cannot open frame: " + img_path)
    h, w = img.shape[:2]

    zones = {}      # e.g. {"A": {"type":"polygon","points":[ [x,y], ... ] }, ...}
    lines = {}      # e.g. {"line_1": {"p1":[x,y],"p2":[x,y]}, ...}
    poly_in_progress = []
    line_in_progress = []
    history = {"zones":[], "lines":[]}  # for undo

    mode = "poly"  # "poly" or "line"

    def on_mouse(ev, x, y, flags, param):
        nonlocal poly_in_progress, line_in_progress
        if ev == cv2.EVENT_LBUTTONDOWN:
            if mode == "poly":
                poly_in_progress.append([int(x), int(y)])
            else:
                line_in_progress.append([int(x), int(y)])

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    print("ROI Editor controls: left-click to add points. 'n' finish polygon, 'l' toggle line mode then click two points, 'r' reset current, 'u' undo last, 's' save, 'q' quit")

    while True:
        canvas = draw_overlay(img, zones, lines, poly_in_progress, line_in_progress)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('n'):  # finish polygon
            if poly_in_progress:
                name = chr(ord('A') + len(zones))  # A, B, C...
                zones[name] = {"type":"polygon", "points": deepcopy(poly_in_progress)}
                history["zones"].append(name)
                poly_in_progress = []
                print(f"Saved polygon zone {name}")
        elif key == ord('l'):  # toggle line mode or finish line
            if line_in_progress and len(line_in_progress) == 2:
                idx = len(lines) + 1
                lines[f"line_{idx}"] = {"p1": deepcopy(line_in_progress[0]), "p2": deepcopy(line_in_progress[1])}
                history["lines"].append(f"line_{idx}")
                line_in_progress = []
                print(f"Saved line_{idx}")
            else:
                # start line mode (user should click twice)
                print("Line mode: click two points to create a line, press 'l' again to save")
        elif key == ord('r'):
            poly_in_progress = []
            line_in_progress = []
            print("Reset current polygon/line")
        elif key == ord('u'):
            # undo last zone or last line
            if history["zones"]:
                last = history["zones"].pop()
                if last in zones: del zones[last]
                print("Undid zone", last)
            elif history["lines"]:
                last = history["lines"].pop()
                if last in lines: del lines[last]
                print("Undid line", last)
            else:
                print("Nothing to undo")
        elif key == ord('s'):
            if not zones:
                print("No zones defined — cannot save empty config")
            else:
                save_config(out_path, seq_id, img_path, w, h, zones, lines)
                break
        elif key == ord('q') or key == 27:
            print("Quit without saving")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
