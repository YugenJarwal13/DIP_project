#!/usr/bin/env python3
"""
entry_exit_validator_strict.py

Stricter, conservative entry-exit validator that minimizes false positives.

Save as: src/validator/entry_exit_validator_strict.py

Usage example:
  .venv\Scripts\activate
  python src/validator/entry_exit_validator_strict.py \
    --tracks_dir outputs/tracks/MVI_20011 \
    --config configs/MVI_20011_config.json \
    --frames_dir data/subset/MVI_20011 \
    --out_dir outputs/alerts_strict/MVI_20011 \
    --disp_window 6 --min_hits 5 --min_zone_confirm 3 --dot_thresh -0.25

Notes:
- Conservative defaults chosen so DETRAC (no real wrong-way) produces no alerts.
- If you want to demo a wrong-way, follow the "simulate" steps below.
"""

import os
import json
import glob
import argparse
from pathlib import Path
from shapely.geometry import Point, Polygon
import numpy as np
import cv2
from tqdm import tqdm
import csv

# ----------------- Defaults (conservative) -----------------
DISP_WINDOW = 6         # frames to compute displacement (recent)
MIN_ZONE_CONFIRM = 3    # frames inside same zone to confirm visit
MIN_HITS = 5            # min hits/length of track to consider
DOT_THRESH = -0.25      # dot < DOT_THRESH considered opposite to expected
CONF_THRESH = 0.45      # minimum detection score to consider
# ----------------------------------------------------------

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    zones = {}
    for k in ['A','B','C']:
        pts = cfg.get('zones', {}).get(k, {}).get('points', None)
        if pts:
            zones[k] = Polygon([(p[0], p[1]) for p in pts])
        else:
            zones[k] = None
    expected = cfg.get('expected_mapping', {})  # dictionary e.g., {"A->C":"normal", "C->A":"wrong_way"}
    meta = {
        'image_width': cfg.get('image_width', None),
        'image_height': cfg.get('image_height', None),
        'fps': cfg.get('fps', None),
        'expected_mapping': expected
    }
    return zones, meta

def read_sorted_track_frames(tracks_dir):
    files = sorted(glob.glob(os.path.join(tracks_dir, "frame_*.json")))
    frames = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        try:
            num = int(base.split("_")[1])
        except:
            continue
        rec = json.load(open(f))
        frames.append((num, rec))
    frames.sort(key=lambda x: x[0])
    return frames

def centroid_from_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def bbox_is_normalized(bbox, img_w, img_h):
    # If all coords <= 1.01, treat as normalized
    x1,y1,x2,y2 = bbox
    return max(x1,y1,x2,y2) <= 1.01

def to_pixel_bbox(bbox, img_w, img_h):
    if bbox_is_normalized(bbox, img_w, img_h):
        x1,y1,x2,y2 = bbox
        return [x1*img_w, y1*img_h, x2*img_w, y2*img_h]
    else:
        return bbox

def compute_disp_vector(cent_history, window):
    if len(cent_history) < window:
        return None
    p_old = np.array(cent_history[-window])
    p_new = np.array(cent_history[-1])
    return p_new - p_old

def point_in_any_zone(pt, zones):
    # returns True if point inside union of A,B,C
    p = Point(pt)
    for z in zones.values():
        if z is None:
            continue
        try:
            if z.covers(p) or z.contains(p):
                return True
        except Exception:
            # fallback: ignore problematic polygon
            continue
    return False

def condensed_zone_sequence(zone_hist, min_confirm):
    """
    zone_hist: list of (frame_idx, zone_label_or_None)
    returns list of confirmed zone labels in order (zones that had at least min_confirm consecutive frames)
    """
    condensed = []
    for frame_idx, z in zone_hist:
        if len(condensed) == 0:
            condensed.append([z,1,frame_idx])
        else:
            if condensed[-1][0] == z:
                condensed[-1][1] += 1
                condensed[-1][2] = frame_idx
            else:
                condensed.append([z,1,frame_idx])
    confirmed = [entry[0] for entry in condensed if entry[0] is not None and entry[1] >= min_confirm]
    return confirmed, condensed

def save_alert(alert, out_dir, crop_img=None):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"alert_{alert['track_id']}_{alert['frame']}.json")
    with open(json_path, 'w') as fh:
        json.dump(alert, fh, indent=2)
    if crop_img is not None:
        crop_path = os.path.join(out_dir, f"alert_{alert['track_id']}_{alert['frame']}.jpg")
        cv2.imwrite(crop_path, crop_img)
        return json_path, crop_path
    return json_path, None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks_dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--disp_window", type=int, default=DISP_WINDOW)
    p.add_argument("--min_zone_confirm", type=int, default=MIN_ZONE_CONFIRM)
    p.add_argument("--min_hits", type=int, default=MIN_HITS)
    p.add_argument("--dot_thresh", type=float, default=DOT_THRESH)
    p.add_argument("--conf_thresh", type=float, default=CONF_THRESH)
    p.add_argument("--visualize", action="store_true")
    args = p.parse_args()

    zones, meta = load_config(args.config)
    expected_map = meta.get('expected_mapping', {})
    frames_tracks = read_sorted_track_frames(args.tracks_dir)
    if len(frames_tracks) == 0:
        print("No track frames found in", args.tracks_dir); return

    # map frame keys to image paths for crops
    # assumes images named img{frame:05d}.jpg in frames_dir
    image_map = {}
    for pth in glob.glob(os.path.join(args.frames_dir, "img*.jpg")):
        name = os.path.splitext(os.path.basename(pth))[0]  # img00001
        key = int(name.replace("img",""))
        image_map[key] = pth

    # per-track histories
    tracks_centers = {}    # tid -> list of (cx,cy)
    tracks_zone_hist = {}  # tid -> list of (frame_idx, zone)
    tracks_hits = {}       # tid -> last hits
    alerts = []
    os.makedirs(args.out_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.out_dir, "visual"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "crops"), exist_ok=True)

    # Precompute zone centroids A and C if present (for direction vector)
    zone_centers = {}
    if zones.get('A') is not None:
        zone_centers['A'] = np.array(zones['A'].centroid.coords[0], dtype=float)
    if zones.get('C') is not None:
        zone_centers['C'] = np.array(zones['C'].centroid.coords[0], dtype=float)

    # iterate frames
    for frame_idx, recs in tqdm(frames_tracks, desc="Validating frames"):
        img_path = image_map.get(frame_idx, None)
        frame_img = cv2.imread(img_path) if img_path else None
        img_h, img_w = frame_img.shape[:2] if frame_img is not None else (None, None)

        for r in recs:
            tid = int(r.get("track_id"))
            raw_bbox = r.get("bbox")  # could be normalized or pixels
            score = float(r.get("score", 0.0))
            hits = int(r.get("hits", 0))

            # basic filters
            if score < args.conf_thresh:
                continue
            if hits < args.min_hits:
                continue

            # convert bbox to pixels if necessary
            if img_w is not None and bbox_is_normalized(raw_bbox, img_w, img_h):
                bbox = to_pixel_bbox(raw_bbox, img_w, img_h)
            else:
                bbox = raw_bbox

            cx, cy = centroid_from_bbox(bbox)
            tracks_centers.setdefault(tid, []).append((cx,cy))
            # determine zone membership
            zone_label = None
            for k,zpoly in zones.items():
                if zpoly is None:
                    continue
                try:
                    if zpoly.covers(Point((cx,cy))) or zpoly.contains(Point((cx,cy))):
                        zone_label = k
                        break
                except Exception:
                    # in case polygon invalid, skip
                    zone_label = None
            tracks_zone_hist.setdefault(tid, []).append((frame_idx, zone_label))
            tracks_hits[tid] = hits

            # now attempt detection of a violation for this track if enough history exists
            # require at least min_zone_confirm*3 confirmed zones overall
            confirmed_seq, condensed = condensed_zone_sequence(tracks_zone_hist[tid], args.min_zone_confirm)
            if len(confirmed_seq) < 3:
                continue
            last3 = confirmed_seq[-3:]
            # check if last3 is reverse C->B->A
            if last3 != ['C','B','A']:
                continue

            # check expected mapping: if config explicitly marks C->A as allowed/normal, skip
            # If config has mapping where "C->A" == "wrong_way" or default behavior assumes C->A is violation, proceed
            expected_CtoA = expected_map.get("C->A", None)
            # if expected mapping present and says normal, skip alert
            if expected_CtoA is not None and expected_CtoA.lower() == "normal":
                # respect config: this route is allowed
                continue

            # displacement vector
            cent_hist = tracks_centers[tid]
            disp = compute_disp_vector(cent_hist, args.disp_window)
            if disp is None:
                # not enough motion history; skip (conservative)
                continue
            disp_norm = np.linalg.norm(disp)
            if disp_norm < 1.0:  # negligible motion threshold (pixels)
                continue

            # must be roughly aligned with C->A direction (zone_centers must exist)
            if ('C' in zone_centers) and ('A' in zone_centers):
                zone_vec = zone_centers['A'] - zone_centers['C']  # vector from C to A
                if np.linalg.norm(zone_vec) < 1e-6:
                    # degenerate, skip
                    continue
                zone_unit = zone_vec / np.linalg.norm(zone_vec)
                disp_unit = disp / disp_norm
                dot = float(np.dot(disp_unit, zone_unit))
                # For C->A violation we expect dot > 0 (movement aligned C->A). But since we earlier used negative thresholds,
                # here we check for alignment with C->A (positive dot). Use conservative threshold > 0.3
                if dot < 0.3:
                    # not strongly aligned, skip
                    continue
            else:
                # cannot compute zone direction reliably -> skip (conservative)
                continue

            # ensure centroid overlaps union of zones (vehicle evidently on road area)
            if not point_in_any_zone((cx,cy), zones):
                # not inside road zones; skip
                continue

            # passed all strict checks -> record alert
            alert = {
                "seq": Path(args.tracks_dir).stem,
                "track_id": tid,
                "frame": frame_idx,
                "zone_seq": last3,
                "bbox": [float(round(v,3)) for v in bbox],
                "centroid": [float(round(cx,3)), float(round(cy,3))],
                "score": score,
                "disp_magnitude": float(round(disp_norm,3)),
                "dot_with_CtoA": float(round(dot,3))
            }

            # crop image if available
            crop_img = None
            if frame_img is not None:
                x1,y1,x2,y2 = [int(round(v)) for v in bbox]
                h,w = frame_img.shape[:2]
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
                if x2 > x1 and y2 > y1:
                    crop_img = frame_img[y1:y2, x1:x2].copy()

            json_path, crop_path = save_alert(alert, os.path.join(args.out_dir, "alerts"), crop_img)
            alerts.append(alert)

            # optional visualization of alert frame
            if args.visualize and frame_img is not None:
                vis = frame_img.copy()
                # draw zones outlines
                for k,zpoly in zones.items():
                    if zpoly is None:
                        continue
                    try:
                        pts = np.array(list(zpoly.exterior.coords)).astype(int)
                        color = (0,255,0) if k=='A' else (0,255,255) if k=='B' else (0,0,255)
                        cv2.polylines(vis, [pts], True, color, 2)
                    except Exception:
                        pass
                # draw bbox and text
                if crop_img is not None:
                    cv2.rectangle(vis, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 3)
                cv2.putText(vis, f"WRONG-WAY ID:{tid}", (int(bbox[0]), max(20, int(bbox[1]-10))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                vis_path = os.path.join(args.out_dir, "visual", f"alert_{tid}_{frame_idx}.jpg")
                cv2.imwrite(vis_path, vis)

            # clear recent zone history for this track to avoid duplicate alerts
            tracks_zone_hist[tid] = []

    # write summary CSV
    csv_path = os.path.join(args.out_dir, "alerts_summary.csv")
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(["seq","track_id","frame","zone_seq","bbox","centroid","score","disp_magnitude","dot"])
        for a in alerts:
            writer.writerow([a['seq'], a['track_id'], a['frame'], "|".join(a['zone_seq']), a['bbox'], a['centroid'], a['score'], a['disp_magnitude'], a['dot_with_CtoA']])
    print(f"Validation finished. Alerts: {len(alerts)}. Summary: {csv_path}")

if __name__ == "__main__":
    main()
