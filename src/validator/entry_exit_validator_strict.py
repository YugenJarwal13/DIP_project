#!/usr/bin/env python3
"""
entry_exit_validator_strict.py

Stricter, conservative entry-exit validator that minimizes false positives.

We raise a wrong-way alert for a track only if all these are true:

0. The track has a real detection in this frame (time_since_update == 0).
   Predicted-only tracks (time_since_update > 0) are skipped to avoid false positives
   from drifting Kalman predictions without actual YOLO detections.

A. The track has confirmed visits to three zones in the reverse order (i.e. C → B → A) 
   with each zone occupied for at least MIN_ZONE_CONFIRM frames.

B. The expected mapping in the sequence config indicates that A → C is the normal direction 
   (or config explicitly marks C->A as wrong_way).

C. The recent centroid displacement over DISP_WINDOW frames is significantly aligned with C→A 
   (dot product above threshold). This prevents spurious alerts due to noise.

D. The bounding box centroid overlaps the union of zones (so the vehicle is on-road, not in empty area).

E. The track length and hits exceed conservative minimums (MIN_HITS) to avoid very short/false tracks.

Usage example:
  .venv\Scripts\activate
  python src/validator/entry_exit_validator_strict.py \
    --tracks_dir outputs/tracks/MVI_20011 \
    --config configs/MVI_20011_config.json \
    --frames_dir data/subset/MVI_20011 \
    --out_dir outputs/alerts_strict/MVI_20011 \
    --visualize
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
DOT_THRESH = 0.3        # dot > DOT_THRESH considered aligned with C->A (wrong direction)
CONF_THRESH = 0.45      # minimum detection score to consider
MIN_DISPLACEMENT = 1.0  # minimum displacement magnitude (pixels) to consider
IOU_DET_THRESH = 0.2    # minimum IoU with detection bbox to confirm alert (NEW)
# ----------------------------------------------------------

def load_config(cfg_path):
    """Load ROI config with zones A, B, C and expected_mapping."""
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    zones = {}
    for k in ['A','B','C']:
        pts = cfg.get('zones', {}).get(k, {}).get('points', None)
        if pts:
            zones[k] = Polygon([(p[0], p[1]) for p in pts])
        else:
            zones[k] = None
    expected = cfg.get('expected_mapping', {})  # e.g., {"A->C":"normal", "C->A":"wrong_way"}
    meta = {
        'image_width': cfg.get('image_width', None),
        'image_height': cfg.get('image_height', None),
        'fps': cfg.get('fps', None),
        'expected_mapping': expected
    }
    return zones, meta

def read_sorted_track_frames(tracks_dir):
    """Read all frame_*.json files from tracks directory."""
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
    """Calculate centroid from bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def bbox_is_normalized(bbox, img_w, img_h):
    """Check if bbox coordinates are normalized (0-1 range)."""
    x1, y1, x2, y2 = bbox
    return max(x1, y1, x2, y2) <= 1.01

def to_pixel_bbox(bbox, img_w, img_h):
    """Convert normalized bbox to pixel coordinates."""
    if bbox_is_normalized(bbox, img_w, img_h):
        x1, y1, x2, y2 = bbox
        return [x1*img_w, y1*img_h, x2*img_w, y2*img_h]
    else:
        return bbox

def compute_iou(bbox1, bbox2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] format
    
    Returns:
        float: IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Compute intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union

def yolo_to_bbox(x_center, y_center, w, h, img_w, img_h):
    """
    Convert YOLO normalized format to absolute pixel bbox.
    
    Args:
        x_center, y_center, w, h: normalized (0-1) YOLO format
        img_w, img_h: image dimensions in pixels
    
    Returns:
        [x1, y1, x2, y2]: absolute pixel coordinates
    """
    x1 = (x_center - w/2) * img_w
    y1 = (y_center - h/2) * img_h
    x2 = (x_center + w/2) * img_w
    y2 = (y_center + h/2) * img_h
    return [x1, y1, x2, y2]

def load_detections_for_frame(frame_num, sequence_name, img_w, img_h):
    """
    Load YOLO detections for a specific frame.
    
    Tries multiple paths:
    1. outputs/detections_txt/{sequence}_det/labels_abs/img{num}.txt (absolute coords)
    2. outputs/detections_txt/{sequence}_det/labels/img{num}.txt (normalized)
    
    Args:
        frame_num: frame number (e.g., 283)
        sequence_name: sequence name (e.g., "MVI_20032")
        img_w, img_h: image dimensions for normalization conversion
    
    Returns:
        list of [x1, y1, x2, y2] bboxes in absolute pixel coordinates
    """
    frame_str = f"img{frame_num:05d}.txt"
    detections = []
    
    # Try absolute format first
    abs_path = os.path.join("outputs", "detections_txt", f"{sequence_name}_det", "labels_abs", frame_str)
    if os.path.exists(abs_path):
        try:
            with open(abs_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        # Format: x1 y1 x2 y2 [confidence] [class]
                        x1, y1, x2, y2 = map(float, parts[:4])
                        detections.append([x1, y1, x2, y2])
            return detections
        except Exception:
            pass
    
    # Try normalized format
    norm_path = os.path.join("outputs", "detections_txt", f"{sequence_name}_det", "labels", frame_str)
    if os.path.exists(norm_path):
        try:
            with open(norm_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Format: class x_center y_center width height
                        cls, x_c, y_c, w, h = map(float, parts[:5])
                        bbox = yolo_to_bbox(x_c, y_c, w, h, img_w, img_h)
                        detections.append(bbox)
            return detections
        except Exception:
            pass
    
    return detections

def has_detection_overlap(track_bbox, detections, iou_thresh):
    """
    Check if track bbox has sufficient overlap with at least one detection.
    
    Args:
        track_bbox: [x1, y1, x2, y2] track bounding box
        detections: list of [x1, y1, x2, y2] detection bboxes
        iou_thresh: minimum IoU threshold (e.g., 0.2)
    
    Returns:
        tuple: (has_overlap: bool, max_iou: float)
    """
    if not detections:
        return False, 0.0
    
    max_iou = 0.0
    for det_bbox in detections:
        iou = compute_iou(track_bbox, det_bbox)
        max_iou = max(max_iou, iou)
        if iou >= iou_thresh:
            return True, iou
    
    return False, max_iou

def compute_disp_vector(cent_history, window):
    """Compute displacement vector from recent centroid history."""
    if len(cent_history) < window:
        return None
    p_old = np.array(cent_history[-window])
    p_new = np.array(cent_history[-1])
    return p_new - p_old

def point_in_any_zone(pt, zones):
    """Check if point is inside union of zones A, B, C."""
    p = Point(pt)
    for z in zones.values():
        if z is None:
            continue
        try:
            if z.covers(p) or z.contains(p):
                return True
        except Exception:
            continue
    return False

def condensed_zone_sequence(zone_hist, min_confirm, gap_tolerance=2):
    """
    Condense zone history into confirmed zone visits with gap tolerance.
    Only zones occupied for at least min_confirm consecutive frames (allowing brief gaps) are considered confirmed.
    
    Args:
        zone_hist: list of (frame_idx, zone) tuples
        min_confirm: minimum frames in zone to confirm visit
        gap_tolerance: maximum consecutive None frames to bridge within same zone (default: 2)
    
    Returns:
        confirmed: list of confirmed zone labels in order
        condensed: detailed list of [zone, count, last_frame]
    """
    if not zone_hist:
        return [], []
    
    # First pass: merge zones separated by short None gaps
    merged = []
    i = 0
    while i < len(zone_hist):
        frame_idx, zone = zone_hist[i]
        
        if zone is None:
            # Count consecutive None frames
            none_count = 0
            j = i
            while j < len(zone_hist) and zone_hist[j][1] is None:
                none_count += 1
                j += 1
            
            # Check if this None gap is short enough to bridge
            if (i > 0 and j < len(zone_hist) and 
                none_count <= gap_tolerance and
                merged and merged[-1][1] == zone_hist[j][1]):
                # Bridge the gap - keep previous zone
                for k in range(i, j):
                    merged.append((zone_hist[k][0], merged[-1][1]))
                i = j
            else:
                # Gap too large or at boundary, keep None
                merged.append((frame_idx, None))
                i += 1
        else:
            merged.append((frame_idx, zone))
            i += 1
    
    # Second pass: condense into runs
    condensed = []
    for frame_idx, z in merged:
        if len(condensed) == 0:
            condensed.append([z, 1, frame_idx])
        else:
            if condensed[-1][0] == z:
                condensed[-1][1] += 1
                condensed[-1][2] = frame_idx
            else:
                condensed.append([z, 1, frame_idx])
    
    # Filter for confirmed zones (at least min_confirm frames)
    confirmed = [entry[0] for entry in condensed 
                 if entry[0] is not None and entry[1] >= min_confirm]
    return confirmed, condensed

def save_alert(alert, out_dir, crop_img=None):
    """Save alert JSON and optional cropped image."""
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
    p = argparse.ArgumentParser(description="Strict wrong-way violation detector")
    p.add_argument("--tracks_dir", required=True, help="Directory with track JSONs")
    p.add_argument("--config", required=True, help="ROI config JSON file")
    p.add_argument("--frames_dir", required=True, help="Directory with frame images")
    p.add_argument("--out_dir", required=True, help="Output directory for alerts")
    p.add_argument("--disp_window", type=int, default=DISP_WINDOW, 
                   help=f"Window for displacement calculation (default: {DISP_WINDOW})")
    p.add_argument("--min_zone_confirm", type=int, default=MIN_ZONE_CONFIRM,
                   help=f"Min frames in zone to confirm visit (default: {MIN_ZONE_CONFIRM})")
    p.add_argument("--min_hits", type=int, default=MIN_HITS,
                   help=f"Min track hits to consider (default: {MIN_HITS})")
    p.add_argument("--dot_thresh", type=float, default=DOT_THRESH,
                   help=f"Dot product threshold for C->A alignment (default: {DOT_THRESH})")
    p.add_argument("--conf_thresh", type=float, default=CONF_THRESH,
                   help=f"Min detection confidence (default: {CONF_THRESH})")
    p.add_argument("--iou_det_thresh", type=float, default=IOU_DET_THRESH,
                   help=f"Min IoU with detection to confirm alert (default: {IOU_DET_THRESH})")
    p.add_argument("--gap_tolerance", type=int, default=2,
                   help="Max consecutive None frames to bridge within same zone (default: 2)")
    p.add_argument("--visualize", action="store_true", help="Generate visualization images")
    args = p.parse_args()

    print(f"\n{'='*60}")
    print("STRICT WRONG-WAY VIOLATION DETECTOR (Enhanced v2.2)")
    print(f"{'='*60}")
    print(f"Tracks: {args.tracks_dir}")
    print(f"Config: {args.config}")
    print(f"Output: {args.out_dir}")
    print(f"\nParameters:")
    print(f"  - Displacement window: {args.disp_window} frames")
    print(f"  - Min zone confirmation: {args.min_zone_confirm} frames")
    print(f"  - Zone gap tolerance: {args.gap_tolerance} frames")
    print(f"  - Min track hits: {args.min_hits}")
    print(f"  - Dot threshold (C->A alignment): {args.dot_thresh}")
    print(f"  - Confidence threshold: {args.conf_thresh}")
    print(f"  - IoU detection threshold: {args.iou_det_thresh}")
    print(f"{'='*60}\n")

    # Load configuration
    zones, meta = load_config(args.config)
    expected_map = meta.get('expected_mapping', {})
    img_w = meta.get('image_width', None)
    img_h = meta.get('image_height', None)
    
    # Extract sequence name from tracks_dir
    sequence_name = Path(args.tracks_dir).stem  # e.g., "MVI_20032"
    
    # Read track frames
    frames_tracks = read_sorted_track_frames(args.tracks_dir)
    if len(frames_tracks) == 0:
        print(f"❌ No track frames found in {args.tracks_dir}")
        return

    print(f"✅ Loaded {len(frames_tracks)} track frames")
    print(f"✅ Sequence: {sequence_name}")
    print(f"✅ Image dimensions: {img_w}x{img_h}")

    # Map frame numbers to image paths
    image_map = {}
    for pth in glob.glob(os.path.join(args.frames_dir, "img*.jpg")):
        name = os.path.splitext(os.path.basename(pth))[0]
        key = int(name.replace("img", ""))
        image_map[key] = pth

    # Per-track histories
    tracks_centers = {}     # tid -> list of (cx, cy)
    tracks_zone_hist = {}   # tid -> list of (frame_idx, zone)
    tracks_hits = {}        # tid -> last hits value
    alerts = []

    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.out_dir, "visual"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "crops"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "alerts"), exist_ok=True)
    
    # Open debug log file
    debug_log_path = os.path.join(args.out_dir, "debug_log.txt")
    debug_log = open(debug_log_path, 'w')
    debug_log.write(f"STRICT WRONG-WAY VALIDATOR DEBUG LOG\n")
    debug_log.write(f"Sequence: {sequence_name}\n")
    from datetime import datetime
    debug_log.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    debug_log.write(f"={'='*80}\n\n")

    # Precompute zone centroids for A and C (for direction vector)
    zone_centers = {}
    if zones.get('A') is not None:
        zone_centers['A'] = np.array(zones['A'].centroid.coords[0], dtype=float)
    if zones.get('C') is not None:
        zone_centers['C'] = np.array(zones['C'].centroid.coords[0], dtype=float)

    # Statistics counters
    skipped_predicted = 0
    skipped_no_detection = 0
    skipped_condition_e = 0
    skipped_condition_a = 0
    skipped_condition_b = 0
    skipped_condition_c = 0
    skipped_condition_d = 0
    total_tracks_processed = 0
    
    # Cache for detections per frame
    detection_cache = {}

    # Iterate through all frames
    for frame_idx, recs in tqdm(frames_tracks, desc="Validating frames"):
        img_path = image_map.get(frame_idx, None)
        frame_img = cv2.imread(img_path) if img_path else None
        frame_img_h, frame_img_w = frame_img.shape[:2] if frame_img is not None else (img_h, img_w)
        
        # Load detections for this frame (with caching)
        if frame_idx not in detection_cache:
            detection_cache[frame_idx] = load_detections_for_frame(
                frame_idx, sequence_name, frame_img_w or img_w, frame_img_h or img_h
            )
        frame_detections = detection_cache[frame_idx]

        for r in recs:
            total_tracks_processed += 1
            tid = int(r.get("track_id"))
            raw_bbox = r.get("bbox")
            score = float(r.get("score", 0.0))
            hits = int(r.get("hits", 0))
            time_since_update = int(r.get("time_since_update", 0))

            # ============ FILTER 0: Skip predicted-only tracks (no real detection) ============
            # When time_since_update > 0, the track position is from Kalman prediction only,
            # not matched to any real YOLO detection. These must be ignored for validation
            # to avoid false positives from drifting predicted boxes.
            if time_since_update > 0:
                skipped_predicted += 1
                debug_log.write(f"[Frame {frame_idx}] Track {tid}: SKIPPED_PREDICTION - time_since_update={time_since_update}\n")
                continue

            # ============ CONDITION E: Track quality check ============
            if score < args.conf_thresh:
                skipped_condition_e += 1
                debug_log.write(f"[Frame {frame_idx}] Track {tid}: SKIPPED_CONDITION_E - score={score:.3f} < {args.conf_thresh}\n")
                continue
            if hits < args.min_hits:
                skipped_condition_e += 1
                debug_log.write(f"[Frame {frame_idx}] Track {tid}: SKIPPED_CONDITION_E - hits={hits} < {args.min_hits}\n")
                continue

            # Convert bbox to pixels if necessary
            if frame_img_w is not None and bbox_is_normalized(raw_bbox, frame_img_w, frame_img_h):
                bbox = to_pixel_bbox(raw_bbox, frame_img_w, frame_img_h)
            else:
                bbox = raw_bbox

            cx, cy = centroid_from_bbox(bbox)
            tracks_centers.setdefault(tid, []).append((cx, cy))

            # Determine current zone membership
            zone_label = None
            for k, zpoly in zones.items():
                if zpoly is None:
                    continue
                try:
                    if zpoly.covers(Point((cx, cy))) or zpoly.contains(Point((cx, cy))):
                        zone_label = k
                        break
                except Exception:
                    zone_label = None

            tracks_zone_hist.setdefault(tid, []).append((frame_idx, zone_label))
            tracks_hits[tid] = hits

            # ============ CONDITION A: Check zone sequence C->B->A ============
            confirmed_seq, condensed = condensed_zone_sequence(
                tracks_zone_hist[tid], args.min_zone_confirm, args.gap_tolerance
            )
            
            if len(confirmed_seq) < 3:
                skipped_condition_a += 1
                debug_log.write(f"[Frame {frame_idx}] Track {tid}: SKIPPED_CONDITION_A - confirmed_seq_len={len(confirmed_seq)} < 3\n")
                continue
            
            last3 = confirmed_seq[-3:]
            if last3 != ['C', 'B', 'A']:
                skipped_condition_a += 1
                debug_log.write(f"[Frame {frame_idx}] Track {tid}: SKIPPED_CONDITION_A - sequence={'->'.join(last3)} != C->B->A\n")
                continue

            # ============ CONDITION B: Check expected mapping ============
            expected_CtoA = expected_map.get("C->A", None)
            if expected_CtoA is not None and expected_CtoA.lower() == "normal":
                # Config explicitly allows C->A, skip alert
                skipped_condition_b += 1
                debug_log.write(f"[Frame {frame_idx}] Track {tid}: SKIPPED_CONDITION_B - C->A is normal direction\n")
                continue

            # ============ CONDITION C: Displacement vector alignment ============
            cent_hist = tracks_centers[tid]
            disp = compute_disp_vector(cent_hist, args.disp_window)
            
            if disp is None:
                # Not enough motion history
                skipped_condition_c += 1
                debug_log.write(f"[Frame {frame_idx}] Track {tid}: SKIPPED_CONDITION_C - insufficient history for displacement\n")
                continue
            
            disp_norm = np.linalg.norm(disp)
            if disp_norm < MIN_DISPLACEMENT:
                # Negligible motion
                skipped_condition_c += 1
                debug_log.write(f"[Frame {frame_idx}] Track {tid}: SKIPPED_CONDITION_C - disp_mag={disp_norm:.2f} < {MIN_DISPLACEMENT}\n")
                continue

            # Check alignment with C->A direction
            if ('C' in zone_centers) and ('A' in zone_centers):
                zone_vec = zone_centers['A'] - zone_centers['C']  # C to A vector
                if np.linalg.norm(zone_vec) < 1e-6:
                    skipped_condition_c += 1
                    debug_log.write(f"[Frame {frame_idx}] Track {tid}: SKIPPED_CONDITION_C - zone vector too small\n")
                    continue
                
                zone_unit = zone_vec / np.linalg.norm(zone_vec)
                disp_unit = disp / disp_norm
                dot = float(np.dot(disp_unit, zone_unit))
                
                # For C->A violation, dot should be positive (aligned with C->A)
                if dot < args.dot_thresh:
                    skipped_condition_c += 1
                    debug_log.write(f"[Frame {frame_idx}] Track {tid}: SKIPPED_CONDITION_C - dot={dot:.3f} < {args.dot_thresh}\n")
                    continue
            else:
                # Cannot compute zone direction, skip
                skipped_condition_c += 1
                debug_log.write(f"[Frame {frame_idx}] Track {tid}: SKIPPED_CONDITION_C - zone centroids missing\n")
                continue

            # ============ CONDITION D: Centroid in zone union ============
            if not point_in_any_zone((cx, cy), zones):
                # Not inside road zones, skip
                skipped_condition_d += 1
                debug_log.write(f"[Frame {frame_idx}] Track {tid}: SKIPPED_CONDITION_D - centroid not in any zone\n")
                continue

            # ============ DETECTION OVERLAP CHECK (FINAL GATE) ============
            # Verify track bbox has real YOLO detection evidence
            has_overlap, max_iou = has_detection_overlap(bbox, frame_detections, args.iou_det_thresh)
            if not has_overlap:
                skipped_no_detection += 1
                debug_log.write(f"[Frame {frame_idx}] Track {tid}: SKIPPED_NO_DETECTION_OVERLAP - max_iou={max_iou:.3f} < {args.iou_det_thresh} (detections: {len(frame_detections)})\n")
                continue

            # ============ ALL CONDITIONS MET: RAISE ALERT ============
            debug_log.write(f"[Frame {frame_idx}] Track {tid}: ALERT_SAVED - sequence={'->'.join(last3)}, dot={dot:.3f}, max_iou={max_iou:.3f}, hits={hits}\n")
            debug_log.flush()  # Ensure immediate write
            
            alert = {
                "seq": Path(args.tracks_dir).stem,
                "track_id": tid,
                "frame": frame_idx,
                "zone_seq": last3,
                "bbox": [float(round(v, 3)) for v in bbox],
                "centroid": [float(round(cx, 3)), float(round(cy, 3))],
                "score": score,
                "disp_magnitude": float(round(disp_norm, 3)),
                "dot_with_CtoA": float(round(dot, 3)),
                "hits": hits,
                "max_detection_iou": float(round(max_iou, 3)),
                "num_detections": len(frame_detections),
                "condensed_history": [[z, cnt, fr] for z, cnt, fr in condensed if z is not None]
            }

            # Crop vehicle image
            crop_img = None
            if frame_img is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                h, w = frame_img.shape[:2]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w-1, x2); y2 = min(h-1, y2)
                if x2 > x1 and y2 > y1:
                    crop_img = frame_img[y1:y2, x1:x2].copy()

            # Save alert
            json_path, crop_path = save_alert(
                alert, 
                os.path.join(args.out_dir, "alerts"), 
                crop_img
            )
            alerts.append(alert)

            # Visualization
            if args.visualize and frame_img is not None:
                vis = frame_img.copy()
                
                # Draw zone polygons
                for k, zpoly in zones.items():
                    if zpoly is None:
                        continue
                    try:
                        pts = np.array(list(zpoly.exterior.coords)).astype(int)
                        if k == 'A':
                            color = (0, 255, 0)  # Green
                        elif k == 'B':
                            color = (0, 255, 255)  # Yellow
                        else:  # C
                            color = (0, 0, 255)  # Red
                        cv2.polylines(vis, [pts], True, color, 2)
                        # Add zone label
                        centroid = zpoly.centroid
                        cv2.putText(vis, f"Zone {k}", 
                                  (int(centroid.x), int(centroid.y)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                    except Exception:
                        pass
                
                # Draw violation bbox and text
                cv2.rectangle(vis, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            (0, 0, 255), 3)
                
                # Add text with background
                text = f"WRONG-WAY ID:{tid}"
                text_pos = (int(bbox[0]), max(30, int(bbox[1]-10)))
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(vis, 
                            (text_pos[0]-5, text_pos[1]-text_h-5),
                            (text_pos[0]+text_w+5, text_pos[1]+5),
                            (0, 0, 255), -1)
                cv2.putText(vis, text, text_pos,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Add info text
                info_text = f"Frame: {frame_idx} | Dot: {dot:.2f} | Disp: {disp_norm:.1f}px"
                cv2.putText(vis, info_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                vis_path = os.path.join(args.out_dir, "visual", 
                                       f"alert_{tid}_{frame_idx}.jpg")
                cv2.imwrite(vis_path, vis)

            # Clear zone history to avoid duplicate alerts
            tracks_zone_hist[tid] = []

    # Write summary CSV
    csv_path = os.path.join(args.out_dir, "alerts_summary.csv")
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(["seq", "track_id", "frame", "zone_seq", "bbox", 
                        "centroid", "score", "hits", "disp_magnitude", "dot_with_CtoA"])
        for a in alerts:
            writer.writerow([
                a['seq'], 
                a['track_id'], 
                a['frame'], 
                "->".join(a['zone_seq']),
                a['bbox'], 
                a['centroid'], 
                a['score'], 
                a['hits'],
                a['disp_magnitude'], 
                a['dot_with_CtoA']
            ])

    print(f"\n{'='*60}")
    print(f"✅ Validation Complete!")
    print(f"{'='*60}")
    print(f"� Processing Statistics:")
    print(f"   Total track instances: {total_tracks_processed}")
    print(f"   Predicted-only (skipped): {skipped_predicted} ({skipped_predicted/total_tracks_processed*100:.1f}%)")
    print(f"   Validated instances: {total_tracks_processed - skipped_predicted}")
    print(f"\n�🚨 Total Alerts: {len(alerts)}")
    print(f"📄 Summary CSV: {csv_path}")
    print(f"📁 Alert Details: {os.path.join(args.out_dir, 'alerts')}")
    if args.visualize:
        print(f"🖼️  Visualizations: {os.path.join(args.out_dir, 'visual')}")
    print(f"{'='*60}\n")

    # Print alert details if any
    if alerts:
        print("\n🚨 DETECTED VIOLATIONS:")
        print(f"{'Track ID':<10} {'Frame':<8} {'Zone Seq':<12} {'Dot':<8} {'Disp (px)':<10}")
        print("-" * 60)
        for a in alerts:
            print(f"{a['track_id']:<10} {a['frame']:<8} {'->'.join(a['zone_seq']):<12} "
                  f"{a['dot_with_CtoA']:<8.3f} {a['disp_magnitude']:<10.1f}")
        print()

if __name__ == "__main__":
    main()
