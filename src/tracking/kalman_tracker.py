#!/usr/bin/env python3
"""
kalman_tracker.py

Multi-object tracker using a simple Kalman filter (constant velocity) + IOU matching + Hungarian.

Usage:
    python src/tracking/kalman_tracker.py --detections_dir outputs/detections/MVI_20011 --img_dir data/subset/MVI_20011 --out_dir outputs/tracks/MVI_20011 --visualize False

Outputs:
    out_dir/frame_000001.json  (list of tracks for each frame)
    If visualize=True, saves overlay images to out_dir/visual/

Notes:
- Expects detection files named like img00001.txt with lines: x1 y1 x2 y2 score class
- Adjust IOU_THRESH, MAX_AGE, MIN_HITS at top to tune behavior.
"""

import os
import json
import argparse
from glob import glob
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
from tqdm import tqdm

# --- Parameters (tweak if needed) ---
IOU_THRESH = 0.3
MAX_AGE = 30       # frames to keep alive without updates
MIN_HITS = 3       # minimum hits to consider a track confirmed
DISP_HISTORY = 6   # used later by validator (not used here)
# ------------------------------------

def iou(bb_test, bb_gt):
    # bb = [x1,y1,x2,y2]
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

class KalmanTrack:
    """
    Simple Kalman filter for bbox centroid + velocity + size.
    State: [cx, cy, vx, vy, w, h]^T
    Measurement: [cx, cy, w, h]
    """
    count = 0
    def __init__(self, bbox, score):
        # bbox: [x1,y1,x2,y2]
        x1,y1,x2,y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        # Initialize state vector
        self.x = np.array([cx, cy, 0., 0., w, h], dtype=np.float32)  # (6,)

        # State covariance
        self.P = np.diag([50.,50.,50.,50.,50.,50.]).astype(np.float32)

        # Motion matrix (constant velocity)
        dt = 1.0
        self.F = np.eye(6, dtype=np.float32)
        self.F[0,2] = dt
        self.F[1,3] = dt

        # Measurement matrix
        self.H = np.zeros((4,6), dtype=np.float32)
        self.H[0,0] = 1.0  # cx
        self.H[1,1] = 1.0  # cy
        self.H[2,4] = 1.0  # w
        self.H[3,5] = 1.0  # h

        # Process & measurement noise
        self.Q = np.eye(6, dtype=np.float32) * 1.0
        self.R = np.eye(4, dtype=np.float32) * 10.0

        # track bookkeeping
        KalmanTrack.count += 1
        self.id = KalmanTrack.count
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.score = score
        self.history = []  # store raw bbox history if needed

    def predict(self):
        # x = F x
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        self.age += 1
        self.time_since_update += 1
        # return predicted bbox
        return self.to_bbox()

    def update(self, bbox, score):
        # measurement z = [cx,cy,w,h]
        x1,y1,x2,y2 = bbox
        z = np.array([(x1+x2)/2.0, (y1+y2)/2.0, x2-x1, y2-y1], dtype=np.float32)

        # Kalman gain
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        y = z - self.H.dot(self.x)
        self.x = self.x + K.dot(y)
        I = np.eye(self.F.shape[0])
        self.P = (I - K.dot(self.H)).dot(self.P)

        self.time_since_update = 0
        self.hits += 1
        self.score = score
        self.history.append(self.to_bbox())

    def to_bbox(self):
        cx,cy,_,_,w,h = self.x
        x1 = cx - w/2.0
        y1 = cy - h/2.0
        x2 = cx + w/2.0
        y2 = cy + h/2.0
        return [float(x1), float(y1), float(x2), float(y2)]

def read_detections_for_frame(det_dir, filename):
    path = os.path.join(det_dir, filename)
    dets = []
    if not os.path.exists(path):
        return dets
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            x1,y1,x2,y2,score,cls = parts
            try:
                dets.append([float(x1), float(y1), float(x2), float(y2), float(score)])
            except:
                continue
    return dets

def associate_detections_to_trackers(detections, trackers, iou_threshold=IOU_THRESH):
    """
    Returns:
        matches: list of (det_idx, track_idx)
        unmatched_detections: list of det_idx
        unmatched_trackers: list of track_idx
    """
    if len(trackers) == 0:
        return [], list(range(len(detections))), []

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det[:4], trk.to_bbox())

    # Hungarian on cost = 1 - iou
    cost_matrix = 1.0 - iou_matrix
    # set large cost for low iou so Hungarian avoids matching them
    cost_matrix[iou_matrix < iou_threshold] = 1.0 + (1.0 - iou_matrix[iou_matrix < iou_threshold])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_dets, unmatched_trks = [], [], []

    assigned_tracks = set()
    assigned_dets = set()
    for r,c in zip(row_ind, col_ind):
        if iou_matrix[r, c] < iou_threshold:
            unmatched_dets.append(r)
            unmatched_trks.append(c)
        else:
            matches.append((r, c))
            assigned_dets.add(r)
            assigned_tracks.add(c)

    for d in range(len(detections)):
        if d not in assigned_dets and d not in unmatched_dets:
            unmatched_dets.append(d)
    for t in range(len(trackers)):
        if t not in assigned_tracks and t not in unmatched_trks:
            unmatched_trks.append(t)

    return matches, unmatched_dets, unmatched_trks

def save_tracks_json(out_dir, frame_idx, tracks):
    out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.json")
    records = []
    for trk in tracks:
        bbox = trk.to_bbox()
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        rec = {
            "track_id": int(trk.id),
            "bbox": [round(float(b),3) for b in bbox],
            "centroid": [round(float(cx),3), round(float(cy),3)],
            "score": float(trk.score),
            "age": int(trk.age),
            "hits": int(trk.hits),
            "time_since_update": int(trk.time_since_update)
        }
        records.append(rec)
    with open(out_path, 'w') as f:
        json.dump(records, f, indent=2)

def overlay_and_save(img_path, detections, tracks, out_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    # draw zones? not here (validator will draw)
    # draw detections
    for d in detections:
        x1,y1,x2,y2,score = d
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 1)
        cv2.putText(img, f"{score:.2f}", (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
    # draw tracks
    for trk in tracks:
        bbox = trk.to_bbox()
        x1,y1,x2,y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(img, f"ID:{trk.id}", (x1, y2+12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0),2)
    cv2.imwrite(out_path, img)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detections_dir", required=True, help="Folder with per-frame detection .txt files")
    p.add_argument("--img_dir", required=False, help="Optional images folder for visualization")
    p.add_argument("--out_dir", required=True, help="Where to write per-frame track JSONs")
    p.add_argument("--visualize", action="store_true", help="Save visualization overlays")
    p.add_argument("--iou_thresh", type=float, default=IOU_THRESH)
    p.add_argument("--max_age", type=int, default=MAX_AGE)
    p.add_argument("--min_hits", type=int, default=MIN_HITS)
    args = p.parse_args()

    det_dir = args.detections_dir
    img_dir = args.img_dir
    out_dir = args.out_dir
    ensure_dirs = [out_dir, os.path.join(out_dir, "visual")]
    for d in ensure_dirs:
        os.makedirs(d, exist_ok=True)

    # collect sorted detection files (assume consistent naming)
    det_files = sorted(glob(os.path.join(det_dir, "*.txt")))
    if len(det_files) == 0:
        print("No detection files found in", det_dir)
        return

    # Initialize trackers list
    trackers = []

    frame_idx = 0
    for det_path in tqdm(det_files, desc="Tracking frames"):
        frame_idx += 1
        fname = os.path.basename(det_path)
        detections = read_detections_for_frame(det_dir, fname)  # list of [x1,y1,x2,y2,score]

        # Predict step for all trackers
        for trk in trackers:
            trk.predict()

        # Associate detections to trackers
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trackers, args.iou_thresh)

        # Update matched trackers with assigned detections
        for (d_idx, t_idx) in matches:
            det = detections[d_idx]
            trackers[t_idx].update(det[:4], det[4])

        # Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_trk = KalmanTrack(det[:4], det[4])
            trackers.append(new_trk)

        # Manage unmatched trackers (increase time_since_update already done in predict)
        to_del = []
        for t_idx, trk in enumerate(trackers):
            if trk.time_since_update > args.max_age:
                to_del.append(trk)
        # remove dead tracks
        for trk in to_del:
            try:
                trackers.remove(trk)
            except ValueError:
                pass

        # Save tracks for this frame (only include confirmed tracks or include all depending on min_hits)
        confirmed = [t for t in trackers if t.hits >= args.min_hits or t.time_since_update==0]
        save_tracks_json(out_dir, frame_idx, confirmed)

        # Visualization optional
        if args.visualize and img_dir:
            img_name = Path(fname).stem
            img_path = os.path.join(img_dir, img_name + ".jpg")
            vis_out = os.path.join(out_dir, "visual", f"{img_name}.jpg")
            overlay_and_save(img_path, detections, confirmed, vis_out)

    print("Tracking completed. Track JSONs saved to", out_dir)

if __name__ == "__main__":
    main()
