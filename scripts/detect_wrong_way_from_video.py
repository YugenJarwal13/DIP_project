#!/usr/bin/env python3
"""
End-to-End Wrong-Way Detection Pipeline for Raw Videos
======================================================

This script processes a raw video file and detects wrong-way violations:
1. Extracts frames from video
2. Runs YOLO object detection
3. Applies DeepSORT tracking
4. Validates wrong-way behavior using zone-based reasoning
5. Generates annotated output video with alerts
6. Produces summary reports (JSON, CSV)

Usage:
    python scripts/detect_wrong_way_from_video.py \
        --video path/to/video.mp4 \
        --config configs/highway_config.json \
        --output outputs/results/video_name \
        --visualize

Requirements:
    - Zone configuration file (JSON) with zones A, B, C defined
    - Trained YOLO model (best.pt - YOLOv10m)
    - ffmpeg (optional, for better frame extraction)
"""

import argparse
import sys
import os
import json
import csv
import shutil
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.tracking.kalman_tracker import (
    KalmanTrack,
    associate_detections_to_trackers,
    IOU_THRESH as DEFAULT_IOU_THRESHOLD_IMPORT
)
from src.utils.side_assignment import (
    assign_side_by_threshold,
    assign_sides_by_kmeans,
    update_track_sides,
    get_side_statistics
)
from src.utils.direction_vector import (
    compute_bidirectional_direction_vectors,
    compute_unidirectional_direction_vector,
    identify_normal_flow_tracks
)
from shapely.geometry import Point, Polygon


# ==================== Configuration ====================
DEFAULT_MODEL = "best.pt"
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.45
DEFAULT_MAX_AGE = 30
DEFAULT_MIN_HITS = 3
DEFAULT_IOU_THRESHOLD = 0.3
MIN_ZONE_CONFIRM = 3
DOT_THRESH = 0.3
DISP_WINDOW = 6
MIN_DISPLACEMENT = 1.0
# =======================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='End-to-end wrong-way detection from raw video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/detect_wrong_way_from_video.py --video highway.mp4 --config configs/highway.json --output results/highway
  
  # With visualization and frame preservation
  python scripts/detect_wrong_way_from_video.py --video highway.mp4 --config configs/highway.json --output results/highway --visualize --keep-frames
  
  # Custom detection parameters
  python scripts/detect_wrong_way_from_video.py --video highway.mp4 --config configs/highway.json --output results/highway --conf 0.5 --fps 10
        """
    )
    
    # Required arguments
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file (.mp4, .avi, etc.)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to zone configuration JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for results')
    
    # Model arguments
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'Path to YOLO model (default: {DEFAULT_MODEL})')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_IMGSZ,
                        help=f'Inference image size (default: {DEFAULT_IMGSZ})')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF,
                        help=f'Confidence threshold (default: {DEFAULT_CONF})')
    
    # Tracking arguments
    parser.add_argument('--max-age', type=int, default=DEFAULT_MAX_AGE,
                        help=f'Max frames to keep track without detection (default: {DEFAULT_MAX_AGE})')
    parser.add_argument('--min-hits', type=int, default=DEFAULT_MIN_HITS,
                        help=f'Min hits before track is confirmed (default: {DEFAULT_MIN_HITS})')
    parser.add_argument('--iou-threshold', type=float, default=DEFAULT_IOU_THRESHOLD,
                        help=f'IoU threshold for matching (default: {DEFAULT_IOU_THRESHOLD})')
    
    # Validation arguments
    parser.add_argument('--min-zone-confirm', type=int, default=MIN_ZONE_CONFIRM,
                        help=f'Frames to confirm zone visit (default: {MIN_ZONE_CONFIRM})')
    parser.add_argument('--dot-thresh', type=float, default=DOT_THRESH,
                        help=f'Displacement alignment threshold (default: {DOT_THRESH})')
    
    # Output options
    parser.add_argument('--visualize', action='store_true',
                        help='Generate annotated output video with alerts')
    parser.add_argument('--keep-frames', action='store_true',
                        help='Keep extracted frames (default: delete after processing)')
    parser.add_argument('--fps', type=float, default=None,
                        help='Output video FPS (default: same as input)')
    parser.add_argument('--skip-frames', type=int, default=1,
                        help='Process every Nth frame (default: 1 = all frames)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load zone configuration from JSON file (supports both unidirectional and bidirectional)."""
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    # Check if bidirectional or unidirectional
    carriageway_mode = cfg.get('carriageway_mode', 'single')
    
    # Parse zones
    zones = {}
    if carriageway_mode == 'bidirectional':
        # Parse bidirectional zones: A_L, B_L, C_L, A_R, B_R, C_R
        for zone_name in ['A_L', 'B_L', 'C_L', 'A_R', 'B_R', 'C_R']:
            pts = cfg.get('zones', {}).get(zone_name, {}).get('points', None)
            if pts:
                zones[zone_name] = Polygon([(p[0], p[1]) for p in pts])
            else:
                zones[zone_name] = None
    else:
        # Parse unidirectional zones: A, B, C
        for zone_name in ['A', 'B', 'C']:
            pts = cfg.get('zones', {}).get(zone_name, {}).get('points', None)
            if pts:
                zones[zone_name] = Polygon([(p[0], p[1]) for p in pts])
            else:
                zones[zone_name] = None
    
    # Parse expected mapping (legacy for unidirectional)
    expected_mapping = cfg.get('expected_mapping', {"A->C": "normal", "C->A": "wrong_way"})
    
    # Get FPS from video_resolution or top-level
    fps = cfg.get('fps', None)
    if fps is None:
        video_res = cfg.get('video_resolution', {})
        fps = video_res.get('fps', 30)  # Default to 30 if not found
    
    return {
        'zones': zones,
        'expected_mapping': expected_mapping,
        'carriageway_mode': carriageway_mode,
        'side_assignment': cfg.get('side_assignment', {}),
        'flow_mappings': cfg.get('flow_mappings', {}),
        'image_width': cfg.get('image_width', cfg.get('video_resolution', {}).get('width', None)),
        'image_height': cfg.get('image_height', cfg.get('video_resolution', {}).get('height', None)),
        'fps': fps,
        'sequence_id': cfg.get('sequence_id', cfg.get('camera_info', {}).get('name', 'unknown'))
    }


def extract_frames_from_video(video_path, output_dir, skip_frames=1):
    """Extract frames from video file."""
    print(f"\n{'='*60}")
    print("STEP 1: EXTRACTING FRAMES FROM VIDEO")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    print(f"  Processing every {skip_frames} frame(s)")
    
    # Create output directory
    frames_dir = Path(output_dir) / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    frame_idx = 0
    saved_count = 0
    
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip frames if specified
            if frame_idx % skip_frames == 0:
                frame_path = frames_dir / f"img{saved_count:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_count += 1
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    
    print(f"✅ Extracted {saved_count} frames to {frames_dir}")
    
    return {
        'frames_dir': frames_dir,
        'total_frames': saved_count,
        'original_fps': fps,
        'width': width,
        'height': height
    }


def run_detection_and_tracking(frames_dir, model_path, output_dir, config, args):
    """Run YOLO detection and Kalman tracking."""
    print(f"\n{'='*60}")
    print("STEP 2: RUNNING DETECTION AND TRACKING")
    print(f"{'='*60}")
    
    # Load YOLO model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Initialize trackers list (following kalman_tracker.py pattern)
    trackers = []
    
    # Get image files
    image_files = sorted(Path(frames_dir).glob('*.jpg'))
    print(f"Processing {len(image_files)} frames...")
    
    # Create output directory for tracks
    tracks_dir = Path(output_dir) / 'tracks'
    tracks_dir.mkdir(parents=True, exist_ok=True)
    
    for frame_idx, img_path in enumerate(tqdm(image_files, desc="Detecting & tracking")):
        # Run YOLO detection
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False
        )
        
        # Extract detections in format [x1, y1, x2, y2, score]
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(boxes.conf[i].cpu().numpy())
                
                # Format: [x1, y1, x2, y2, score]
                detections.append([box[0], box[1], box[2], box[3], conf])
        
        # Predict step for all trackers
        for trk in trackers:
            trk.predict()
        
        # Associate detections to trackers
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, trackers, args.iou_threshold
        )
        
        # Update matched trackers
        for (d_idx, t_idx) in matches:
            det = detections[d_idx]
            trackers[t_idx].update(det[:4], det[4])
        
        # Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_trk = KalmanTrack(det[:4], det[4])
            trackers.append(new_trk)
        
        # Remove dead trackers
        to_del = []
        for trk in trackers:
            if trk.time_since_update > args.max_age:
                to_del.append(trk)
        for trk in to_del:
            try:
                trackers.remove(trk)
            except ValueError:
                pass
        
        # Get confirmed tracks
        confirmed = [t for t in trackers if t.hits >= args.min_hits or t.time_since_update == 0]
        
        # Save frame tracks
        frame_tracks = []
        for trk in confirmed:
            bbox = trk.to_bbox()
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            
            track_data = {
                'track_id': int(trk.id),
                'bbox': [round(float(b), 3) for b in bbox],
                'centroid': [round(float(cx), 3), round(float(cy), 3)],
                'score': float(trk.score),
                'age': int(trk.age),
                'hits': int(trk.hits),
                'time_since_update': int(trk.time_since_update)
            }
            frame_tracks.append(track_data)
        
        # Assign sides to tracks if config has bidirectional mode
        if config.get('carriageway_mode') == 'bidirectional' and 'side_assignment' in config:
            frame_tracks = update_track_sides(frame_tracks, config['side_assignment'])
        
        # Save per-frame JSON
        track_file = tracks_dir / f"frame_{frame_idx:06d}.json"
        with open(track_file, 'w') as f:
            json.dump(frame_tracks, f, indent=2)
    
    print(f"✅ Generated {len(image_files)} track files in {tracks_dir}")
    max_track_id = max([t.id for t in trackers]) if trackers else 0
    print(f"   Total unique tracks: {max_track_id}")
    
    return tracks_dir


def update_track_sides(frame_tracks, side_assignment):
    """Assign LEFT or RIGHT side to each track based on centroid position."""
    method = side_assignment.get('method', 'centroid_x_threshold')
    threshold = side_assignment.get('threshold', 640)
    
    for track in frame_tracks:
        cx, cy = track['centroid']
        
        if method == 'centroid_x_threshold':
            # If centroid_x < threshold: LEFT side, else RIGHT side
            if cx < threshold:
                track['side'] = 'LEFT'
            else:
                track['side'] = 'RIGHT'
        else:
            track['side'] = 'UNKNOWN'
    
    return frame_tracks


def centroid_from_bbox(bbox):
    """Calculate centroid from bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def condensed_zone_sequence(zone_hist, min_confirm):
    """Get condensed zone sequence from history."""
    if not zone_hist:
        return [], ""
    
    confirmed = []
    current_zone = None
    count = 0
    
    for _, zone in zone_hist:
        if zone == current_zone:
            count += 1
        else:
            if current_zone is not None and count >= min_confirm:
                confirmed.append(current_zone)
            current_zone = zone
            count = 1
    
    # Add final zone
    if current_zone is not None and count >= min_confirm:
        confirmed.append(current_zone)
    
    condensed = '->'.join(confirmed) if confirmed else ""
    return confirmed, condensed


def compute_displacement_vector(centroid_hist, window):
    """Compute displacement vector from recent centroids."""
    if len(centroid_hist) < window:
        return None
    
    recent = centroid_hist[-window:]
    start = np.array(recent[0])
    end = np.array(recent[-1])
    
    disp = end - start
    return disp


def validate_wrong_way(tracks_dir, config, output_dir, args):
    """Validate tracks for wrong-way behavior with bi-directional support."""
    print(f"\n{'='*60}")
    print("STEP 3: VALIDATING WRONG-WAY BEHAVIOR")
    print(f"{'='*60}")
    
    zones = config['zones']
    carriageway_mode = config.get('carriageway_mode', 'single')
    flow_mappings = config.get('flow_mappings', {})
    
    # Load all track frames
    track_files = sorted(Path(tracks_dir).glob("frame_*.json"))
    
    # Track history
    tracks_centers = {}
    tracks_zone_hist = {}
    tracks_side = {}
    alerts = []
    
    print(f"Processing {len(track_files)} frames for validation...")
    if carriageway_mode == 'bidirectional':
        print("✅ Bi-directional mode: Checking both LEFT and RIGHT carriageways")
    
    for track_file in tqdm(track_files, desc="Validating"):
        frame_num = int(track_file.stem.split('_')[1])
        
        with open(track_file, 'r') as f:
            frame_tracks = json.load(f)
        
        for track in frame_tracks:
            tid = track['track_id']
            bbox = track['bbox']
            cx, cy = centroid_from_bbox(bbox)
            time_since_update = track.get('time_since_update', 0)
            hits = track.get('hits', 0)
            track_side = track.get('side', 'UNKNOWN')
            
            # Skip predicted-only tracks
            if time_since_update > 0:
                continue
            
            # Skip low-quality tracks
            if hits < args.min_hits:
                continue
            
            # Update centroid history and side
            tracks_centers.setdefault(tid, []).append((cx, cy))
            tracks_side[tid] = track_side
            
            # Determine current zone
            zone_label = None
            for zone_name, zone_poly in zones.items():
                if zone_poly is None:
                    continue
                try:
                    if zone_poly.contains(Point(cx, cy)):
                        zone_label = zone_name
                        break
                except:
                    pass
            
            tracks_zone_hist.setdefault(tid, []).append((frame_num, zone_label))
            
            # Check for wrong-way pattern
            confirmed_seq, _ = condensed_zone_sequence(
                tracks_zone_hist[tid], 
                args.min_zone_confirm
            )
            
            # Need at least 3 zones
            if len(confirmed_seq) < 3:
                continue
            
            # For bi-directional, check side-specific patterns
            if carriageway_mode == 'bidirectional' and track_side in flow_mappings:
                side_flows = flow_mappings[track_side]
                is_wrong_way = False
                matched_pattern = None
                
                # Check all defined patterns for this side
                for pattern, flow_type in side_flows.items():
                    if flow_type == "wrong_way" and '-' in pattern:
                        expected_zones = pattern.split('-')
                        if len(expected_zones) == 3:
                            # Check if last 3 confirmed zones match wrong-way pattern
                            last3 = confirmed_seq[-3:]
                            if last3 == expected_zones:
                                is_wrong_way = True
                                matched_pattern = pattern
                                break
                
                if not is_wrong_way:
                    continue
            else:
                # Legacy single-direction logic
                last3 = confirmed_seq[-3:]
                if last3 != ['C', 'B', 'A']:
                    continue
                matched_pattern = 'C->B->A'
            
            # Check displacement alignment (optional for better accuracy)
            cent_hist = tracks_centers[tid]
            disp = compute_displacement_vector(cent_hist, DISP_WINDOW)
            
            alignment_score = 1.0  # Default if no displacement check
            if disp is not None:
                disp_norm = np.linalg.norm(disp)
                if disp_norm >= MIN_DISPLACEMENT:
                    disp_unit = disp / disp_norm
                    # For bi-directional, we skip strict direction check
                    # Just use displacement magnitude as confidence
                    alignment_score = min(disp_norm / 100.0, 1.0)
            
            # ALERT: Wrong-way detected!
            alert = {
                'frame': frame_num,
                'track_id': tid,
                'alert_type': 'WRONG_WAY',
                'side': track_side,
                'zone_sequence': '-'.join(confirmed_seq[-3:]) if matched_pattern else '->'.join(confirmed_seq[-3:]),
                'matched_pattern': matched_pattern,
                'centroid': [cx, cy],
                'bbox': bbox,
                'alignment_score': float(alignment_score),
                'hits': hits,
                'timestamp': frame_num / config.get('fps', 30) if config.get('fps') else None
            }
            
            # Check if already alerted for this track
            if not any(a['track_id'] == tid for a in alerts):
                alerts.append(alert)
                print(f"\n⚠️  WRONG-WAY ALERT: Track {tid} ({track_side}) at frame {frame_num}")
                print(f"    Zone sequence: {alert['zone_sequence']}")
                print(f"    Pattern matched: {matched_pattern}")
    
    # Save alerts
    alerts_dir = Path(output_dir) / 'alerts'
    alerts_dir.mkdir(parents=True, exist_ok=True)
    
    alerts_json = alerts_dir / 'wrong_way_alerts.json'
    with open(alerts_json, 'w') as f:
        json.dump({
            'total_alerts': len(alerts),
            'alerts': alerts,
            'detection_params': {
                'min_zone_confirm': args.min_zone_confirm,
                'dot_threshold': args.dot_thresh,
                'min_hits': args.min_hits
            }
        }, f, indent=2)
    
    # Save CSV summary
    alerts_csv = alerts_dir / 'wrong_way_alerts.csv'
    with open(alerts_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Track ID', 'Zone Sequence', 'Centroid X', 'Centroid Y', 'Alignment Score', 'Timestamp (s)'])
        for alert in alerts:
            writer.writerow([
                alert['frame'],
                alert['track_id'],
                alert['zone_sequence'],
                f"{alert['centroid'][0]:.1f}",
                f"{alert['centroid'][1]:.1f}",
                f"{alert['alignment_score']:.3f}",
                f"{alert['timestamp']:.2f}" if alert['timestamp'] else 'N/A'
            ])
    
    print(f"\n✅ Detected {len(alerts)} wrong-way violations")
    print(f"   Alerts saved to: {alerts_json}")
    print(f"   CSV saved to: {alerts_csv}")
    
    return alerts


def create_visualization_video(frames_dir, tracks_dir, alerts, config, output_dir, video_info, args):
    """Create annotated output video with alerts."""
    print(f"\n{'='*60}")
    print("STEP 4: CREATING VISUALIZATION VIDEO")
    print(f"{'='*60}")
    
    image_files = sorted(Path(frames_dir).glob('*.jpg'))
    
    if not image_files:
        print("⚠️  No frames found for visualization")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(image_files[0]))
    h, w = first_frame.shape[:2]
    
    # Setup video writer
    output_fps = args.fps if args.fps else video_info['original_fps']
    output_video = Path(output_dir) / 'wrong_way_detection_output.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video), fourcc, output_fps, (w, h))
    
    # Create alert lookup
    alert_frames = {alert['frame']: alert for alert in alerts}
    alert_tracks = {alert['track_id'] for alert in alerts}
    
    zones = config['zones']
    is_bidirectional = config.get('carriageway_mode') == 'bidirectional'
    flow_mappings = config.get('flow_mappings', {})
    
    print(f"Generating video at {output_fps:.1f} FPS...")
    
    for frame_idx, img_path in enumerate(tqdm(image_files, desc="Rendering frames")):
        frame = cv2.imread(str(img_path))
        
        # Draw zones (semi-transparent)
        overlay = frame.copy()
        
        if is_bidirectional:
            # Bidirectional zone colors
            zone_colors = {
                'A_L': (50, 200, 50), 'B_L': (50, 200, 200), 'C_L': (50, 50, 200),
                'A_R': (100, 255, 100), 'B_R': (100, 255, 255), 'C_R': (100, 100, 255)
            }
        else:
            zone_colors = {'A': (0, 255, 0), 'B': (0, 255, 255), 'C': (0, 0, 255)}
        
        for zone_name, zone_data in zones.items():
            if zone_data is None:
                continue
            
            if isinstance(zone_data, dict):
                pts = np.array(zone_data['points'], dtype=np.int32)
            else:
                pts = np.array(zone_data.exterior.coords, dtype=np.int32)
            
            color = zone_colors.get(zone_name, (100, 100, 100))
            cv2.fillPoly(overlay, [pts], color)
        
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Draw zone labels on ROI regions
        for zone_name, zone_data in zones.items():
            if zone_data is None:
                continue
            
            if isinstance(zone_data, dict):
                pts = zone_data['points']
            else:
                pts = list(zone_data.exterior.coords)
            
            # Calculate centroid
            center_x = int(sum([p[0] for p in pts]) / len(pts))
            center_y = int(sum([p[1] for p in pts]) / len(pts))
            
            # Zone label with background
            label = f"{zone_name}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, 
                         (center_x - text_size[0]//2 - 5, center_y - text_size[1]//2 - 5),
                         (center_x + text_size[0]//2 + 5, center_y + text_size[1]//2 + 5),
                         (0, 0, 0), -1)
            cv2.putText(frame, label, (center_x - text_size[0]//2, center_y + text_size[1]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Load tracks for this frame
        track_file = Path(tracks_dir) / f"frame_{frame_idx:06d}.json"
        frame_tracks = []
        current_frame_alert_tracks = []
        
        if track_file.exists():
            with open(track_file, 'r') as f:
                frame_tracks = json.load(f)
            
            # Identify which tracks in this frame are alerts
            for track in frame_tracks:
                if track['track_id'] in alert_tracks:
                    current_frame_alert_tracks.append(track['track_id'])
            
            # Draw tracks
            for track in frame_tracks:
                tid = track['track_id']
                bbox = track['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Different color for wrong-way tracks
                if tid in alert_tracks:
                    color = (0, 0, 255)  # Red
                    thickness = 3
                    label = f"WRONG WAY! ID:{tid}"
                else:
                    color = (0, 255, 0)  # Green
                    thickness = 2
                    label = f"ID:{tid}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw alert banner if any wrong-way tracks are visible in this frame
        if current_frame_alert_tracks:
            # Show banner for each wrong-way vehicle in frame
            for idx, tid in enumerate(current_frame_alert_tracks):
                banner_y = idx * 55
                banner_text = f"WRONG-WAY ALERT! Track ID {tid}"
                
                # Red banner at top
                cv2.rectangle(frame, (0, banner_y), (w, banner_y + 50), (0, 0, 200), -1)
                cv2.putText(frame, banner_text, (10, banner_y + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # ========== LEFT SIDE PANEL: Flow Information ==========
        panel_width = 280
        panel_height = 150
        panel_x = 10
        panel_y = 60
        
        # Semi-transparent background
        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(panel_overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "CORRECT FLOW", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if is_bidirectional:
            # Get normal flow patterns from config
            left_normal = [k for k, v in flow_mappings.get('LEFT', {}).items() if v == 'normal']
            right_normal = [k for k, v in flow_mappings.get('RIGHT', {}).items() if v == 'normal']
            
            # LEFT side flow
            cv2.putText(frame, "LEFT SIDE:", (panel_x + 10, panel_y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 2)
            left_text = left_normal[0].replace('-', ' -> ') if left_normal else "A -> B -> C"
            cv2.putText(frame, left_text, (panel_x + 20, panel_y + 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50), 2)
            
            # RIGHT side flow
            cv2.putText(frame, "RIGHT SIDE:", (panel_x + 10, panel_y + 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 200, 255), 2)
            right_text = right_normal[0].replace('-', ' -> ') if right_normal else "C -> B -> A"
            cv2.putText(frame, right_text, (panel_x + 20, panel_y + 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 150, 255), 2)
        else:
            # Unidirectional flow
            expected = config.get('expected_mapping', {})
            normal_flow = [k for k, v in expected.items() if v == 'normal']
            if normal_flow:
                flow_text = f"{normal_flow[0].replace('->', ' -> ')}"
                cv2.putText(frame, flow_text, (panel_x + 10, panel_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # ========== BOTTOM PANEL: Frame Info & FPS ==========
        bottom_panel_height = 60
        bottom_y = h - bottom_panel_height
        
        # Semi-transparent bottom bar
        bottom_overlay = frame.copy()
        cv2.rectangle(bottom_overlay, (0, bottom_y), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(bottom_overlay, 0.7, frame, 0.3, 0, frame)
        
        # Frame counter and FPS
        info_text = f"Frame: {frame_idx + 1}/{len(image_files)}"
        cv2.putText(frame, info_text, (15, bottom_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        fps_text = f"FPS: {output_fps:.1f}"
        cv2.putText(frame, fps_text, (15, bottom_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
        
        # Alert count for current frame
        current_alert_count = len(current_frame_alert_tracks)
        alert_text = f"Alerts: {current_alert_count}"
        alert_color = (0, 0, 255) if current_alert_count > 0 else (0, 255, 0)
        cv2.putText(frame, alert_text, (w - 150, bottom_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
        
        # ROI status
        roi_text = f"ROI Zones: {len(zones)}"
        cv2.putText(frame, roi_text, (w - 180, bottom_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        
        video_writer.write(frame)
    
    video_writer.release()
    
    print(f"✅ Visualization video saved: {output_video}")
    print(f"   Resolution: {w}x{h}")
    print(f"   FPS: {output_fps:.1f}")
    print(f"   Duration: {len(image_files)/output_fps:.1f} seconds")


def main():
    """Main pipeline execution."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("WRONG-WAY DETECTION PIPELINE")
    print("="*60)
    print(f"Video: {args.video}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print("="*60)
    
    # Validate inputs
    if not Path(args.video).exists():
        print(f"❌ Error: Video file not found: {args.video}")
        return
    
    if not Path(args.config).exists():
        print(f"❌ Error: Config file not found: {args.config}")
        return
    
    if not Path(args.model).exists():
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Step 1: Extract frames
    video_info = extract_frames_from_video(args.video, output_dir, args.skip_frames)
    
    # Step 2: Run detection and tracking
    tracks_dir = run_detection_and_tracking(
        video_info['frames_dir'],
        args.model,
        output_dir,
        config,
        args
    )
    
    # Step 3: Validate wrong-way behavior
    alerts = validate_wrong_way(tracks_dir, config, output_dir, args)
    
    # Step 4: Create visualization video (if requested)
    if args.visualize:
        create_visualization_video(
            video_info['frames_dir'],
            tracks_dir,
            alerts,
            config,
            output_dir,
            video_info,
            args
        )
    
    # Cleanup frames (if not keeping)
    if not args.keep_frames:
        print(f"\nCleaning up extracted frames...")
        shutil.rmtree(video_info['frames_dir'])
        print(f"✅ Temporary frames deleted")
    
    # Final summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"✅ Processed: {video_info['total_frames']} frames")
    print(f"✅ Detected: {len(alerts)} wrong-way violations")
    print(f"✅ Results saved to: {output_dir}")
    print(f"\nOutput files:")
    print(f"  - Tracks: {output_dir}/tracks/")
    print(f"  - Alerts (JSON): {output_dir}/alerts/wrong_way_alerts.json")
    print(f"  - Alerts (CSV): {output_dir}/alerts/wrong_way_alerts.csv")
    if args.visualize:
        print(f"  - Video: {output_dir}/wrong_way_detection_output.mp4")
    if args.keep_frames:
        print(f"  - Frames: {output_dir}/frames/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
