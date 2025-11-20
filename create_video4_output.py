import cv2
import json
import os
from pathlib import Path
import numpy as np

# Paths
video_path = "Wrongway_videos/video4.mp4"
config_path = "configs/video4_config.json"
tracks_dir = "results/video4/tracks"
alerts_path = "results/video4/alerts/wrong_way_alerts.json"
output_path = "results/video4/wrong_way_detection_output.mp4"

# Load config
with open(config_path) as f:
    config = json.load(f)

zones = config['zones']
flow_mappings = config['flow_mappings']

# Flatten flow mappings
flow_info = {}
for side in ['LEFT', 'RIGHT']:
    if side in flow_mappings:
        for pattern, flow_type in flow_mappings[side].items():
            if pattern != 'description':
                flow_info[pattern] = flow_type

# Load alerts
with open(alerts_path) as f:
    alerts_data = json.load(f)

alert_tracks = set()
for alert in alerts_data['alerts']:
    alert_tracks.add(alert['track_id'])

# Open video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Creating output video: {width}x{height} @ {fps} fps")
print(f"Total frames: {total_frames}")
print(f"Alert tracks: {alert_tracks}")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Load track data for this frame
    track_file = os.path.join(tracks_dir, f"frame_{frame_idx:06d}.json")
    
    current_alerts = 0
    
    if os.path.exists(track_file):
        with open(track_file) as f:
            tracks = json.load(f)
        
        # Draw each track
        for track in tracks:
            tid = track['track_id']
            bbox = track['bbox']
            cx, cy = track['centroid']
            side = track.get('side', 'UNKNOWN')
            
            # Check if this track is a wrong-way track
            is_alert = tid in alert_tracks
            
            if is_alert:
                current_alerts += 1
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 0, 255) if is_alert else (0, 255, 0)  # Red for alert, Green for normal
            thickness = 3 if is_alert else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw track ID
            label = f"ID {tid}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            
            # Draw centroid
            cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
    
    # Draw zones (lighter, semi-transparent)
    overlay = frame.copy()
    for zone_name, zone_data in zones.items():
        pts = np.array(zone_data['points'], np.int32)
        color = (0, 150, 0) if 'L' in zone_name else (150, 0, 0)  # Green for LEFT, Blue for RIGHT
        cv2.polylines(overlay, [pts], True, color, 2)
        
        # Zone label
        M = cv2.moments(pts)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.putText(overlay, zone_name, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, color, 2)
    
    # Blend overlay with frame
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw alert banner if there are wrong-way vehicles in frame
    if current_alerts > 0:
        cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 255), -1)
        cv2.putText(frame, "WRONG-WAY VEHICLE DETECTED!", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Draw info panel
    panel_height = 200
    cv2.rectangle(frame, (0, height - panel_height), (500, height), (0, 0, 0), -1)
    
    # Frame info
    y_offset = height - panel_height + 30
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames-1}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Flow directions
    y_offset += 30
    cv2.putText(frame, "LEFT Flow:", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    y_offset += 25
    cv2.putText(frame, "  Normal: A_L -> B_L -> C_L", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y_offset += 20
    cv2.putText(frame, "  Wrong: C_L -> B_L -> A_L", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    y_offset += 30
    cv2.putText(frame, "RIGHT Flow:", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    y_offset += 25
    cv2.putText(frame, "  Normal: A_R -> B_R -> C_R", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y_offset += 20
    cv2.putText(frame, "  Wrong: C_R -> B_R -> A_R", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Alert count
    cv2.rectangle(frame, (width - 250, height - 80), (width, height), (50, 50, 50), -1)
    alert_color = (0, 0, 255) if current_alerts > 0 else (0, 255, 0)
    cv2.putText(frame, f"Alerts: {current_alerts}", (width - 230, height - 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, alert_color, 2)
    
    # Write frame
    out.write(frame)
    
    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"Processed {frame_idx}/{total_frames} frames...")

cap.release()
out.release()

print(f"\n✅ Output video saved to: {output_path}")
print(f"Total frames processed: {frame_idx}")
print(f"Wrong-way tracks: {alert_tracks}")
