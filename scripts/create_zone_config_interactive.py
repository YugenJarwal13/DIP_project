"""
Interactive Zone Configuration Tool v2.0 - Bi-Directional Support
==================================================================

MANUAL ZONE DEFINITION - Click to create zones for LEFT and RIGHT carriageways!

Usage:
    # Unidirectional (3 zones)
    python scripts/create_zone_config_interactive.py --video wrongway.mp4 --output configs/wrongway_config.json --mode unidirectional
    
    # Bidirectional (6 zones)
    python scripts/create_zone_config_interactive.py --video wrongway.mp4 --output configs/wrongway_config.json --mode bidirectional

Instructions:
1. Video frame opens in window
2. For UNIDIRECTIONAL: Click 4+ points for zones A, B, C
3. For BIDIRECTIONAL: Click 4+ points for zones A_L, B_L, C_L, A_R, B_R, C_R
4. Press SPACE to move to next zone
5. Press 'q' when done to configure flow and save

Controls:
    - LEFT CLICK: Add point to current zone
    - 'r': Reset current zone
    - SPACE: Move to next zone
    - 'q': Configure flow and save
"""

import cv2
import json
import numpy as np
import sys

class ZoneCreator:
    """Interactive zone creation tool with bi-directional support"""
    
    def __init__(self, video_path, mode='unidirectional'):
        self.video_path = video_path
        self.mode = mode  # 'unidirectional' or 'bidirectional'
        self.cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read first frame
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError(f"Cannot read video: {video_path}")
        
        # Auto-scale to fit screen (max 1280x720)
        max_display_width = 1280
        max_display_height = 720
        
        if self.width > max_display_width or self.height > max_display_height:
            scale_w = max_display_width / self.width
            scale_h = max_display_height / self.height
            self.display_scale = min(scale_w, scale_h)
            
            self.display_width = int(self.width * self.display_scale)
            self.display_height = int(self.height * self.display_scale)
            
            print(f"Original resolution: {self.width}x{self.height}")
            print(f"Display resolution: {self.display_width}x{self.display_height}")
            print(f"Scale factor: {self.display_scale:.3f}")
        else:
            self.display_scale = 1.0
            self.display_width = self.width
            self.display_height = self.height
        
        self.original_frame = self.frame.copy()
        
        # Zone data based on mode
        if mode == 'bidirectional':
            self.zones = {
                'A_L': [], 'B_L': [], 'C_L': [],
                'A_R': [], 'B_R': [], 'C_R': []
            }
            self.zone_order = ['A_L', 'B_L', 'C_L', 'A_R', 'B_R', 'C_R']
            self.zone_colors = {
                'A_L': (0, 255, 0),    # Green (LEFT)
                'B_L': (0, 255, 255),  # Yellow (LEFT)
                'C_L': (0, 0, 255),    # Red (LEFT)
                'A_R': (100, 255, 100),    # Light Green (RIGHT)
                'B_R': (100, 255, 255),  # Light Yellow (RIGHT)
                'C_R': (100, 100, 255)     # Light Red (RIGHT)
            }
        else:
            self.zones = {'A': [], 'B': [], 'C': []}
            self.zone_order = ['A', 'B', 'C']
            self.zone_colors = {
                'A': (0, 255, 0),    # Green
                'B': (0, 255, 255),  # Yellow
                'C': (0, 0, 255)     # Red
            }
        
        self.current_zone_idx = 0
        self.current_zone = self.zone_order[0]
        
        self.cap.release()
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Scale coordinates back to original resolution
            actual_x = int(x / self.display_scale)
            actual_y = int(y / self.display_scale)
            
            # Add point to current zone
            self.zones[self.current_zone].append([actual_x, actual_y])
            print(f"Added point ({actual_x}, {actual_y}) to Zone {self.current_zone}")
            self.redraw()
    
    def redraw(self):
        """Redraw frame with current zones"""
        self.frame = self.original_frame.copy()
        
        # Draw completed zones
        for zone_name, points in self.zones.items():
            if len(points) > 0:
                color = self.zone_colors[zone_name]
                pts = np.array(points, dtype=np.int32)
                
                # Draw points
                for pt in points:
                    cv2.circle(self.frame, tuple(pt), 5, color, -1)
                
                # Draw lines between points
                if len(points) > 1:
                    cv2.polylines(self.frame, [pts], False, color, 2)
                
                # Draw filled polygon if zone is complete (4+ points)
                if len(points) >= 4:
                    overlay = self.frame.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.addWeighted(overlay, 0.3, self.frame, 0.7, 0, self.frame)
                
                # Label
                if len(points) > 0:
                    center_x = int(np.mean([p[0] for p in points]))
                    center_y = int(np.mean([p[1] for p in points]))
                    cv2.putText(self.frame, f"Zone {zone_name}", (center_x-40, center_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw minimal status overlay (top-right corner)
        status_text = f"Zone {self.current_zone} [{len(self.zones[self.current_zone])}/4+]"
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(self.frame, (self.width - text_size[0] - 20, 10), 
                     (self.width - 5, 40), (0, 0, 0), -1)
        cv2.putText(self.frame, status_text, 
                   (self.width - text_size[0] - 15, 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.zone_colors[self.current_zone], 2)
        
        # Scale frame for display if needed
        if self.display_scale != 1.0:
            display_frame = cv2.resize(self.frame, (self.display_width, self.display_height))
        else:
            display_frame = self.frame
        
        cv2.imshow("Zone Creator", display_frame)
    
    def run(self):
        """Run interactive zone creation"""
        print(f"\n{'='*80}")
        print(f"INTERACTIVE ZONE CREATION - Mode: {self.mode.upper()}")
        print(f"{'='*80}")
        print(f"Video: {self.video_path}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"\nInstructions:")
        
        if self.mode == 'bidirectional':
            print(f"  LEFT CARRIAGEWAY:")
            print(f"    1. Click 4+ points for Zone A_L (left entry)")
            print(f"    2. Press SPACE, click 4+ points for Zone B_L (left middle)")
            print(f"    3. Press SPACE, click 4+ points for Zone C_L (left exit)")
            print(f"  RIGHT CARRIAGEWAY:")
            print(f"    4. Press SPACE, click 4+ points for Zone A_R (right entry)")
            print(f"    5. Press SPACE, click 4+ points for Zone B_R (right middle)")
            print(f"    6. Press SPACE, click 4+ points for Zone C_R (right exit)")
        else:
            print(f"  1. Click 4+ points to define Zone A (entry - typically top)")
            print(f"  2. Press SPACE to move to Zone B (middle)")
            print(f"  3. Click 4+ points for Zone B")
            print(f"  4. Press SPACE to move to Zone C (exit - typically bottom)")
            print(f"  5. Click 4+ points for Zone C")
        
        print(f"\n  Press 'q' to configure flow and save")
        print(f"\nZones should cover the entire road width!")
        print(f"{'='*80}\n")
        
        cv2.namedWindow("Zone Creator")
        cv2.setMouseCallback("Zone Creator", self.mouse_callback)
        
        self.redraw()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit and save
                if all(len(self.zones[z]) >= 4 for z in self.zone_order):
                    print("\n✅ All zones defined!")
                    cv2.destroyAllWindows()
                    return self.zones
                else:
                    incomplete = [z for z in self.zone_order if len(self.zones[z]) < 4]
                    print(f"\n⚠️  Cannot save - incomplete zones: {', '.join(incomplete)}")
                    print(f"    Each zone needs at least 4 points.")
            
            elif key == ord(' '):
                # Next zone
                if len(self.zones[self.current_zone]) >= 4:
                    if self.current_zone_idx < len(self.zone_order) - 1:
                        self.current_zone_idx += 1
                        self.current_zone = self.zone_order[self.current_zone_idx]
                        print(f"\n✅ Zone {self.zone_order[self.current_zone_idx - 1]} complete! Now define Zone {self.current_zone}")
                    else:
                        print(f"\n✅ All zones complete! Press 'q' to configure flow and save.")
                    self.redraw()
                else:
                    print(f"\n⚠️  Zone {self.current_zone} needs at least 4 points (has {len(self.zones[self.current_zone])})")
            
            elif key == ord('r'):
                # Reset current zone
                self.zones[self.current_zone] = []
                print(f"\n🔄 Reset Zone {self.current_zone}")
                self.redraw()
            
            elif key == 27:  # ESC
                print("\n❌ Cancelled")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()

def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Create zone configuration interactively")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--output", default="configs/wrongway_config.json",
                       help="Output config file path")
    parser.add_argument("--mode", choices=['unidirectional', 'bidirectional'], 
                       default='unidirectional',
                       help="Zone mode: unidirectional (3 zones) or bidirectional (6 zones)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Create zones interactively
    creator = ZoneCreator(args.video, mode=args.mode)
    zones_dict = creator.run()
    
    if zones_dict is None:
        print("\n❌ Cancelled - no config saved")
        return
    
    # Configure flow mappings
    print(f"\n{'='*80}")
    print("FLOW CONFIGURATION")
    print(f"{'='*80}")
    
    config = {
        "video_resolution": {
            "width": creator.width,
            "height": creator.height
        },
        "camera_info": {
            "name": os.path.basename(args.video),
            "description": "Manually configured zones"
        }
    }
    
    if args.mode == 'bidirectional':
        # Bidirectional configuration
        print("\nConfiguring BIDIRECTIONAL flow...")
        
        # Side assignment threshold (use center by default)
        default_threshold = creator.width // 2
        threshold = default_threshold
        print(f"\nUsing side assignment threshold: {threshold} (frame center)")
        
        config["carriageway_mode"] = "bidirectional"
        config["side_assignment"] = {
            "method": "centroid_x_threshold",
            "threshold": threshold,
            "description": f"Vehicles with centroid_x < {threshold} assigned to LEFT, >= {threshold} to RIGHT"
        }
        
        # Zones
        config["zones"] = {}
        for zone_name in ['A_L', 'B_L', 'C_L', 'A_R', 'B_R', 'C_R']:
            side = "LEFT" if "_L" in zone_name else "RIGHT"
            config["zones"][zone_name] = {
                "type": "polygon",
                "side": side,
                "points": zones_dict[zone_name]
            }
        
        # LEFT flow (default: A_L → B_L → C_L)
        print("\nLEFT CARRIAGEWAY: Using default A_L → B_L → C_L (normal)")
        left_normal = "A_L-B_L-C_L"
        left_wrong = "C_L-B_L-A_L"
        
        # RIGHT flow (default: C_R → B_R → A_R for opposite direction)
        print("RIGHT CARRIAGEWAY: Using default C_R → B_R → A_R (normal, opposite to LEFT)")
        right_normal = "C_R-B_R-A_R"
        right_wrong = "A_R-B_R-C_R"
        
        config["flow_mappings"] = {
            "LEFT": {
                left_normal: "normal",
                left_wrong: "wrong_way",
                "description": f"Left carriageway: {left_normal} is correct flow"
            },
            "RIGHT": {
                right_normal: "normal",
                right_wrong: "wrong_way",
                "description": f"Right carriageway: {right_normal} is correct flow"
            }
        }
        
        # Dynamic direction vectors
        config["direction_vectors"] = {
            "LEFT": {
                "expected_normal": None,
                "expected_wrong": None,
                "computation": "dynamic",
                "description": "Computed from normal-flow tracks on left side at runtime"
            },
            "RIGHT": {
                "expected_normal": None,
                "expected_wrong": None,
                "computation": "dynamic",
                "description": "Computed from normal-flow tracks on right side at runtime"
            }
        }
        
        config["validation_params"] = {
            "MIN_ZONE_CONFIRM": 3,
            "DOT_THRESH": 0.3,
            "DISP_WINDOW": 6,
            "MIN_DISPLACEMENT": 1.0
        }
        
    else:
        # Unidirectional configuration (legacy format, default A→B→C)
        print("\nConfiguring UNIDIRECTIONAL flow (default A→B→C normal)...")
        
        config["zones"] = {
            "A": {
                "points": zones_dict['A'],
                "description": "Entry zone - Normal traffic enters here"
            },
            "B": {
                "points": zones_dict['B'],
                "description": "Middle zone - Transition area"
            },
            "C": {
                "points": zones_dict['C'],
                "description": "Exit zone - Normal traffic exits here"
            }
        }
        
        config["expected_mapping"] = {
            "A->C": "normal",
            "C->A": "wrong_way"
        }
    
    # Save config
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ CONFIGURATION SAVED")
    print(f"{'='*80}")
    print(f"Config file: {args.output}")
    print(f"Mode: {args.mode.upper()}")
    print(f"\nZone summary:")
    for zone_name, zone_data in zones_dict.items():
        print(f"  Zone {zone_name}: {len(zone_data)} points")
    
    print(f"\n📋 Next: Run detection with your config:")
    print(f"\n   python scripts/detect_wrong_way_from_video.py \\")
    print(f"       --video {args.video} \\")
    print(f"       --config {args.output} \\")
    print(f"       --output results/wrongway \\")
    print(f"       --visualize")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
