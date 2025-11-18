"""
Convert YOLO normalized format to absolute coordinates for tracking.
YOLO format: class x_center y_center width height (normalized 0-1)
Tracker format: x1 y1 x2 y2 score class (absolute pixels)
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse


def convert_yolo_to_abs(yolo_label, img_width, img_height, confidence=0.9):
    """Convert YOLO normalized format to absolute bbox coordinates."""
    class_id, x_center, y_center, width, height = map(float, yolo_label.strip().split())
    
    # Convert to absolute coordinates
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    # Convert to x1, y1, x2, y2
    x1 = x_center_abs - width_abs / 2
    y1 = y_center_abs - height_abs / 2
    x2 = x_center_abs + width_abs / 2
    y2 = y_center_abs + height_abs / 2
    
    return f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {confidence:.3f} {int(class_id)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_dir', required=True, help='Directory with YOLO normalized labels')
    parser.add_argument('--img_dir', required=True, help='Directory with images to get dimensions')
    parser.add_argument('--out_dir', required=True, help='Output directory for converted labels')
    args = parser.parse_args()
    
    yolo_dir = Path(args.yolo_dir)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    label_files = sorted(yolo_dir.glob('*.txt'))
    
    print(f"Converting {len(label_files)} label files...")
    for label_file in tqdm(label_files):
        # Find corresponding image
        img_name = label_file.stem + '.jpg'
        img_path = img_dir / img_name
        
        if not img_path.exists():
            print(f"Warning: Image not found for {label_file.name}")
            continue
        
        # Get image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Convert labels
        with open(label_file, 'r') as f:
            yolo_labels = f.readlines()
        
        abs_labels = []
        for yolo_label in yolo_labels:
            if yolo_label.strip():
                abs_label = convert_yolo_to_abs(yolo_label, img_width, img_height)
                abs_labels.append(abs_label)
        
        # Write converted labels
        out_file = out_dir / label_file.name
        with open(out_file, 'w') as f:
            f.write('\n'.join(abs_labels))
            if abs_labels:
                f.write('\n')
    
    print(f"Conversion complete! Labels saved to {out_dir}")


if __name__ == '__main__':
    main()
