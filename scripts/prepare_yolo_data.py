"""
Prepare YOLO training data structure by copying/symlinking files.

This script reads the train.txt and val.txt files and populates
the yolov3_data directory structure with images and labels.

Usage:
    python scripts/prepare_yolo_data.py
"""

import os
import shutil
from pathlib import Path


def prepare_yolo_data():
    """Prepare YOLO data structure from train/val split files."""
    
    # Define paths
    project_root = Path(__file__).parent.parent
    subset_dir = project_root / "data" / "subset"
    labels_dir = project_root / "data" / "annotations_yolo" / "labels"
    
    train_file = project_root / "configs" / "train.txt"
    val_file = project_root / "configs" / "val.txt"
    
    yolo_data_dir = project_root / "yolov3_data"
    
    # Create target directories (should already exist)
    (yolo_data_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_data_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_data_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_data_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    def copy_files(split_file, split_name):
        """Copy images and labels for a given split."""
        print(f"\nProcessing {split_name} split...")
        
        if not split_file.exists():
            print(f"Error: {split_file} not found!")
            return 0
        
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        copied_images = 0
        copied_labels = 0
        missing_images = 0
        missing_labels = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse line: data/subset/sequence_name/img_name.jpg
            parts = line.split('/')
            if len(parts) != 4 or parts[0] != 'data' or parts[1] != 'subset':
                print(f"Warning: Invalid line format: {line}")
                continue
            
            sequence_name, img_name = parts[2], parts[3]
            
            # Source paths
            src_image = subset_dir / sequence_name / img_name
            src_label = labels_dir / img_name.replace('.jpg', '.txt')
            
            # Destination paths
            dst_image = yolo_data_dir / "images" / split_name / img_name
            dst_label = yolo_data_dir / "labels" / split_name / img_name.replace('.jpg', '.txt')
            
            # Copy image
            if src_image.exists():
                shutil.copy2(src_image, dst_image)
                copied_images += 1
            else:
                missing_images += 1
                print(f"Warning: Image not found: {src_image}")
            
            # Copy label
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
                copied_labels += 1
            else:
                # Create empty label file if it doesn't exist
                dst_label.touch()
                copied_labels += 1
                missing_labels += 1
        
        print(f"  Images copied: {copied_images}")
        print(f"  Labels copied: {copied_labels}")
        if missing_images > 0:
            print(f"  Missing images: {missing_images}")
        if missing_labels > 0:
            print(f"  Empty labels created: {missing_labels}")
        
        return copied_images
    
    # Process train and val splits
    print("="*60)
    print("Preparing YOLO training data structure")
    print("="*60)
    
    train_count = copy_files(train_file, "train")
    val_count = copy_files(val_file, "val")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Train images: {train_count}")
    print(f"  Val images: {val_count}")
    print(f"  Total images: {train_count + val_count}")
    print(f"\nData structure ready at: {yolo_data_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    prepare_yolo_data()
