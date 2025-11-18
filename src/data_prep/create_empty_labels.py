#!/usr/bin/env python3
"""
create_empty_labels.py

Create empty YOLO .txt label files for images in data/subset that don't already
have a corresponding label in data/annotations_yolo/labels.

Usage (from project root):
    python src/data_prep/create_empty_labels.py --subset_dir data/subset --labels_dir data/annotations_yolo/labels

This script:
 - walks every image file (jpg/png) under subset_dir/<sequence>/
 - for each image, ensures labels_dir/<image_basename>.txt exists (creates empty file if missing)
 - prints summary counts at end and writes a small CSV 'data/annotations_yolo/created_empty_labels.csv'
"""
import os
import argparse
from glob import glob

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def find_images(root):
    imgs = []
    for seq in sorted(os.listdir(root)):
        seqp = os.path.join(root, seq)
        if not os.path.isdir(seqp):
            continue
        for ext in IMG_EXTS:
            imgs.extend(glob(os.path.join(seqp, f"*{ext}")))
    return sorted(imgs)

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset_dir", required=True, help="Root folder with subset images (e.g. data/subset)")
    p.add_argument("--labels_dir", required=True, help="Folder where YOLO label .txt will be stored (e.g. data/annotations_yolo/labels)")
    args = p.parse_args()

    subset_dir = args.subset_dir
    labels_dir = args.labels_dir

    if not os.path.isdir(subset_dir):
        raise SystemExit(f"ERROR: subset_dir not found: {subset_dir}")
    ensure_dir(labels_dir)

    images = find_images(subset_dir)
    total_images = len(images)
    created = 0
    skipped = 0
    created_list = []

    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(labels_dir, base + ".txt")
        if not os.path.exists(lbl_path):
            # create empty file
            open(lbl_path, "w").close()
            created += 1
            created_list.append(base + ".txt")
        else:
            skipped += 1

    # write a small CSV of created files for auditing
    csv_out = os.path.join(os.path.dirname(labels_dir), "created_empty_labels.csv")
    with open(csv_out, "w") as f:
        f.write("filename\n")
        for x in created_list:
            f.write(x + "\n")

    print("=== create_empty_labels.py summary ===")
    print(f"Subset images scanned : {total_images}")
    print(f"Existing label files   : {skipped}")
    print(f"Empty labels created   : {created}")
    print(f"CSV of created files   : {csv_out}")
    print("Done.")

if __name__ == '__main__':
    main()
