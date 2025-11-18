#!/usr/bin/env python3
"""
detrac_to_yolo.py
Convert UA-DETRAC annotations (XML/TXT) into YOLO labels + metadata JSON.
"""
import os, json, argparse
from glob import glob
from xml.etree import ElementTree as ET
from PIL import Image

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def convert_bbox_to_yolo(left, top, w, h, img_w, img_h):
    x_c = (left + w/2) / img_w
    y_c = (top + h/2) / img_h
    return x_c, y_c, w/img_w, h/img_h

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frames = {}
    for frame in root.findall(".//frame"):
        fnum = int(frame.get("num") or frame.get("number") or 0)
        for tgt in frame.findall(".//target"):
            box = tgt.find(".//box")
            if box is None: continue
            l, t, w, h = map(float, [box.get("left"), box.get("top"), box.get("width"), box.get("height")])
            occ = tgt.findtext(".//attribute[@name='occlusion']", default=None)
            frames.setdefault(fnum, []).append({"bbox":[l,t,w,h],"occ":occ})
    return frames

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset_dir", required=True)
    p.add_argument("--anno_dir", required=True)
    p.add_argument("--out_labels", required=True)
    p.add_argument("--out_meta", required=True)
    args = p.parse_args()

    ensure_dir(args.out_labels)
    ensure_dir(args.out_meta)

    # iterate each sequence you sampled
    seqs = [d for d in os.listdir(args.subset_dir) if os.path.isdir(os.path.join(args.subset_dir,d))]
    for seq in seqs:
        seq_path = os.path.join(args.subset_dir, seq)
        xml_path = os.path.join(args.anno_dir, seq + ".xml")
        if not os.path.exists(xml_path):
            print(f"⚠️  No annotation for {seq}, skipping")
            continue
        ann = parse_xml(xml_path)
        imgs = sorted(glob(os.path.join(seq_path, "*.jpg")))
        if not imgs: continue

        print(f"Converting {seq} ...")
        for img_path in imgs:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Extract frame number from "imgXXXXX" format
            frame_id = int(base_name.replace("img", ""))
            if frame_id not in ann: continue
            try:
                img_w, img_h = Image.open(img_path).size
            except: continue

            yolo_lines = []
            meta = {"image": img_path, "width": img_w, "height": img_h, "boxes":[]}
            for obj in ann[frame_id]:
                l,t,w,h = obj["bbox"]
                x_c,y_c,wn,hn = convert_bbox_to_yolo(l,t,w,h,img_w,img_h)
                yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {wn:.6f} {hn:.6f}")
                meta["boxes"].append({
                    "bbox":[l,t,w,h],
                    "occlusion":obj["occ"]
                })

            base = os.path.splitext(os.path.basename(img_path))[0]
            with open(os.path.join(args.out_labels, base+".txt"),"w") as f:
                f.write("\n".join(yolo_lines))
            with open(os.path.join(args.out_meta, base+".json"),"w") as f:
                json.dump(meta,f,indent=2)
    print("✅ Conversion complete!")

if __name__ == "__main__":
    main()
