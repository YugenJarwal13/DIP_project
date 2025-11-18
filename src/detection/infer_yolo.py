#!/usr/bin/env python3
"""
infer_yolo.py

Run YOLOv3 inference on the images under data/subset and write per-image detection files.

Two supported model loading modes (choose one):
  1) PyTorch (.pt) using ultralytics/yolov3 hub model (recommended if you trained a .pt with a compatible repo)
     --model_type pt --weights models/yolov3_best.pt
  2) Darknet cfg+weights via OpenCV DNN (original YOLOv3 .cfg + .weights)
     --model_type darknet --cfg externals/yolov3/cfg/yolov3.cfg --weights externals/yolov3/weights/yolov3.weights

Output:
  outputs/detections/<seq>/<img_basename>.txt
Each line:
  x1 y1 x2 y2 score class_id

Usage examples:
  python src/detection/infer_yolo.py --model_type pt --weights models/yolov3_best.pt --input_dir data/subset --out_dir outputs/detections --img_size 416 --conf 0.4
  python src/detection/infer_yolo.py --model_type darknet --cfg externals/yolov3/cfg/yolov3.cfg --weights externals/yolov3/weights/yolov3.weights --input_dir data/subset --out_dir outputs/detections --img_size 416 --conf 0.4

Notes:
 - Runs on CPU by default. If you have CUDA and installed torch with CUDA, PyTorch mode will use GPU.
 - The script is defensive: it creates output directories, skips images it can't open, and prints progress.
"""

import os
import sys
import argparse
import time
from glob import glob
from pathlib import Path

import cv2
import numpy as np

# PyTorch import deferred inside function if not required
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["pt","darknet"], required=True,
                   help="Model type to use: 'pt' for PyTorch .pt (ultralytics), 'darknet' for cfg+weights via OpenCV DNN")
    p.add_argument("--weights", help="Path to a .pt weights file (for model_type pt)")
    p.add_argument("--cfg", help="Path to darknet cfg file (for model_type darknet)")
    p.add_argument("--weights_darknet", dest="weights_darknet", help="Path to darknet .weights file (for model_type darknet)")
    p.add_argument("--input_dir", required=True, help="Root folder with sequences (data/subset)")
    p.add_argument("--out_dir", required=True, help="Where to write outputs/detections")
    p.add_argument("--img_size", type=int, default=416, help="Image size for network (square).")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--device", default="cpu", help="Device to use for PyTorch: 'cpu' or 'cuda:0'")
    p.add_argument("--visualize", action="store_true", help="Save debug overlay images in out_dir/visual")
    return p.parse_args()

# ---------- Utility ----------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def find_sequence_images(root):
    seqs = []
    for entry in sorted(os.listdir(root)):
        seqp = os.path.join(root, entry)
        if os.path.isdir(seqp):
            imgs = []
            for ext in IMG_EXTS:
                imgs.extend(sorted(glob(os.path.join(seqp, f"*{ext}"))))
            if imgs:
                seqs.append((entry, imgs))
    return seqs

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ---------- Darknet (OpenCV DNN) ----------
def prep_darknet_net(cfg_path, weights_path, inp_size):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    # Prefer GPU if OpenCV built with CUDA; otherwise CPU
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    except Exception:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    ln = net.getLayerNames()
    out_names = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
    return net, out_names

def darknet_infer_on_image(net, out_names, img, img_size, conf_thresh, iou_thresh):
    h0, w0 = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (img_size, img_size), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_names)
    # collect detections
    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for det in out.reshape(-1, out.shape[-1]):
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id]) * float(det[4])
            if conf > conf_thresh:
                cx = float(det[0]) * w0
                cy = float(det[1]) * h0
                bw = float(det[2]) * w0
                bh = float(det[3]) * h0
                x1 = int(cx - bw/2)
                y1 = int(cy - bh/2)
                x2 = int(cx + bw/2)
                y2 = int(cy + bh/2)
                boxes.append([x1,y1,x2,y2])
                confidences.append(conf)
                class_ids.append(class_id)
    # NMS
    if len(boxes) == 0:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, iou_thresh)
    res = []
    for i in idxs:
        i = i[0] if isinstance(i, (tuple,list,np.ndarray)) else i
        x1,y1,x2,y2 = boxes[i]
        res.append([x1,y1,x2,y2, confidences[i], class_ids[i]])
    return res

# ---------- PyTorch (ultralytics yolov3 via hub) ----------
def load_pt_model(weights_path, device):
    import torch
    # Try to load ultralytics yolov3 via torch.hub
    try:
        # This line downloads the repo code if not present and builds the model architecture
        model = torch.hub.load('ultralytics/yolov3', 'yolov3', pretrained=False)
        # load state
        ckpt = torch.load(weights_path, map_location=device)
        # ultralytics yolov3 checkpoint may have 'model' key or be plain state_dict
        if isinstance(ckpt, dict) and 'model' in ckpt:
            sd = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
        model.load_state_dict(sd, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load PyTorch YOLOv3 via hub: {e}\nMake sure you trained with a compatible repo (ultralytics/yolov3) or use darknet mode.")

def pt_infer_on_image(model, device, img, img_size, conf_thresh, iou_thresh):
    import torch
    from torchvision import transforms
    h0, w0 = img.shape[:2]
    # Prepare tensor
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_in, (img_size, img_size))
    img_resized = img_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_resized).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(tensor)[0]  # model should return (bs, num_boxes, 6) [x1,y1,x2,y2,conf,class]
    # If model returns in other format, try to adapt
    # preds shape/format is not guaranteed; we try common interpretation:
    boxes = []
    for det in preds.cpu().numpy():
        # Try format [x1,y1,x2,y2,conf,cls]
        if det[4] < conf_thresh:
            continue
        x1 = int(max(0, min(w0-1, det[0])))
        y1 = int(max(0, min(h0-1, det[1])))
        x2 = int(max(0, min(w0-1, det[2])))
        y2 = int(max(0, min(h0-1, det[3])))
        conf = float(det[4])
        cls = int(det[5]) if det.shape[0] > 5 else 0
        boxes.append([x1,y1,x2,y2,conf,cls])
    return boxes

# ---------- Main runner ----------
def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    vis_dir = None
    if args.visualize:
        vis_dir = os.path.join(args.out_dir, "visual")
        ensure_dir(vis_dir)

    seqs = find_sequence_images(args.input_dir)
    if not seqs:
        print("No sequences/images found under", args.input_dir)
        return

    model_pt = None
    net_dark = None
    out_names = None
    device = args.device

    if args.model_type == "pt":
        if not args.weights:
            raise SystemExit("For model_type 'pt' you must provide --weights path to .pt file.")
        try:
            import torch
        except Exception:
            raise SystemExit("PyTorch not available in environment. Install torch in your venv.")
        print("Loading PyTorch YOLOv3 model (this may download repo code if needed)...")
        model_pt = load_pt_model(args.weights, device)
        print("Loaded PyTorch model.")
    else:
        # darknet mode
        if not args.cfg or not args.weights_darknet and not args.weights:
            raise SystemExit("For model_type 'darknet' you must provide --cfg and --weights (use --weights or --weights_darknet).")
        cfg_path = args.cfg
        weights_path = args.weights_darknet if args.weights_darknet else args.weights
        print("Loading Darknet model via OpenCV DNN...")
        net_dark, out_names = prep_darknet_net(cfg_path, weights_path, args.img_size)
        print("Loaded Darknet network.")

    total_images = 0
    t0 = time.time()
    for seq_name, imgs in seqs:
        out_seq_dir = os.path.join(args.out_dir, seq_name)
        ensure_dir(out_seq_dir)
        print(f"[SEQ] {seq_name} -> {len(imgs)} frames")
        for img_path in imgs:
            total_images += 1
            imgname = Path(img_path).stem
            out_txt = os.path.join(out_seq_dir, imgname + ".txt")
            # skip if already exists? we overwrite to be safe
            img = cv2.imread(img_path)
            if img is None:
                print("  ! failed to read", img_path); continue

            if args.model_type == "darknet":
                dets = darknet_infer_on_image(net_dark, out_names, img, args.img_size, args.conf, args.iou)
            else:
                dets = pt_infer_on_image(model_pt, device, img, args.img_size, args.conf, args.iou)

            # write detections
            # format: x1 y1 x2 y2 score class
            with open(out_txt, "w") as f:
                for d in dets:
                    x1,y1,x2,y2,score,cls = d[:6]
                    f.write(f"{int(x1)} {int(y1)} {int(x2)} {int(y2)} {float(score):.4f} {int(cls)}\n")

            # optional visualization
            if args.visualize:
                vis = img.copy()
                for d in dets:
                    x1,y1,x2,y2,score,cls = d[:6]
                    cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                    cv2.putText(vis, f"{cls}:{score:.2f}", (int(x1), int(y1)-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.imwrite(os.path.join(vis_dir, seq_name + "_" + imgname + ".jpg"), vis)

            # progress print every 200 frames
            if total_images % 200 == 0:
                print(f"Processed {total_images} images ...")

    dt = time.time() - t0
    print(f"Done. Processed {total_images} images in {dt:.1f}s ({total_images/dt:.2f} fps)")

if __name__ == "__main__":
    main()
