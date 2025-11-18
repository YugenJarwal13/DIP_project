#!/usr/bin/env python3
import os
import json
import shutil
import argparse
from glob import glob
import math
import random


def list_frames(seq_dir):
    exts = ('*.jpg', '*.png')
    files = []
    for e in exts:
        files.extend(sorted(glob(os.path.join(seq_dir, e))))
    return files


def even_sample(files, n):
    if n >= len(files):
        return files[:]
    step = len(files) / float(n)
    picks = [files[int(i * step)] for i in range(n)]
    return picks


def split_list(items, train_frac=0.8, val_frac=0.1):
    random.shuffle(items)
    train_n = int(len(items) * train_frac)
    val_n = int(len(items) * val_frac)
    train = items[:train_n]
    val = items[train_n:train_n + val_n]
    test = items[train_n + val_n:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_root', required=True, help='UA-DETRAC images parent path')
    parser.add_argument('--plan', required=True, help='JSON plan mapping seq_name->count')
    parser.add_argument('--out', required=True, help='output subset dir (AI_Project/data/subset)')
    parser.add_argument('--configs', required=True, help='configs dir to write train/val/test txt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    with open(args.plan, 'r') as f:
        plan = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    train_all, val_all, test_all = [], [], []

    for seq_name, count in plan.items():
        seq_dir = os.path.join(args.seq_root, seq_name)
        if not os.path.isdir(seq_dir):
            print("Missing seq dir:", seq_dir)
            continue
        files = list_frames(seq_dir)
        if not files:
            print("No images in", seq_dir)
            continue
        picks = even_sample(files, count)
        seq_out = os.path.join(args.out, seq_name)
        os.makedirs(seq_out, exist_ok=True)
        for p in picks:
            base = os.path.basename(p)
            dst = os.path.join(seq_out, base)
            if not os.path.exists(dst):
                shutil.copy2(p, dst)
        picks_local = sorted(glob(os.path.join(seq_out, '*.*')))
        train, val, test = split_list(picks_local, train_frac=0.8, val_frac=0.1)
        train_all.extend(train)
        val_all.extend(val)
        test_all.extend(test)
        print(f"{seq_name}: sampled {len(picks_local)} -> train {len(train)} val {len(val)} test {len(test)}")

    os.makedirs(args.configs, exist_ok=True)

    def write_list(name, items):
        path = os.path.join(args.configs, name)
        with open(path, 'w') as f:
            for it in items:
                f.write(os.path.relpath(it, start=os.getcwd()).replace('\\', '/') + '\n')
        print("Wrote", path, len(items))

    write_list('train.txt', train_all)
    write_list('val.txt', val_all)
    write_list('test.txt', test_all)


if __name__ == '__main__':
    main()
