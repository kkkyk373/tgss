#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gene_src_tgt.py

ComOD-dataset/data/ 内のエリアIDから、
split_train_valid_test を使って test エリア（ターゲット）と
train+valid エリア（ソース）を分割し、
指定された SEED 値ごとに以下を出力：

- targets_seed{seed}.txt
- sources_seed{seed}.txt
"""

import os
import random
import argparse
from src.utils.split_areas import load_all_areas, split_train_valid_test


def generate_source_and_target_files(data_dir, output_dir, seeds, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    os.makedirs(output_dir, exist_ok=True)
    base_areas = load_all_areas(dir_path=data_dir, if_shuffle=False)

    for seed in seeds:
        random.seed(seed)
        areas = base_areas.copy()
        random.shuffle(areas)

        train, valid, test = split_train_valid_test(
            areas,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio
        )

        # 出力パス
        tgt_path = os.path.join(output_dir, f"targets_seed{seed}.txt")
        src_path = os.path.join(output_dir, f"sources_seed{seed}.txt")

        with open(tgt_path, "w") as f:
            for area in test:
                f.write(f"{area}\n")

        with open(src_path, "w") as f:
            for area in (train + valid):
                f.write(f"{area}\n")

        print(f"✅ Seed {seed} → {len(train + valid)} sources / {len(test)} targets")


def main():
    parser = argparse.ArgumentParser(description="Generate source and target area lists from data directory.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to directory containing area data (e.g., ComOD-dataset/data)")
    parser.add_argument('--output_dir', type=str, default="source_target_lists",
                        help="Directory to save sources_seed*.txt and targets_seed*.txt files")
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)),
                        help="List of seed integers to use (default: 0-9)")
    args = parser.parse_args()

    generate_source_and_target_files(args.data_dir, args.output_dir, args.seeds)


if __name__ == "__main__":
    main()
