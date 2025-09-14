#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate source/target lists for 18domains dataset.
Mirrors src/tools/gen_src_tgt.py behavior, but kept separate for clarity.
"""

import argparse
from src.tools.gen_src_tgt import generate_source_and_target_files


def main():
    parser = argparse.ArgumentParser(description="Generate sources/targets for 18domains")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to 18domains root (contains subfolders like grid__..., radial__...)')
    parser.add_argument('--output_dir', type=str, default='source_target_lists',
                        help='Directory to write sources_seed*.txt and targets_seed*.txt')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                        help='Seeds to generate (default: 0)')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    args = parser.parse_args()

    generate_source_and_target_files(
        args.data_dir, args.output_dir, args.seeds,
        train_ratio=args.train_ratio, valid_ratio=args.valid_ratio, test_ratio=args.test_ratio
    )


if __name__ == '__main__':
    main()

