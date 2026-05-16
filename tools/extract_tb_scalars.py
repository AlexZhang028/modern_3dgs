#!/usr/bin/env python3
"""
Extract scalar tags from TensorBoard event files into CSV files.

Usage examples:
  python tools/extract_tb_scalars.py --logdir output/bar-release/logs --outdir /tmp/tb_csv
  python tools/extract_tb_scalars.py --logdir runs --outdir /tmp/tb_csv --tag-prefix Timing/ --merge

By default the script writes one CSV per scalar tag with columns: `wall_time,step,value`.
If `--merge` is passed, it will also write `merged.csv` with rows keyed by `step` and columns for each tag.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception as e:
    raise RuntimeError("tensorboard is required. Install with `pip install tensorboard`") from e


def find_event_files(logdir: str) -> List[str]:
    pattern = os.path.join(logdir, "**", "events.*")
    files = glob.glob(pattern, recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    return files


def sanitize_tag(tag: str) -> str:
    return tag.replace('/', '__').replace(' ', '_')


def extract_scalars_from_file(path: str) -> Dict[str, List[Tuple[float, int, float]]]:
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tag_scalars: Dict[str, List[Tuple[float, int, float]]] = {}
    tags = ea.Tags().get('scalars', [])
    for tag in tags:
        scalars = ea.Scalars(tag)
        # Each ScalarEvent: wall_time, step, value
        tag_scalars[tag] = [(s.wall_time, s.step, s.value) for s in scalars]
    return tag_scalars


def merge_tag_lists(all_lists: List[Dict[str, List[Tuple[float, int, float]]]]) -> Dict[str, List[Tuple[float, int, float]]]:
    merged: Dict[str, List[Tuple[float, int, float]]] = defaultdict(list)
    for d in all_lists:
        for tag, vals in d.items():
            merged[tag].extend(vals)
    # remove duplicates by (step, value) if any and sort by step
    for tag in list(merged.keys()):
        seen = set()
        uniq = []
        for w, s, v in merged[tag]:
            key = (s, v)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((w, s, v))
        uniq.sort(key=lambda x: x[1])
        merged[tag] = uniq
    return merged


def write_per_tag_csv(outdir: str, tag: str, rows: List[Tuple[float, int, float]]):
    fname = os.path.join(outdir, f"{sanitize_tag(tag)}.csv")
    os.makedirs(outdir, exist_ok=True)
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['wall_time', 'step', 'value'])
        for wt, step, val in rows:
            w.writerow([f"{wt:.6f}", step, f"{val:.6g}"])


def write_merged_csv(outdir: str, merged: Dict[str, List[Tuple[float, int, float]]]):
    # Build set of all steps
    steps = set()
    for vals in merged.values():
        for _, s, _ in vals:
            steps.add(s)
    steps = sorted(steps)
    tags = sorted(merged.keys())
    fname = os.path.join(outdir, 'merged.csv')
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['step'] + tags
        w.writerow(header)
        # build dict per tag for quick lookup
        tag_dict = {tag: {s: v for _, s, v in vals} for tag, vals in merged.items()}
        for s in steps:
            row = [s]
            for tag in tags:
                v = tag_dict.get(tag, {}).get(s, '')
                row.append(v)
            w.writerow(row)


def main():
    p = argparse.ArgumentParser(description='Extract TensorBoard scalar tags to CSV')
    p.add_argument('--logdir', '-l', required=True, help='TensorBoard log directory (searched recursively)')
    p.add_argument('--outdir', '-o', required=True, help='Output directory for CSV files')
    p.add_argument('--tag-prefix', '-t', default=None, help='Optional tag prefix filter (e.g. "Timing/")')
    p.add_argument('--merge', action='store_true', help='Also produce merged.csv combining tags by step')
    args = p.parse_args()

    files = find_event_files(args.logdir)
    if not files:
        print(f"No event files found under {args.logdir!r}")
        return

    all_tag_lists = []
    for fpath in files:
        try:
            tag_scalars = extract_scalars_from_file(fpath)
            all_tag_lists.append(tag_scalars)
        except Exception as exc:
            print(f"Warning: failed to read {fpath}: {exc}")

    merged = merge_tag_lists(all_tag_lists)

    # Apply prefix filter if provided
    if args.tag_prefix:
        merged = {k: v for k, v in merged.items() if k.startswith(args.tag_prefix)}

    if not merged:
        print("No scalar tags found (after filtering).")
        return

    os.makedirs(args.outdir, exist_ok=True)
    for tag, rows in merged.items():
        write_per_tag_csv(args.outdir, tag, rows)
    if args.merge:
        write_merged_csv(args.outdir, merged)

    print(f"Wrote {len(merged)} tag CSV files to {args.outdir}")
    if args.merge:
        print(f"Wrote merged CSV to {os.path.join(args.outdir, 'merged.csv')}")


if __name__ == '__main__':
    main()
