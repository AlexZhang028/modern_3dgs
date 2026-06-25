"""
SharpTimeGS Initialization — CLI entry point.

Usage:
    # Run with a config file (recommended):
    python tools/sharptime_init.py config/sharptime_init_bar.yaml

    # Override individual fields on the command line:
    python tools/sharptime_init.py config/sharptime_init_bar.yaml \\
        --source_path /data/bar-release \\
        --output_path /data/bar-release/sharptime_init.ply \\
        --device cuda

    # Dump a default config template and exit:
    python tools/sharptime_init.py --dump-config config/sharptime_init_template.yaml

The YAML file format is documented in config/sharptime_init_default.yaml.
All pipeline logic lives in initialization/pipeline.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from initialization.init_config import InitConfig
from initialization.pipeline import run_pipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="SharpTimeGS initialization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "config",
        nargs="?",
        help="Path to YAML config file (see config/sharptime_init_default.yaml)",
    )
    p.add_argument(
        "--dump-config",
        metavar="PATH",
        help="Write a default config template to PATH and exit",
    )

    # --- selective field overrides (most commonly changed at runtime) ---
    g = p.add_argument_group("field overrides (take precedence over config file)")
    g.add_argument("--source_path",    help="dataset.source_path")
    g.add_argument("--output_path",    help="dataset.output_path")
    g.add_argument("--work_dir",       help="dataset.work_dir")
    g.add_argument("--start_frame",    type=int, help="dataset.start_frame")
    g.add_argument("--end_frame",      type=int, help="dataset.end_frame")
    g.add_argument("--device",         choices=["cuda", "cpu"], help="device")
    g.add_argument("--overwrite",      action="store_true", help="overwrite=true")
    g.add_argument("--skip_static",    action="store_true", help="static.skip=true")
    g.add_argument("--skip_flow",      action="store_true", help="flow.skip=true")
    g.add_argument("--skip_seg",       action="store_true", help="segmentation.skip=true")
    g.add_argument("--skip_dynamic",   action="store_true", help="dynamic.skip=true")
    g.add_argument("--reference_frame", type=int, help="static.reference_frame")
    g.add_argument("--colmap_bin",     help="static.colmap_bin")

    return p.parse_args()


def main():
    args = parse_args()

    # ---- dump-config mode ----
    if args.dump_config:
        cfg = InitConfig()
        cfg.to_yaml(args.dump_config)
        print(f"Default config written to: {args.dump_config}")
        return

    if not args.config:
        print("Error: provide a config YAML file or use --dump-config.")
        print("       python tools/sharptime_init.py --help")
        sys.exit(1)

    # ---- load config from YAML ----
    cfg = InitConfig.from_yaml(args.config)

    # ---- apply CLI overrides ----
    if args.source_path:    cfg.dataset.source_path    = args.source_path
    if args.output_path:    cfg.dataset.output_path    = args.output_path
    if args.work_dir:       cfg.dataset.work_dir       = args.work_dir
    if args.start_frame is not None: cfg.dataset.start_frame = args.start_frame
    if args.end_frame   is not None: cfg.dataset.end_frame   = args.end_frame
    if args.device:         cfg.device                 = args.device
    if args.overwrite:      cfg.overwrite              = True
    if args.skip_static:    cfg.static.skip            = True
    if args.skip_flow:      cfg.flow.skip              = True
    if args.skip_seg:       cfg.segmentation.skip      = True
    if args.skip_dynamic:   cfg.dynamic.skip           = True
    if args.reference_frame is not None: cfg.static.reference_frame = args.reference_frame
    if args.colmap_bin:     cfg.static.colmap_bin      = args.colmap_bin

    # ---- run ----
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
