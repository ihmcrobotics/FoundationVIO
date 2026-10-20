"""Run FoundationVIO on one sequence, e.g. python -m foundation_vio.run --config configs/scarf.yaml --sequence r05."""

import argparse
import time

import yaml

from foundation_vio.app import FoundationVIO
from foundation_vio.io.load import load_yaml


def set_key(config, path, value):
    *sections, key = path.split(".")
    node = config
    for name in sections:
        node = node.setdefault(name, {})
    node[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="dataset config, configs/<dataset>.yaml")
    parser.add_argument("--sequence", required=True, help="an entry of the config's sequences table, e.g. r05")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value (YAML-parsed); repeatable",
    )
    parser.add_argument("--max-iter", type=int, help="stop after this many batches; omit to run the whole sequence")
    parser.add_argument("--view", action="store_true", help="live Rerun window (SLAM map and trajectory)")
    parser.add_argument("--ceiling", type=float, help="hide points this [m] above the highest pose")
    args = parser.parse_args()

    config = load_yaml(args.config)
    sequences = config.pop("sequences")
    if args.sequence not in sequences:
        raise SystemExit(f"--sequence must be one of: {', '.join(sequences)}")
    for path, value in sequences[args.sequence].items():
        set_key(config, path, value)
    for item in args.set:
        path, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects section.key=value, got {item!r}")
        set_key(config, path, yaml.safe_load(raw))
    if args.max_iter is not None:
        config["sequence"]["max_iter"] = args.max_iter

    app = FoundationVIO(config)
    if args.view:
        from foundation_vio.viz import LiveViewer  # noqa: PLC0415 - Rerun only when --view

        app.viewer = LiveViewer(ceiling=args.ceiling)
    app.load()
    state, total_batches = app.prep_iterate()
    t0 = time.perf_counter()
    while state.batch_id < total_batches and app.iterate(state) is not None:
        pass
    state.pbar.close()
    loop_s = time.perf_counter() - t0
    last = min(state.batch_id * app.stride + app.batch_overlap - 1, len(app.t_ns) - 1)  # last image processed
    footage_s = (app.t_ns[last] - app.t_ns[0]) * 1e-9
    print(
        f"Batch loop: {loop_s:.1f} s for {footage_s:.1f} s of footage ({loop_s / max(footage_s, 1e-9):.2f}x real time)"
    )
    final = app.optimize(state)
    if app.viewer is not None:
        app.viewer.relocate(app, final)
    app.save_data(final)
    if app.viewer is not None:
        app.viewer.hold()


if __name__ == "__main__":
    main()
