#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from validate_dataset import (
    ValidationReport,
    collect_hdf5_files,
    collect_raw_demo_files,
    detect_format,
    feature_dimension_names,
    find_gripper_index,
    h5py,
    np,
    pq,
    require_h5py,
    require_numpy,
    require_pyarrow,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
except ImportError as exc:  # pragma: no cover - runtime dependency
    matplotlib = None
    plt = None
    LineCollection = None
    Normalize = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-trajectory gripper signals for a dataset. "
            "Each trajectory is drawn as one colored line."
        )
    )
    parser.add_argument("path", type=Path, help="Dataset directory or a single HDF5 file.")
    parser.add_argument(
        "--format",
        choices=("auto", "raw_demos_npz", "lerobot_v21", "xval_hdf5"),
        default="auto",
        help="Dataset format. Default: auto.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output image path, for example quality_test/reports/gripper.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI. Default: 180.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.0,
        help="Line width for each trajectory. Default: 2.0.",
    )
    parser.add_argument(
        "--episode-height",
        type=float,
        default=0.32,
        help="Vertical height allocated per trajectory. Default: 0.32.",
    )
    return parser.parse_args()


def require_matplotlib() -> None:
    if matplotlib is None or plt is None or LineCollection is None or Normalize is None:
        raise SystemExit(
            "Missing dependency: matplotlib. Install it with `python -m pip install matplotlib`."
        ) from _MATPLOTLIB_IMPORT_ERROR


def load_lerobot_gripper_trajectories(
    root: Path,
) -> tuple[list[np.ndarray], list[str], int | None]:
    report = ValidationReport(dataset_format="lerobot_v21", root=root)
    if not require_pyarrow(report):
        raise SystemExit(report.errors[0])

    import json

    info_path = root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)

    features = info.get("features", {})
    state_feature = features.get("observation.state")
    if not isinstance(state_feature, dict):
        raise SystemExit("The LeRobot dataset is missing the observation.state feature.")

    state_names = feature_dimension_names(state_feature, "state")
    gripper_index = find_gripper_index(state_names)
    if gripper_index is None:
        raise SystemExit("Could not find a gripper dimension in LeRobot observation.state.")

    total_episodes = int(info.get("total_episodes", 0))
    chunks_size = int(info.get("chunks_size", 1000))
    data_path_pattern = str(info.get("data_path"))
    fps = int(info.get("fps", 0)) or None

    trajectories: list[np.ndarray] = []
    labels: list[str] = []
    for episode_index in range(total_episodes):
        chunk_index = episode_index // chunks_size
        parquet_path = root / data_path_pattern.format(
            episode_chunk=chunk_index,
            episode_index=episode_index,
        )
        if not parquet_path.exists():
            raise SystemExit(f"Missing parquet file: {parquet_path}")

        table = pq.read_table(parquet_path, columns=["observation.state"])
        column = table.column("observation.state").combine_chunks()
        width = column.type.list_size
        values = column.values.to_numpy(zero_copy_only=False)
        state_array = np.asarray(values).reshape(len(column), width)
        trajectories.append(state_array[:, gripper_index].astype(np.float64, copy=False))
        labels.append(parquet_path.stem)

    return trajectories, labels, fps


def load_xval_gripper_trajectories(path: Path) -> tuple[list[np.ndarray], list[str], None]:
    report = ValidationReport(dataset_format="xval_hdf5", root=path)
    if not require_h5py(report):
        raise SystemExit(report.errors[0])

    hdf5_files = collect_hdf5_files(path)
    if not hdf5_files:
        raise SystemExit(f"No HDF5 files were found under {path}.")

    trajectories: list[np.ndarray] = []
    labels: list[str] = []
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as handle:
            if "puppet/joint_position" not in handle:
                raise SystemExit(f"{hdf5_path} is missing puppet/joint_position.")
            joint_position = np.asarray(handle["puppet/joint_position"], dtype=np.float64)
            if joint_position.ndim != 2 or joint_position.shape[1] < 1:
                raise SystemExit(
                    f"Invalid puppet/joint_position shape in {hdf5_path}: {joint_position.shape}"
                )
            trajectories.append(joint_position[:, -1])
            labels.append(hdf5_path.name)

    return trajectories, labels, None


def load_raw_demo_gripper_trajectories(
    path: Path,
) -> tuple[list[np.ndarray], list[str], int | None]:
    require_numpy()
    raw_demo_files = collect_raw_demo_files(path)
    if not raw_demo_files:
        raise SystemExit(f"No raw demo npz files were found under {path}.")

    trajectories: list[np.ndarray] = []
    labels: list[str] = []
    fps_values: set[int] = set()
    for npz_path in raw_demo_files:
        with np.load(npz_path, allow_pickle=True) as data:
            if "gripper" not in data or "fps" not in data:
                raise SystemExit(f"{npz_path} is missing gripper or fps.")
            gripper = np.asarray(data["gripper"], dtype=np.float64)
            if gripper.ndim != 1:
                raise SystemExit(f"Invalid gripper shape in {npz_path}: {gripper.shape}")
            trajectories.append(gripper)
            labels.append(npz_path.name)
            fps_values.add(int(np.asarray(data["fps"]).item()))

    fps = next(iter(fps_values)) if len(fps_values) == 1 else None
    return trajectories, labels, fps


def build_segments(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    points = np.column_stack([x_values, y_values])
    return np.stack([points[:-1], points[1:]], axis=1)


def draw_gripper_plot(
    *,
    trajectories: list[np.ndarray],
    labels: list[str],
    output_path: Path,
    title: str,
    dpi: int,
    line_width: float,
    episode_height: float,
    fps: int | None,
) -> None:
    require_matplotlib()
    if not trajectories:
        raise SystemExit("No trajectories available for plotting.")
    if len(trajectories) != len(labels):
        raise SystemExit("The number of trajectory labels does not match the number of trajectories.")

    max_length = max(len(values) for values in trajectories)
    figure_height = max(4.0, len(trajectories) * episode_height + 1.5)
    max_label_length = max((len(label) for label in labels), default=0)
    left_margin_bonus = min(10.0, max_label_length * 0.09)
    figure_width = max(10.0 + left_margin_bonus, min(28.0, max_length / 80.0 + 3.0 + left_margin_bonus))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), constrained_layout=True)

    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("RdYlGn")
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for idx, gripper_values in enumerate(trajectories):
        if len(gripper_values) == 0:
            continue

        y_center = float(idx)
        amplitude = 0.36
        x_values = np.arange(len(gripper_values), dtype=np.float64)

        if len(gripper_values) == 1:
            ax.scatter(
                x_values,
                np.asarray([y_center]),
                c=gripper_values,
                cmap=cmap,
                norm=norm,
                s=14,
                zorder=3,
            )
        else:
            y_values = y_center + (gripper_values - 0.5) * amplitude
            segments = build_segments(x_values, y_values)
            colors = 0.5 * (gripper_values[:-1] + gripper_values[1:])
            collection = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidths=line_width,
            )
            collection.set_array(colors)
            ax.add_collection(collection)

        tick_positions.append(y_center)
        tick_labels.append(labels[idx])

    ax.set_xlim(0, max_length - 1 if max_length > 0 else 1)
    ax.set_ylim(-0.8, len(trajectories) - 0.2)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_ylabel("File name")

    if fps is not None and fps > 0:
        ax.set_xlabel(f"Frame index (fps={fps})")
    else:
        ax.set_xlabel("Frame index")

    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax, pad=0.02)
    colorbar.set_label("Gripper value")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    require_numpy()
    args = parse_args()
    input_path = args.path.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Path does not exist: {input_path}")

    dataset_format = args.format
    if dataset_format == "auto":
        dataset_format = detect_format(input_path)

    if dataset_format == "lerobot_v21":
        trajectories, labels, fps = load_lerobot_gripper_trajectories(input_path)
        default_title = f"LeRobot v2.1 gripper trajectories: {input_path.name}"
    elif dataset_format == "raw_demos_npz":
        trajectories, labels, fps = load_raw_demo_gripper_trajectories(input_path)
        default_title = f"raw_demos gripper trajectories: {input_path.name}"
    elif dataset_format == "xval_hdf5":
        trajectories, labels, fps = load_xval_gripper_trajectories(input_path)
        default_title = f"XVAL HDF5 gripper trajectories: {input_path.name}"
    else:  # pragma: no cover - argparse prevents this
        raise SystemExit(f"Unsupported dataset format: {dataset_format}")

    draw_gripper_plot(
        trajectories=trajectories,
        labels=labels,
        output_path=output_path,
        title=args.title or default_title,
        dpi=args.dpi,
        line_width=args.line_width,
        episode_height=args.episode_height,
        fps=fps,
    )

    print(f"[Done] Generated gripper trajectory plot: {output_path}")
    print(f"[Info] format={dataset_format}, trajectories={len(trajectories)}")
    print(f"[Info] max_length={max((len(values) for values in trajectories), default=0)}")


if __name__ == "__main__":
    main()
