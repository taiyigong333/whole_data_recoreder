#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime dependency
    np = None
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - runtime dependency
    pa = None
    pq = None
    _PYARROW_IMPORT_ERROR = exc
else:
    _PYARROW_IMPORT_ERROR = None

try:
    import h5py
except ImportError as exc:  # pragma: no cover - runtime dependency
    h5py = None
    _H5PY_IMPORT_ERROR = exc
else:
    _H5PY_IMPORT_ERROR = None


LEROBOT_META_FILES = (
    "meta/info.json",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
    "meta/tasks.jsonl",
)
RAW_DEMO_REQUIRED_KEYS = (
    "images",
    "tcp_poses",
    "gripper",
    "instruction",
    "fps",
)
DEFAULT_GRIPPER_NAMES = {"gripper", "gripper_position", "grip", "grip_pos"}
DEFAULT_JOINT_PREFIXES = ("joint", "shoulder", "elbow", "wrist")


@dataclass
class ValidationReport:
    dataset_format: str
    root: Path
    summary: dict[str, Any] = field(default_factory=dict)
    ranges: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_note(self, message: str) -> None:
        self.notes.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.dataset_format,
            "path": str(self.root),
            "passed": self.passed,
            "summary": self.summary,
            "ranges": self.ranges,
            "warnings": self.warnings,
            "errors": self.errors,
            "notes": self.notes,
        }


class VectorRangeAccumulator:
    def __init__(self, names: list[str]):
        self.names = list(names)
        self.minimum = None
        self.maximum = None

    def update(self, values) -> None:
        array = _ensure_2d_float(values)
        if array.size == 0:
            return
        current_min = array.min(axis=0)
        current_max = array.max(axis=0)
        if self.minimum is None:
            self.minimum = current_min
            self.maximum = current_max
            return
        self.minimum = np.minimum(self.minimum, current_min)
        self.maximum = np.maximum(self.maximum, current_max)

    def is_empty(self) -> bool:
        return self.minimum is None or self.maximum is None

    def to_named_dict(self) -> dict[str, dict[str, float]]:
        if self.is_empty():
            return {}
        return {
            name: {
                "min": _to_python_float(self.minimum[idx]),
                "max": _to_python_float(self.maximum[idx]),
            }
            for idx, name in enumerate(self.names)
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate XVAL-Code HDF5 datasets and LeRobot Dataset v2.1 datasets. "
            "The tool checks integrity, missing values, range statistics, and additional consistency rules."
        )
    )
    parser.add_argument("path", type=Path, help="Dataset directory or single HDF5 file.")
    parser.add_argument(
        "--format",
        choices=("auto", "raw_demos_npz", "lerobot_v21", "xval_hdf5"),
        default="auto",
        help="Dataset format. Default: auto",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full report as JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when warnings exist.",
    )
    return parser.parse_args()


def require_numpy() -> None:
    if np is None:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "Missing dependency: numpy. Install it before running this validator."
        ) from _NUMPY_IMPORT_ERROR


def require_pyarrow(report: ValidationReport) -> bool:
    if pa is None or pq is None:  # pragma: no cover - runtime dependency
        report.add_error(
            "Missing dependency: pyarrow. Install it before validating a LeRobot v2.1 dataset."
        )
        return False
    return True


def require_h5py(report: ValidationReport) -> bool:
    if h5py is None:  # pragma: no cover - runtime dependency
        report.add_error(
            "Missing dependency: h5py. Install it before validating an XVAL-Code HDF5 dataset."
        )
        return False
    return True


def _ensure_2d_float(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _to_python_float(value: Any) -> float:
    return float(np.asarray(value).item())


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            content = line.strip()
            if not content:
                continue
            try:
                row = json.loads(content)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"Invalid JSONL row at {path}:{line_no}: expected an object.")
            rows.append(row)
    return rows


def detect_format(path: Path) -> str:
    if path.is_file() and path.suffix.lower() in {".hdf5", ".h5"}:
        return "xval_hdf5"
    if path.is_dir() and all((path / meta_file).exists() for meta_file in LEROBOT_META_FILES):
        return "lerobot_v21"
    if path.is_dir():
        npz_files = sorted(path.glob("episode_*.npz"))
        if npz_files:
            return "raw_demos_npz"
        hdf5_files = sorted(path.glob("*.hdf5")) + sorted(path.glob("*.h5"))
        if hdf5_files:
            return "xval_hdf5"
    raise SystemExit(
        "Unable to detect dataset format. Pass --format explicitly or provide a valid "
        "raw_demos directory / LeRobot v2.1 dataset directory / XVAL-Code HDF5 file or directory."
    )


def _print_report(report: ValidationReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(f"[{status}] format={report.dataset_format} path={report.root}")
    if report.summary:
        print("Summary:")
        for key in sorted(report.summary):
            print(f"  - {key}: {report.summary[key]}")
    if report.ranges:
        print("Ranges:")
        for section in sorted(report.ranges):
            value = report.ranges[section]
            print(f"  - {section}:")
            if isinstance(value, dict):
                for inner_key in sorted(value):
                    print(f"      {inner_key}: {value[inner_key]}")
            else:
                print(f"      {value}")
    if report.notes:
        print("Notes:")
        for message in report.notes:
            print(f"  - {message}")
    if report.warnings:
        print("Warnings:")
        for message in report.warnings:
            print(f"  - {message}")
    if report.errors:
        print("Errors:")
        for message in report.errors:
            print(f"  - {message}")


def _write_json_report(report: ValidationReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _arrow_recursive_null_count(array) -> int:
    count = int(array.null_count)
    if pa.types.is_fixed_size_list(array.type) or pa.types.is_list(array.type):
        count += _arrow_recursive_null_count(array.values)
    return count


def _arrow_array_to_numpy(array):
    if pa.types.is_fixed_size_list(array.type):
        width = array.type.list_size
        values = array.values.to_numpy(zero_copy_only=False)
        return np.asarray(values).reshape(len(array), width)
    if pa.types.is_list(array.type):
        rows = array.to_pylist()
        return np.asarray(rows)
    return np.asarray(array.to_numpy(zero_copy_only=False))


def feature_dimension_names(feature: dict[str, Any], fallback_prefix: str) -> list[str]:
    names = feature.get("names")
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    shape = feature.get("shape") or []
    if shape and len(shape) == 1 and int(shape[0]) > 0:
        return [f"{fallback_prefix}_{idx}" for idx in range(int(shape[0]))]
    return [fallback_prefix]


def find_gripper_index(names: list[str]) -> int | None:
    for idx, name in enumerate(names):
        if str(name).lower() in DEFAULT_GRIPPER_NAMES:
            return idx
    return None


def _find_joint_indices(names: list[str]) -> list[int]:
    indices: list[int] = []
    for idx, name in enumerate(names):
        lowered = str(name).lower()
        if lowered.startswith(DEFAULT_JOINT_PREFIXES):
            indices.append(idx)
    return indices


def _validate_numeric_array(
    *,
    array,
    label: str,
    report: ValidationReport,
) -> Any:
    if array.size == 0:
        report.add_error(f"{label}: empty array.")
        return array
    if np.issubdtype(array.dtype, np.number):
        if np.isnan(array).any():
            report.add_error(f"{label}: contains NaN values.")
        if np.isinf(array).any():
            report.add_error(f"{label}: contains Inf values.")
    return array


def collect_raw_demo_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".npz":
            return []
        return [path]
    return sorted(path.glob("episode_*.npz"))


def validate_raw_demos_npz(path: Path) -> ValidationReport:
    require_numpy()
    report = ValidationReport(dataset_format="raw_demos_npz", root=path)

    raw_demo_files = collect_raw_demo_files(path)
    if not raw_demo_files:
        report.add_error(f"No raw demo npz files were found under {path}.")
        return report

    tcp_names = ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"]
    tcp_ranges = VectorRangeAccumulator(tcp_names)
    gripper_range = VectorRangeAccumulator(["gripper"])

    fps_values: set[int] = set()
    instructions: set[str] = set()
    main_shapes: set[tuple[int, int, int]] = set()
    wrist_shapes: set[tuple[int, int, int]] = set()
    has_wrist_values: set[bool] = set()

    total_frames = 0
    min_length = None
    max_length = None

    for npz_path in raw_demo_files:
        with np.load(npz_path, allow_pickle=True) as data:
            data_files = list(data.files)
            missing_keys = [key for key in RAW_DEMO_REQUIRED_KEYS if key not in data_files]
            if missing_keys:
                report.add_error(f"{npz_path}: missing required keys {missing_keys}.")
                continue

            images = np.asarray(data["images"])
            tcp_poses = np.asarray(data["tcp_poses"])
            gripper = np.asarray(data["gripper"])
            instruction = str(np.asarray(data["instruction"]).item())
            fps = int(np.asarray(data["fps"]).item())
            images_wrist = np.asarray(data["images_wrist"]) if "images_wrist" in data_files else None

            _validate_numeric_array(
                array=tcp_poses,
                label=f"{npz_path}: tcp_poses",
                report=report,
            )
            _validate_numeric_array(
                array=gripper,
                label=f"{npz_path}: gripper",
                report=report,
            )

            if images.ndim != 4 or images.shape[-1] != 3:
                report.add_error(
                    f"{npz_path}: images must have shape (T, H, W, 3), got {images.shape}."
                )
                continue
            if images.dtype != np.uint8:
                report.add_warning(
                    f"{npz_path}: images dtype is {images.dtype}, expected uint8."
                )
            if images.min() < 0 or images.max() > 255:
                report.add_warning(
                    f"{npz_path}: images values fall outside the common [0, 255] range."
                )

            if tcp_poses.ndim != 2 or tcp_poses.shape[1] != 6:
                report.add_error(
                    f"{npz_path}: tcp_poses must have shape (T, 6), got {tcp_poses.shape}."
                )
                continue
            if gripper.ndim != 1:
                report.add_error(f"{npz_path}: gripper must be a 1D array, got {gripper.shape}.")
                continue

            frame_count = int(images.shape[0])
            total_frames += frame_count
            min_length = frame_count if min_length is None else min(min_length, frame_count)
            max_length = frame_count if max_length is None else max(max_length, frame_count)

            if frame_count <= 0:
                report.add_error(f"{npz_path}: empty trajectory.")
                continue
            if tcp_poses.shape[0] != frame_count:
                report.add_error(
                    f"{npz_path}: frame count mismatch between images ({frame_count}) "
                    f"and tcp_poses ({tcp_poses.shape[0]})."
                )
            if gripper.shape[0] != frame_count:
                report.add_error(
                    f"{npz_path}: frame count mismatch between images ({frame_count}) "
                    f"and gripper ({gripper.shape[0]})."
                )

            if images_wrist is not None:
                if images_wrist.ndim != 4 or images_wrist.shape[-1] != 3:
                    report.add_error(
                        f"{npz_path}: images_wrist must have shape (T, H, W, 3), got {images_wrist.shape}."
                    )
                elif images_wrist.shape[0] != frame_count:
                    report.add_error(
                        f"{npz_path}: frame count mismatch between images ({frame_count}) "
                        f"and images_wrist ({images_wrist.shape[0]})."
                    )
                wrist_shapes.add(tuple(int(value) for value in images_wrist.shape[1:]))

            main_shapes.add(tuple(int(value) for value in images.shape[1:]))
            has_wrist_values.add(images_wrist is not None)
            fps_values.add(fps)
            instructions.add(instruction)

            tcp_ranges.update(tcp_poses)
            gripper_range.update(gripper)

            if np.allclose(gripper, gripper[0], atol=1e-8):
                report.add_warning(
                    f"{npz_path}: gripper value is constant ({float(gripper[0])})."
                )
            if gripper.min() < 0.0 or gripper.max() > 1.0:
                report.add_warning(
                    f"{npz_path}: gripper values fall outside the common [0, 1] range."
                )
            if not instruction.strip():
                report.add_warning(f"{npz_path}: instruction is empty.")
            if fps <= 0:
                report.add_error(f"{npz_path}: fps must be positive, got {fps}.")

    if len(main_shapes) > 1:
        report.add_error(f"images resolution mismatch across raw demos: {sorted(main_shapes)}.")
    if len(wrist_shapes) > 1:
        report.add_error(
            f"images_wrist resolution mismatch across raw demos: {sorted(wrist_shapes)}."
        )
    if len(has_wrist_values) > 1:
        report.add_error("Some raw demos contain images_wrist while others do not.")
    if len(fps_values) > 1:
        report.add_error(f"fps mismatch across raw demos: {sorted(fps_values)}.")

    report.ranges["tcp_poses"] = tcp_ranges.to_named_dict()
    report.ranges["gripper_position"] = gripper_range.to_named_dict()
    report.summary = {
        "episodes": len(raw_demo_files),
        "frames": total_frames,
        "fps_values": sorted(fps_values),
        "instructions": sorted(instructions),
        "has_wrist_camera": sorted(has_wrist_values),
        "main_image_shapes": sorted(main_shapes),
        "wrist_image_shapes": sorted(wrist_shapes),
        "min_length": min_length,
        "max_length": max_length,
    }
    return report


def validate_lerobot_v21(root: Path) -> ValidationReport:
    require_numpy()
    report = ValidationReport(dataset_format="lerobot_v21", root=root)
    if not require_pyarrow(report):
        return report

    for meta_file in LEROBOT_META_FILES:
        if not (root / meta_file).exists():
            report.add_error(f"Missing required file: {root / meta_file}")
    if report.errors:
        return report

    info = _load_json(root / "meta" / "info.json")
    episodes_rows = _load_jsonl(root / "meta" / "episodes.jsonl")
    tasks_rows = _load_jsonl(root / "meta" / "tasks.jsonl")
    episodes_stats_rows = _load_jsonl(root / "meta" / "episodes_stats.jsonl")

    codebase_version = str(info.get("codebase_version"))
    if codebase_version != "v2.1":
        report.add_error(
            f"Expected LeRobot codebase_version=v2.1, but found {codebase_version!r}."
        )

    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    fps = int(info.get("fps", 0))
    chunks_size = int(info.get("chunks_size", 1000))
    data_path_pattern = info.get("data_path")
    video_path_pattern = info.get("video_path")
    features = info.get("features", {})

    if not isinstance(features, dict) or not features:
        report.add_error("meta/info.json: missing or invalid features section.")
        return report

    state_feature = features.get("observation.state")
    action_feature = features.get("action")
    if not isinstance(state_feature, dict):
        report.add_error("meta/info.json: missing observation.state feature.")
        return report
    if not isinstance(action_feature, dict):
        report.add_error("meta/info.json: missing action feature.")
        return report

    state_names = feature_dimension_names(state_feature, "state")
    action_names = feature_dimension_names(action_feature, "action")
    state_ranges = VectorRangeAccumulator(state_names)
    action_ranges = VectorRangeAccumulator(action_names)

    gripper_index = find_gripper_index(state_names)
    joint_indices = _find_joint_indices(state_names)
    gripper_range = VectorRangeAccumulator(["gripper"]) if gripper_index is not None else None
    joint_range = (
        VectorRangeAccumulator([state_names[idx] for idx in joint_indices])
        if joint_indices
        else None
    )

    expected_parquet_feature_keys = sorted(
        key for key, feature in features.items() if feature.get("dtype") != "video"
    )
    video_keys = sorted(
        key for key, feature in features.items() if feature.get("dtype") == "video"
    )

    episodes_by_index: dict[int, dict[str, Any]] = {}
    for row in episodes_rows:
        if "episode_index" not in row:
            report.add_error("meta/episodes.jsonl: row without episode_index.")
            continue
        episode_index = int(row["episode_index"])
        if episode_index in episodes_by_index:
            report.add_error(f"meta/episodes.jsonl: duplicate episode_index={episode_index}.")
            continue
        episodes_by_index[episode_index] = row

    tasks_by_index: dict[int, str] = {}
    task_names = set()
    for row in tasks_rows:
        if "task_index" not in row or "task" not in row:
            report.add_error("meta/tasks.jsonl: row without task_index/task.")
            continue
        task_index = int(row["task_index"])
        task_text = str(row["task"])
        tasks_by_index[task_index] = task_text
        task_names.add(task_text)

    episode_stats_by_index: dict[int, dict[str, Any]] = {}
    for row in episodes_stats_rows:
        if "episode_index" not in row or "stats" not in row:
            report.add_error("meta/episodes_stats.jsonl: row without episode_index/stats.")
            continue
        episode_index = int(row["episode_index"])
        if episode_index in episode_stats_by_index:
            report.add_error(f"meta/episodes_stats.jsonl: duplicate episode_index={episode_index}.")
            continue
        episode_stats_by_index[episode_index] = row["stats"]

    if len(episodes_by_index) != total_episodes:
        report.add_error(
            f"Episode count mismatch: info.total_episodes={total_episodes}, "
            f"episodes.jsonl={len(episodes_by_index)}."
        )
    if len(episode_stats_by_index) != total_episodes:
        report.add_error(
            f"Episode stats count mismatch: info.total_episodes={total_episodes}, "
            f"episodes_stats.jsonl={len(episode_stats_by_index)}."
        )

    parquet_files = sorted(root.glob("data/chunk-*/*.parquet"))
    if len(parquet_files) != total_episodes:
        report.add_error(
            f"Parquet file count mismatch: expected {total_episodes}, found {len(parquet_files)}."
        )

    if int(info.get("total_videos", 0)) != total_episodes * len(video_keys):
        report.add_error(
            f"total_videos mismatch: info.total_videos={info.get('total_videos')}, "
            f"expected={total_episodes * len(video_keys)}."
        )

    total_rows = 0
    expected_global_index = 0
    max_timestamp_deviation = 0.0

    for episode_index in range(total_episodes):
        episode_meta = episodes_by_index.get(episode_index)
        if episode_meta is None:
            report.add_error(f"Missing episode metadata for episode_index={episode_index}.")
            continue

        chunk_index = episode_index // chunks_size
        parquet_path = root / str(data_path_pattern).format(
            episode_chunk=chunk_index,
            episode_index=episode_index,
        )
        if not parquet_path.exists():
            report.add_error(f"Missing parquet file: {parquet_path}")
            continue

        table = pq.read_table(parquet_path)
        row_count = int(table.num_rows)
        total_rows += row_count

        expected_length = int(episode_meta.get("length", -1))
        if row_count != expected_length:
            report.add_error(
                f"Episode {episode_index}: parquet rows={row_count}, meta length={expected_length}."
            )
        if row_count <= 0:
            report.add_error(f"Episode {episode_index}: empty parquet file.")
            continue

        actual_columns = sorted(table.column_names)
        if actual_columns != expected_parquet_feature_keys:
            report.add_error(
                f"Episode {episode_index}: parquet schema mismatch. "
                f"expected={expected_parquet_feature_keys}, actual={actual_columns}."
            )

        columns_np: dict[str, Any] = {}
        for column_name in table.column_names:
            column = table.column(column_name).combine_chunks()
            null_count = _arrow_recursive_null_count(column)
            if null_count > 0:
                report.add_error(
                    f"Episode {episode_index}: column {column_name} contains {null_count} null values."
                )
            array = _arrow_array_to_numpy(column)
            columns_np[column_name] = array

            expected_feature = features.get(column_name)
            if isinstance(expected_feature, dict) and expected_feature.get("shape"):
                shape = list(expected_feature.get("shape") or [])
                if column_name in ("observation.state", "action"):
                    expected_width = int(shape[0])
                    if array.ndim != 2 or array.shape[1] != expected_width:
                        report.add_error(
                            f"Episode {episode_index}: {column_name} shape mismatch. "
                            f"expected (*, {expected_width}), actual={tuple(array.shape)}."
                        )

            columns_np[column_name] = _validate_numeric_array(
                array=array,
                label=f"Episode {episode_index}: {column_name}",
                report=report,
            )

        state_array = columns_np.get("observation.state")
        action_array = columns_np.get("action")
        timestamp_array = np.asarray(columns_np.get("timestamp"))
        frame_index_array = np.asarray(columns_np.get("frame_index"))
        episode_index_array = np.asarray(columns_np.get("episode_index"))
        global_index_array = np.asarray(columns_np.get("index"))
        task_index_array = np.asarray(columns_np.get("task_index"))

        state_ranges.update(state_array)
        action_ranges.update(action_array)

        if gripper_range is not None:
            gripper_range.update(state_array[:, gripper_index])
        if joint_range is not None:
            joint_range.update(state_array[:, joint_indices])

        if not np.array_equal(frame_index_array, np.arange(row_count, dtype=frame_index_array.dtype)):
            report.add_error(f"Episode {episode_index}: frame_index is not contiguous from 0.")

        expected_indices = np.arange(
            expected_global_index,
            expected_global_index + row_count,
            dtype=global_index_array.dtype,
        )
        if not np.array_equal(global_index_array, expected_indices):
            report.add_error(
                f"Episode {episode_index}: global index is not contiguous from {expected_global_index}."
            )
        expected_global_index += row_count

        if not np.all(episode_index_array == episode_index):
            report.add_error(f"Episode {episode_index}: episode_index column contains invalid values.")

        if np.any(np.diff(timestamp_array) <= 0):
            report.add_error(f"Episode {episode_index}: timestamps are not strictly increasing.")
        timestamp_reference = frame_index_array.astype(np.float64) / max(fps, 1)
        max_deviation = float(np.max(np.abs(timestamp_array.astype(np.float64) - timestamp_reference)))
        max_timestamp_deviation = max(max_timestamp_deviation, max_deviation)
        if max_deviation > (1.0 / max(fps, 1)):
            report.add_warning(
                f"Episode {episode_index}: timestamp deviation from frame_index/fps is {max_deviation:.6f}s."
            )

        task_indices = {int(value) for value in task_index_array.tolist()}
        unknown_task_indices = sorted(task_indices - set(tasks_by_index))
        if unknown_task_indices:
            report.add_error(
                f"Episode {episode_index}: unknown task_index values {unknown_task_indices}."
            )

        for task_name in episode_meta.get("tasks", []):
            if str(task_name) not in task_names:
                report.add_error(
                    f"Episode {episode_index}: task {task_name!r} not present in meta/tasks.jsonl."
                )

        if (
            state_array.ndim == 2
            and action_array.ndim == 2
            and state_array.shape[1] == action_array.shape[1]
            and state_names == action_names
            and row_count > 1
        ):
            if not np.allclose(action_array[:-1], state_array[1:], atol=1e-5):
                mismatch_positions = np.where(
                    ~np.isclose(action_array[:-1], state_array[1:], atol=1e-5)
                )
                first_row = int(mismatch_positions[0][0])
                first_col = int(mismatch_positions[1][0])
                report.add_error(
                    "Episode "
                    f"{episode_index}: action-to-next-state continuity failed at row={first_row}, "
                    f"dim={action_names[first_col]!r}."
                )

        episode_stats = episode_stats_by_index.get(episode_index)
        if episode_stats is None:
            report.add_error(f"Episode {episode_index}: missing entry in episodes_stats.jsonl.")
        else:
            for feature_name in ("observation.state", "action", "timestamp"):
                feature_stats = episode_stats.get(feature_name)
                if not isinstance(feature_stats, dict):
                    report.add_error(
                        f"Episode {episode_index}: missing stats for {feature_name!r}."
                    )
                    continue
                count_values = feature_stats.get("count")
                if not isinstance(count_values, list) or not count_values:
                    report.add_error(
                        f"Episode {episode_index}: invalid count for {feature_name!r} stats."
                    )
                    continue
                if int(count_values[0]) != row_count:
                    report.add_error(
                        f"Episode {episode_index}: stats count for {feature_name!r} "
                        f"is {count_values[0]}, expected {row_count}."
                    )

        for video_key in video_keys:
            if not video_path_pattern:
                report.add_error("meta/info.json: missing video_path for video features.")
                break
            video_path = root / str(video_path_pattern).format(
                episode_chunk=chunk_index,
                video_key=video_key,
                episode_index=episode_index,
            )
            if not video_path.exists():
                report.add_error(f"Episode {episode_index}: missing video file {video_path}.")
            elif video_path.stat().st_size <= 0:
                report.add_error(f"Episode {episode_index}: empty video file {video_path}.")

    if total_rows != total_frames:
        report.add_error(
            f"Total frame count mismatch: info.total_frames={total_frames}, parquet_rows={total_rows}."
        )

    if joint_range is None:
        report.add_note(
            "No joint position dimensions were found in observation.state names. "
            "This dataset appears to store tcp pose + gripper rather than robot joint angles."
        )
    else:
        report.ranges["joint_position"] = joint_range.to_named_dict()

    if gripper_range is None:
        report.add_warning("No gripper dimension was found in observation.state names.")
    else:
        report.ranges["gripper_position"] = gripper_range.to_named_dict()
        gripper_stats = next(iter(gripper_range.to_named_dict().values()))
        if gripper_stats["min"] == gripper_stats["max"]:
            report.add_warning(
                f"Gripper value is constant across the dataset: {gripper_stats['min']}."
            )
        if gripper_stats["min"] < 0.0 or gripper_stats["max"] > 1.0:
            report.add_warning(
                "Gripper values fall outside the common [0, 1] range. "
                "Confirm the expected encoding before training."
            )

    report.ranges["observation.state"] = state_ranges.to_named_dict()
    report.ranges["action"] = action_ranges.to_named_dict()
    report.summary = {
        "episodes": total_episodes,
        "frames": total_frames,
        "fps": fps,
        "parquet_files": len(parquet_files),
        "video_keys": video_keys,
        "max_timestamp_deviation_s": round(max_timestamp_deviation, 8),
    }
    return report


def collect_hdf5_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.hdf5")) + sorted(path.glob("*.h5"))
    return files


def _hdf5_dataset_exists(handle, dataset_path: str) -> bool:
    return dataset_path in handle


def validate_xval_hdf5(path: Path) -> ValidationReport:
    require_numpy()
    report = ValidationReport(dataset_format="xval_hdf5", root=path)
    if not require_h5py(report):
        return report

    hdf5_files = collect_hdf5_files(path)
    if not hdf5_files:
        report.add_error(f"No HDF5 files were found under {path}.")
        return report

    joint_names = [f"joint_{idx}" for idx in range(1, 7)] + ["gripper"]
    ee_names = ["x", "y", "z", "euler_x", "euler_y", "euler_z"]
    joint_ranges = VectorRangeAccumulator(joint_names)
    gripper_range = VectorRangeAccumulator(["gripper"])
    end_effector_ranges = VectorRangeAccumulator(ee_names)

    reference_cameras: dict[str, tuple[int, int, int]] | None = None
    total_frames = 0

    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as handle:
            required_datasets = (
                "puppet/end_effector",
                "puppet/joint_position",
            )
            for dataset_path in required_datasets:
                if not _hdf5_dataset_exists(handle, dataset_path):
                    report.add_error(f"{hdf5_path}: missing dataset {dataset_path}.")
            if report.errors and any(message.startswith(str(hdf5_path)) for message in report.errors):
                continue

            if "observations" not in handle or "images" not in handle["observations"]:
                report.add_error(f"{hdf5_path}: missing observations/images group.")
                continue

            images_group = handle["observations/images"]
            camera_names = sorted(images_group.keys())
            if not camera_names:
                report.add_error(f"{hdf5_path}: no camera datasets found under observations/images.")
                continue

            end_effector = np.asarray(handle["puppet/end_effector"])
            joint_position = np.asarray(handle["puppet/joint_position"])
            _validate_numeric_array(
                array=end_effector,
                label=f"{hdf5_path}: puppet/end_effector",
                report=report,
            )
            _validate_numeric_array(
                array=joint_position,
                label=f"{hdf5_path}: puppet/joint_position",
                report=report,
            )

            if end_effector.ndim != 2 or end_effector.shape[1] != 6:
                report.add_error(
                    f"{hdf5_path}: puppet/end_effector must have shape (T, 6), got {end_effector.shape}."
                )
                continue
            if joint_position.ndim != 2 or joint_position.shape[1] != 7:
                report.add_error(
                    f"{hdf5_path}: puppet/joint_position must have shape (T, 7), got {joint_position.shape}."
                )
                continue

            frame_count = int(end_effector.shape[0])
            total_frames += frame_count
            if frame_count <= 0:
                report.add_error(f"{hdf5_path}: empty trajectory.")
                continue
            if joint_position.shape[0] != frame_count:
                report.add_error(
                    f"{hdf5_path}: frame count mismatch between end_effector "
                    f"({frame_count}) and joint_position ({joint_position.shape[0]})."
                )

            camera_shapes: dict[str, tuple[int, int, int]] = {}
            for camera_name in camera_names:
                dataset = images_group[camera_name]
                camera_array = np.asarray(dataset)
                if camera_array.ndim != 4 or camera_array.shape[0] != frame_count or camera_array.shape[-1] != 3:
                    report.add_error(
                        f"{hdf5_path}: camera {camera_name!r} must have shape (T, H, W, 3), "
                        f"got {camera_array.shape}."
                    )
                    continue
                if not np.issubdtype(camera_array.dtype, np.integer):
                    report.add_warning(
                        f"{hdf5_path}: camera {camera_name!r} dtype is {camera_array.dtype}, expected uint8-like integers."
                    )
                if camera_array.min() < 0 or camera_array.max() > 255:
                    report.add_warning(
                        f"{hdf5_path}: camera {camera_name!r} values fall outside the common [0, 255] image range."
                    )
                camera_shapes[camera_name] = (
                    int(camera_array.shape[1]),
                    int(camera_array.shape[2]),
                    int(camera_array.shape[3]),
                )

            if reference_cameras is None:
                reference_cameras = camera_shapes
            elif camera_shapes != reference_cameras:
                report.add_error(
                    f"{hdf5_path}: camera schema mismatch. expected={reference_cameras}, actual={camera_shapes}."
                )

            language_instruction = handle.attrs.get("language_instruction")
            if language_instruction is None:
                report.add_warning(f"{hdf5_path}: missing language_instruction attribute.")
            elif not str(language_instruction).strip():
                report.add_warning(f"{hdf5_path}: empty language_instruction attribute.")

            joint_ranges.update(joint_position)
            gripper_range.update(joint_position[:, -1])
            end_effector_ranges.update(end_effector)

            if np.allclose(joint_position[:, :6], 0.0, atol=1e-8):
                report.add_warning(
                    f"{hdf5_path}: first 6 joint_position dimensions are all zeros. "
                    "This is acceptable for the current converter, but it means joint angles were not recorded."
                )
            if np.allclose(joint_position[:, -1], joint_position[0, -1], atol=1e-8):
                report.add_warning(
                    f"{hdf5_path}: gripper is constant ({joint_position[0, -1]}). "
                    "Confirm that the gripper signal was captured correctly."
                )

    report.ranges["joint_position"] = joint_ranges.to_named_dict()
    report.ranges["gripper_position"] = gripper_range.to_named_dict()
    report.ranges["end_effector"] = end_effector_ranges.to_named_dict()
    report.summary = {
        "files": len(hdf5_files),
        "frames": total_frames,
        "camera_schema": reference_cameras or {},
    }

    gripper_stats = next(iter(gripper_range.to_named_dict().values()), None)
    if gripper_stats is not None and (gripper_stats["min"] < 0.0 or gripper_stats["max"] > 1.0):
        report.add_warning(
            "Gripper values fall outside the common [0, 1] range. "
            "Confirm the expected encoding before training."
        )
    return report


def main() -> None:
    args = parse_args()
    target_path = args.path.expanduser().resolve()
    if not target_path.exists():
        raise SystemExit(f"Path does not exist: {target_path}")

    dataset_format = args.format
    if dataset_format == "auto":
        dataset_format = detect_format(target_path)

    if dataset_format == "lerobot_v21":
        report = validate_lerobot_v21(target_path)
    elif dataset_format == "raw_demos_npz":
        report = validate_raw_demos_npz(target_path)
    elif dataset_format == "xval_hdf5":
        report = validate_xval_hdf5(target_path)
    else:  # pragma: no cover - argparse prevents this
        raise SystemExit(f"Unsupported format: {dataset_format}")

    _print_report(report)
    if args.json_out is not None:
        _write_json_report(report, args.json_out.expanduser().resolve())

    if not report.passed:
        raise SystemExit(1)
    if args.strict and report.warnings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
