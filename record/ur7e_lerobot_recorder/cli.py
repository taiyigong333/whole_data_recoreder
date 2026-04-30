from __future__ import annotations

from .config import load_camera_specs, parse_args, validate_args
from .convert import run_conversion
from .recorder import run_recording, scan_available_cameras
from .runtime import require_runtime_dependencies


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.command == "convert":
        require_runtime_dependencies(need_rtde=False, need_ffmpeg=True)
        run_conversion(args)
        return

    if args.scan_cameras:
        require_runtime_dependencies(need_rtde=False, need_ffmpeg=False)
        scan_available_cameras(
            scan_camera_type=args.scan_camera_type,
            max_camera_index=args.max_camera_index,
        )
        return

    camera_specs = load_camera_specs(args)
    require_runtime_dependencies(camera_specs=camera_specs)
    run_recording(args, camera_specs)
