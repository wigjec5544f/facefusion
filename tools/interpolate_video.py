#!/usr/bin/env python3
"""Standalone video frame interpolator (RIFE) — increases the FPS of a video by
inserting AI-generated intermediate frames between every consecutive pair.

This is the user-facing CLI for `facefusion.frame_interpolator`. The shared
ffmpeg orchestration lives in `facefusion.frame_interpolator.interpolate_video_file`
so the same logic backs both this CLI and the headless workflow's
`--frame-interpolator-target-fps` flag.

Usage:

	# Double the framerate (30 -> 60)
	python tools/interpolate_video.py --input swap.mp4 --output swap_60fps.mp4 --multiplier 2

	# Triple to 90 fps
	python tools/interpolate_video.py --input swap.mp4 --output swap_90fps.mp4 --multiplier 3

The output video has `multiplier × source_fps` frames per second; the original
frames are kept verbatim and `(multiplier - 1)` AI-generated intermediates are
inserted between every consecutive pair of source frames.

Requires ffmpeg on PATH.
"""
import argparse
import pathlib
import sys
from typing import List, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from facefusion import frame_interpolator, state_manager  # noqa: E402


def init_inference_state(execution_providers : List[str]) -> None:
	"""Initialize the minimal state_manager keys that inference_manager needs."""
	if state_manager.get_item('execution_device_ids') is None:
		state_manager.init_item('execution_device_ids', [ 0 ])
	if state_manager.get_item('execution_providers') is None:
		state_manager.init_item('execution_providers', execution_providers)
	if state_manager.get_item('execution_thread_count') is None:
		state_manager.init_item('execution_thread_count', 4)
	if state_manager.get_item('execution_queue_count') is None:
		state_manager.init_item('execution_queue_count', 1)


def run(input_path : pathlib.Path, output_path : pathlib.Path, multiplier : int, codec : str, crf : int, execution_providers : List[str]) -> int:
	if multiplier < 2:
		print('--multiplier must be >= 2', file = sys.stderr)
		return 2
	if not input_path.is_file():
		print('input not found:', input_path, file = sys.stderr)
		return 2

	init_inference_state(execution_providers)
	width, height, source_fps = frame_interpolator._probe_video(str(input_path))
	target_fps = source_fps * multiplier
	print(f'source: {input_path.name} {width}x{height} @ {source_fps:.3f} fps -> target {target_fps:.3f} fps (multiplier {multiplier})')

	rc = frame_interpolator.interpolate_video_file(
		input_path = str(input_path),
		output_path = str(output_path),
		multiplier = multiplier,
		codec = codec,
		crf = crf
	)
	if rc == 3:
		print('frame_interpolator.pre_check() failed (model download / hash check)', file = sys.stderr)
	elif rc == 4:
		print('ffmpeg encoder exited with non-zero status', file = sys.stderr)
	elif rc == 0:
		print(f'wrote -> {output_path}')
	return rc


def main(argv : Optional[list] = None) -> int:
	parser = argparse.ArgumentParser(description = 'increase video FPS by AI frame interpolation (RIFE 4.9)')
	parser.add_argument('--input', required = True, help = 'source video path')
	parser.add_argument('--output', required = True, help = 'destination video path (e.g. swap_60fps.mp4)')
	parser.add_argument('--multiplier', type = int, default = 2, help = 'fps multiplier (2 = double, 3 = triple, ...)')
	parser.add_argument('--codec', default = 'libx264', help = 'output video codec (default: libx264)')
	parser.add_argument('--crf', type = int, default = 18, help = 'output video CRF (lower = higher quality, default 18)')
	parser.add_argument('--execution-provider', action = 'append', default = None, help = 'onnxruntime execution provider, repeat to chain (default: cpu)')
	args = parser.parse_args(argv)

	execution_providers = args.execution_provider or [ 'cpu' ]
	return run(pathlib.Path(args.input).expanduser().resolve(), pathlib.Path(args.output).expanduser().resolve(), args.multiplier, args.codec, args.crf, execution_providers)


if __name__ == '__main__':
	sys.exit(main())
