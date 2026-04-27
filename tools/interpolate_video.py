#!/usr/bin/env python3
"""Standalone video frame interpolator (RIFE) — increases the FPS of a video by
inserting AI-generated intermediate frames between every consecutive pair.

This is the user-facing CLI for `facefusion.frame_interpolator` (Đợt 1.A3).
Pipeline integration into the main facefusion process flow ships in a separate
PR once we have GPU validation; in the meantime this tool can be chained after
`python facefusion.py headless-run …` to upgrade output to 60 fps / 90 fps /
120 fps.

Usage:

	# Double the framerate (30 -> 60)
	python tools/interpolate_video.py --input swap.mp4 --output swap_60fps.mp4 --multiplier 2

	# Triple to 90 fps
	python tools/interpolate_video.py --input swap.mp4 --output swap_90fps.mp4 --multiplier 3

The output video has `multiplier × source_fps` frames per second; the original
frames are kept verbatim and `(multiplier - 1)` AI-generated intermediates are
inserted between every consecutive pair of source frames.

Requires ffmpeg on PATH. Reads the source video frame-by-frame via an ffmpeg
pipe (rgb24 raw stream) — no temp disk extraction.
"""
import argparse
import json
import pathlib
import subprocess
import sys
from typing import Iterator, List, Optional, Tuple

import numpy

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from facefusion import frame_interpolator, state_manager  # noqa: E402


def probe_video(input_path : pathlib.Path) -> Tuple[int, int, float]:
	"""Return (width, height, fps) for the input video using ffprobe."""
	cmd =\
	[
		'ffprobe', '-v', 'error',
		'-select_streams', 'v:0',
		'-show_entries', 'stream=width,height,r_frame_rate',
		'-of', 'json',
		str(input_path)
	]
	result = subprocess.run(cmd, capture_output = True, text = True, check = True)
	stream = json.loads(result.stdout)['streams'][0]
	width = int(stream['width'])
	height = int(stream['height'])
	num, den = stream['r_frame_rate'].split('/')
	fps = float(num) / float(den) if float(den) != 0 else float(num)
	return width, height, fps


def iter_source_frames(input_path : pathlib.Path, width : int, height : int) -> Iterator[numpy.ndarray]:
	"""Yield consecutive HxWx3 BGR uint8 frames from the input video."""
	cmd =\
	[
		'ffmpeg',
		'-hide_banner', '-loglevel', 'error',
		'-i', str(input_path),
		'-f', 'rawvideo', '-pix_fmt', 'bgr24',
		'pipe:1'
	]
	frame_size = width * height * 3
	process = subprocess.Popen(cmd, stdout = subprocess.PIPE)
	try:
		assert process.stdout is not None
		while True:
			buffer = process.stdout.read(frame_size)
			if len(buffer) < frame_size:
				break
			yield numpy.frombuffer(buffer, dtype = numpy.uint8).reshape(height, width, 3)
	finally:
		if process.stdout is not None:
			process.stdout.close()
		process.wait()


def open_writer(output_path : pathlib.Path, width : int, height : int, target_fps : float, codec : str, crf : int) -> subprocess.Popen:
	cmd =\
	[
		'ffmpeg',
		'-hide_banner', '-loglevel', 'error', '-y',
		'-f', 'rawvideo', '-vcodec', 'rawvideo',
		'-s', '{0}x{1}'.format(width, height),
		'-pix_fmt', 'bgr24',
		'-r', str(target_fps),
		'-i', 'pipe:0',
		'-c:v', codec,
		'-crf', str(crf),
		'-pix_fmt', 'yuv420p',
		str(output_path)
	]
	return subprocess.Popen(cmd, stdin = subprocess.PIPE)


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


def interpolate_frames(prev_frame : numpy.ndarray, next_frame : numpy.ndarray, multiplier : int) -> List[numpy.ndarray]:
	"""Return [interp_1, interp_2, ..., interp_(multiplier - 1)] between prev and next."""
	intermediates : List[numpy.ndarray] = []
	for step in range(1, multiplier):
		timestep = step / multiplier
		intermediates.append(frame_interpolator.interpolate_pair(prev_frame, next_frame, timestep = timestep))
	return intermediates


def run(input_path : pathlib.Path, output_path : pathlib.Path, multiplier : int, codec : str, crf : int, execution_providers : List[str]) -> int:
	if multiplier < 2:
		print('--multiplier must be >= 2', file = sys.stderr)
		return 2
	if not input_path.is_file():
		print('input not found:', input_path, file = sys.stderr)
		return 2

	width, height, source_fps = probe_video(input_path)
	target_fps = source_fps * multiplier
	print(f'source: {input_path.name} {width}x{height} @ {source_fps:.3f} fps -> target {target_fps:.3f} fps (multiplier {multiplier})')

	init_inference_state(execution_providers)
	if not frame_interpolator.pre_check():
		print('frame_interpolator.pre_check() failed (model download / hash check)', file = sys.stderr)
		return 3

	writer = open_writer(output_path, width, height, target_fps, codec, crf)
	assert writer.stdin is not None

	frame_count = 0
	pair_count = 0
	prev_frame : Optional[numpy.ndarray] = None
	try:
		for frame in iter_source_frames(input_path, width, height):
			if prev_frame is not None:
				writer.stdin.write(prev_frame.tobytes())
				frame_count += 1
				for intermediate in interpolate_frames(prev_frame, frame, multiplier):
					writer.stdin.write(intermediate.tobytes())
					frame_count += 1
				pair_count += 1
				if pair_count % 25 == 0:
					print(f'  processed {pair_count} pairs ({frame_count} output frames)')
			prev_frame = frame
		if prev_frame is not None:
			writer.stdin.write(prev_frame.tobytes())
			frame_count += 1
	finally:
		if writer.stdin is not None:
			writer.stdin.close()
		writer.wait()

	print(f'wrote {frame_count} frames -> {output_path}')
	return 0


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
