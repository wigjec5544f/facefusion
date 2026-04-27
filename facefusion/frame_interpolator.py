"""Frame interpolator (RIFE) — pair-wise + video-level helpers.

The pair-wise primitive (`interpolate_pair`) shipped in Đợt 1.A3 part 1
(PR #6); part 3 adds `interpolate_video_file` so other code paths
(notably the headless workflow in `facefusion.workflows.image_to_video`
and the standalone `tools/interpolate_video.py` CLI) can reuse the
same logic without duplicating ffmpeg orchestration.

Weights are hosted at https://huggingface.co/ngoqquyen/facefusion-extras
(see `ULTRA_ROADMAP.md` and `frame_interpolator/README.md` in that repo).
The HF org defaults to `ngoqquyen`; override with `FACEFUSION_HF_NAMESPACE`
(see `facefusion/download.py`) to point at a different mirror that mirrors
the same `facefusion-extras/frame_interpolator/...` layout.
"""
import json
import os
import subprocess
from functools import lru_cache
from typing import Iterator, List, Optional, Tuple

import numpy

from facefusion import inference_manager
from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.filesystem import resolve_relative_path
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.types import DownloadScope, InferencePool, ModelOptions, ModelSet, VisionFrame

EXTRAS_DEFAULT_NAMESPACE = 'ngoqquyen'
EXTRAS_DEFAULT_REPO = 'facefusion-extras'


def resolve_extras_url(file_name : str) -> str:
	namespace = os.environ.get('FACEFUSION_HF_NAMESPACE', EXTRAS_DEFAULT_NAMESPACE).strip().strip('/') or EXTRAS_DEFAULT_NAMESPACE
	repo = os.environ.get('FACEFUSION_EXTRAS_REPO', EXTRAS_DEFAULT_REPO).strip().strip('/') or EXTRAS_DEFAULT_REPO
	return 'https://huggingface.co/' + namespace + '/' + repo + '/resolve/main/' + file_name


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'rife_4_9':
		{
			'__metadata__':
			{
				'vendor': 'hzwer/Practical-RIFE',
				'license': 'MIT',
				'year': 2024
			},
			'hashes':
			{
				'frame_interpolator':
				{
					'url': resolve_extras_url('frame_interpolator/rife_4_9.hash'),
					'path': resolve_relative_path('../.assets/models/rife_4_9.hash')
				}
			},
			'sources':
			{
				'frame_interpolator':
				{
					'url': resolve_extras_url('frame_interpolator/rife_4_9.onnx'),
					'path': resolve_relative_path('../.assets/models/rife_4_9.onnx')
				}
			}
		}
	}


def get_inference_pool(model_name : Optional[str] = None) -> InferencePool:
	model_names = [ model_name or 'rife_4_9' ]
	model_source_set = get_model_options(model_names[0]).get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool(model_name : Optional[str] = None) -> None:
	model_names = [ model_name or 'rife_4_9' ]
	inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options(model_name : Optional[str] = None) -> ModelOptions:
	return create_static_model_set('full').get(model_name or 'rife_4_9')


def pre_check(model_name : Optional[str] = None) -> bool:
	model_hash_set = get_model_options(model_name).get('hashes')
	model_source_set = get_model_options(model_name).get('sources')

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def interpolate_pair(prev_frame : VisionFrame, next_frame : VisionFrame, timestep : float = 0.5, model_name : Optional[str] = None) -> VisionFrame:
	"""Interpolate one frame between `prev_frame` and `next_frame` at fractional position `timestep` (0..1).

	Both inputs are HxWx3 BGR uint8 (`VisionFrame`). Returns a HxWx3 BGR uint8 frame at the same resolution.
	"""
	prev_tensor = _frame_to_tensor(prev_frame)
	next_tensor = _frame_to_tensor(next_frame)
	timestep_tensor = numpy.array([ float(timestep) ], dtype = numpy.float32)
	output_tensor = forward(prev_tensor, next_tensor, timestep_tensor, model_name)
	return _tensor_to_frame(output_tensor)


def forward(prev_tensor : numpy.ndarray, next_tensor : numpy.ndarray, timestep_tensor : numpy.ndarray, model_name : Optional[str] = None) -> numpy.ndarray:
	frame_interpolator = get_inference_pool(model_name).get('frame_interpolator')

	with conditional_thread_semaphore():
		output_tensor = frame_interpolator.run(None,
		{
			'img0': prev_tensor,
			'img1': next_tensor,
			'timestep': timestep_tensor
		})[0]
	return output_tensor


def _frame_to_tensor(frame : VisionFrame) -> numpy.ndarray:
	# BGR uint8 HxWx3 -> RGB float32 [0..1] 1x3xHxW
	rgb = frame[:, :, ::-1].astype(numpy.float32) / 255.0
	return numpy.expand_dims(rgb.transpose(2, 0, 1), axis = 0)


def _tensor_to_frame(tensor : numpy.ndarray) -> VisionFrame:
	# 1x3xHxW float32 [0..1] -> BGR uint8 HxWx3
	chw = tensor[0]
	rgb = chw.transpose(1, 2, 0)
	rgb = numpy.clip(rgb * 255.0, 0, 255).astype(numpy.uint8)
	return rgb[:, :, ::-1]


# --- Video-level helpers (shared by tools/interpolate_video.py and the
# headless workflow). They wrap ffmpeg / ffprobe so the rest of the project
# only depends on a single function: `interpolate_video_file`. ---


def _probe_video(input_path : str) -> Tuple[int, int, float]:
	"""Return (width, height, fps) for the input video using ffprobe."""
	cmd =\
	[
		'ffprobe', '-v', 'error',
		'-select_streams', 'v:0',
		'-show_entries', 'stream=width,height,r_frame_rate',
		'-of', 'json',
		input_path
	]
	result = subprocess.run(cmd, capture_output = True, text = True, check = True)
	stream = json.loads(result.stdout)['streams'][0]
	width = int(stream['width'])
	height = int(stream['height'])
	num, den = stream['r_frame_rate'].split('/')
	fps = float(num) / float(den) if float(den) != 0 else float(num)
	return width, height, fps


def _iter_source_frames(input_path : str, width : int, height : int) -> Iterator[numpy.ndarray]:
	"""Yield consecutive HxWx3 BGR uint8 frames from the input video."""
	cmd =\
	[
		'ffmpeg',
		'-hide_banner', '-loglevel', 'error',
		'-i', input_path,
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


def _open_writer(output_path : str, width : int, height : int, target_fps : float, codec : str, crf : int) -> 'subprocess.Popen[bytes]':
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
		output_path
	]
	return subprocess.Popen(cmd, stdin = subprocess.PIPE)


def _safe_write(process : 'subprocess.Popen[bytes]', payload : bytes) -> bool:
	"""Write payload to process.stdin; return False if the encoder pipe is broken."""
	if process.stdin is None:
		return False
	try:
		process.stdin.write(payload)
		return True
	except (BrokenPipeError, OSError):
		return False


def _interpolate_intermediates(prev_frame : numpy.ndarray, next_frame : numpy.ndarray, multiplier : int, model_name : Optional[str] = None) -> List[numpy.ndarray]:
	"""Return [interp_1, ..., interp_(multiplier - 1)] between prev and next."""
	intermediates : List[numpy.ndarray] = []
	for step in range(1, multiplier):
		timestep = step / multiplier
		intermediates.append(interpolate_pair(prev_frame, next_frame, timestep = timestep, model_name = model_name))
	return intermediates


def interpolate_video_file(
	input_path : str,
	output_path : str,
	multiplier : int,
	codec : str = 'libx264',
	crf : int = 18,
	model_name : Optional[str] = None
) -> int:
	"""Run RIFE interpolation on a video, writing to `output_path`.

	Returns 0 on success, non-zero on failure (mirrors CLI exit codes):
	* 2 — invalid args (multiplier < 2, missing input)
	* 3 — model download / hash check failed
	* 4 — ffmpeg encoder exited non-zero
	"""
	if multiplier < 2:
		return 2
	if not os.path.isfile(input_path):
		return 2
	if not pre_check(model_name):
		return 3

	width, height, source_fps = _probe_video(input_path)
	target_fps = source_fps * multiplier

	writer = _open_writer(output_path, width, height, target_fps, codec, crf)
	encoder_died = False
	prev_frame : Optional[numpy.ndarray] = None
	try:
		for frame in _iter_source_frames(input_path, width, height):
			if prev_frame is not None:
				if not _safe_write(writer, prev_frame.tobytes()):
					encoder_died = True
					break
				for intermediate in _interpolate_intermediates(prev_frame, frame, multiplier, model_name):
					if not _safe_write(writer, intermediate.tobytes()):
						encoder_died = True
						break
				if encoder_died:
					break
			prev_frame = frame
		if not encoder_died and prev_frame is not None:
			_safe_write(writer, prev_frame.tobytes())
	finally:
		if writer.stdin is not None:
			try:
				writer.stdin.close()
			except (BrokenPipeError, OSError):
				pass
		writer.wait()

	if writer.returncode != 0:
		return 4
	return 0
