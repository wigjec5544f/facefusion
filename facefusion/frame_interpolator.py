"""Frame interpolator (RIFE) — single-pair inference helper.

This module provides the inference primitive needed by the upcoming
`frame_interpolator` workflow stage (Đợt 1.A3 in `ULTRA_ROADMAP.md`). It only
exposes pair-wise interpolation (img0, img1, timestep -> middle frame); the
video-level loop, batching, and pipeline integration ship in a follow-up PR
once this primitive is verified end-to-end.

Weights are hosted at https://huggingface.co/ngoqquyen/facefusion-extras
(see `ULTRA_ROADMAP.md` and `frame_interpolator/README.md` in that repo).
The HF org defaults to `ngoqquyen`; override with `FACEFUSION_HF_NAMESPACE`
(see `facefusion/download.py`) to point at a different mirror that mirrors
the same `facefusion-extras/frame_interpolator/...` layout.
"""
import os
from functools import lru_cache
from typing import Optional

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
