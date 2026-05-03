from functools import lru_cache
from typing import List, Tuple

import numpy

from facefusion import inference_manager
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.face_helper import warp_face_by_face_landmark_5
from facefusion.filesystem import resolve_relative_path
from facefusion.processors.batching import run_with_dynamic_batch
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.types import DownloadScope, Embedding, FaceLandmark5, InferencePool, ModelOptions, ModelSet, VisionFrame


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'arcface':
		{
			'__metadata__':
			{
				'vendor': 'InsightFace',
				'license': 'Non-Commercial',
				'year': 2018
			},
			'hashes':
			{
				'face_recognizer':
				{
					'url': resolve_download_url('models-3.0.0', 'arcface_w600k_r50.hash'),
					'path': resolve_relative_path('../.assets/models/arcface_w600k_r50.hash')
				}
			},
			'sources':
			{
				'face_recognizer':
				{
					'url': resolve_download_url('models-3.0.0', 'arcface_w600k_r50.onnx'),
					'path': resolve_relative_path('../.assets/models/arcface_w600k_r50.onnx')
				}
			},
			'template': 'arcface_112_v2',
			'size': (112, 112)
		}
	}


def get_inference_pool() -> InferencePool:
	model_names = [ 'arcface' ]
	model_source_set = get_model_options().get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ 'arcface' ]
	inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options() -> ModelOptions:
	return create_static_model_set('full').get('arcface')


def pre_check() -> bool:
	model_hash_set = get_model_options().get('hashes')
	model_source_set = get_model_options().get('sources')

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def calculate_face_embedding(temp_vision_frame : VisionFrame, face_landmark_5 : FaceLandmark5) -> Tuple[Embedding, Embedding]:
	# Single-face convenience wrapper -- delegates to the batch path with
	# N=1 so both call sites share the exact same warp / normalise / ONNX
	# code. Output is bit-identical to the previous serial implementation.
	embeddings = calculate_face_embeddings(temp_vision_frame, [ face_landmark_5 ])
	return embeddings[0]


def calculate_face_embeddings(temp_vision_frame : VisionFrame, face_landmarks_5 : List[FaceLandmark5]) -> List[Tuple[Embedding, Embedding]]:
	"""Compute ArcFace embeddings for a list of faces in *temp_vision_frame*.

	Each face is warped to the recogniser's template and stacked into a
	single (N, 3, H, W) batch. If the underlying ONNX model declares a
	dynamic batch axis, all N faces run in **one** session call; otherwise
	we transparently fall back to the same per-face loop the codebase used
	before -- the output is bit-equal in either case.

	Returns a list aligned with *face_landmarks_5*; each entry is the
	``(embedding, embedding_norm)`` tuple expected by ``Face``.
	"""
	if not face_landmarks_5:
		return []

	model_template = get_model_options().get('template')
	model_size = get_model_options().get('size')
	prepared_crops : List[numpy.ndarray] = []

	for face_landmark_5 in face_landmarks_5:
		crop_vision_frame, _ = warp_face_by_face_landmark_5(temp_vision_frame, face_landmark_5, model_template, model_size)
		crop_vision_frame = crop_vision_frame / 127.5 - 1
		crop_vision_frame = crop_vision_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)
		prepared_crops.append(numpy.expand_dims(crop_vision_frame, axis = 0))

	batched_input = numpy.concatenate(prepared_crops, axis = 0)
	batched_embeddings = forward_batch(batched_input)

	results : List[Tuple[Embedding, Embedding]] = []
	for index in range(batched_embeddings.shape[0]):
		face_embedding = batched_embeddings[index].ravel()
		face_embedding_norm = face_embedding / numpy.linalg.norm(face_embedding)
		results.append((face_embedding, face_embedding_norm))
	return results


def forward(crop_vision_frame : VisionFrame) -> Embedding:
	face_recognizer = get_inference_pool().get('face_recognizer')

	with conditional_thread_semaphore():
		face_embedding = face_recognizer.run(None,
		{
			'input': crop_vision_frame
		})[0]

	return face_embedding


def forward_batch(crop_vision_frames : numpy.ndarray) -> numpy.ndarray:
	"""Run the ArcFace recogniser over a stacked ``(N, 3, H, W)`` batch.

	When the loaded ONNX model has a dynamic batch axis, the entire batch
	is dispatched in a single ``session.run`` call. Otherwise we fall back
	to ``run_session_looped`` which preserves the historical per-face
	behaviour byte-for-byte.
	"""
	face_recognizer = get_inference_pool().get('face_recognizer')

	with conditional_thread_semaphore():
		face_embeddings = run_with_dynamic_batch(face_recognizer, {}, 'input', crop_vision_frames)

	return face_embeddings
