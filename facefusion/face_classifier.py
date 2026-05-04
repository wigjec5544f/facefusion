from functools import lru_cache
from typing import List, Tuple

import numpy

from facefusion import inference_manager
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.face_helper import warp_face_by_face_landmark_5
from facefusion.filesystem import resolve_relative_path
from facefusion.processors.batching import run_with_dynamic_batch_multi
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.types import Age, DownloadScope, FaceLandmark5, Gender, InferencePool, ModelOptions, ModelSet, Race, VisionFrame


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'fairface':
		{
			'__metadata__':
			{
				'vendor': 'dchen236',
				'license': 'CC-BY-4.0',
				'year': 2021
			},
			'hashes':
			{
				'face_classifier':
				{
					'url': resolve_download_url('models-3.0.0', 'fairface.hash'),
					'path': resolve_relative_path('../.assets/models/fairface.hash')
				}
			},
			'sources':
			{
				'face_classifier':
				{
					'url': resolve_download_url('models-3.0.0', 'fairface.onnx'),
					'path': resolve_relative_path('../.assets/models/fairface.onnx')
				}
			},
			'template': 'arcface_112_v2',
			'size': (224, 224),
			'mean': [ 0.485, 0.456, 0.406 ],
			'standard_deviation': [ 0.229, 0.224, 0.225 ]
		}
	}


def get_inference_pool() -> InferencePool:
	model_names = [ 'fairface' ]
	model_source_set = get_model_options().get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ 'fairface' ]
	inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options() -> ModelOptions:
	return create_static_model_set('full').get('fairface')


def pre_check() -> bool:
	model_hash_set = get_model_options().get('hashes')
	model_source_set = get_model_options().get('sources')

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def classify_face(temp_vision_frame : VisionFrame, face_landmark_5 : FaceLandmark5) -> Tuple[Gender, Age, Race]:
	# Single-face wrapper -- routes through the batched implementation
	# with N=1 so both call sites share identical pre/post-processing.
	# Output is bit-identical to the previous serial implementation.
	[ result ] = classify_faces(temp_vision_frame, [ face_landmark_5 ])
	return result


def classify_faces(temp_vision_frame : VisionFrame, face_landmarks_5 : List[FaceLandmark5]) -> List[Tuple[Gender, Age, Race]]:
	"""Classify gender / age / race for a list of faces in
	``temp_vision_frame`` using a single batched ``fairface`` ONNX call.

	Each face is warped to the classifier's template and stacked into a
	``(N, 3, H, W)`` batch. When the underlying ONNX model declares a
	dynamic batch axis the entire batch runs in **one** ``session.run``;
	otherwise we transparently fall back to the per-face loop the
	codebase used before -- the per-face output is bit-equal in either
	case.

	Returns a list aligned with *face_landmarks_5*.
	"""
	if not face_landmarks_5:
		return []

	model_template = get_model_options().get('template')
	model_size = get_model_options().get('size')
	model_mean = get_model_options().get('mean')
	model_standard_deviation = get_model_options().get('standard_deviation')
	prepared_crops : List[numpy.ndarray] = []

	for face_landmark_5 in face_landmarks_5:
		crop_vision_frame, _ = warp_face_by_face_landmark_5(temp_vision_frame, face_landmark_5, model_template, model_size)
		crop_vision_frame = crop_vision_frame.astype(numpy.float32)[:, :, ::-1] / 255.0
		crop_vision_frame -= model_mean
		crop_vision_frame /= model_standard_deviation
		crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
		prepared_crops.append(numpy.expand_dims(crop_vision_frame, axis = 0))

	batched_input = numpy.concatenate(prepared_crops, axis = 0)
	gender_ids, age_ids, race_ids = forward_batch(batched_input)

	results : List[Tuple[Gender, Age, Race]] = []
	for index in range(batched_input.shape[0]):
		gender = categorize_gender(gender_ids[index])
		age = categorize_age(age_ids[index])
		race = categorize_race(race_ids[index])
		results.append((gender, age, race))
	return results


def forward(crop_vision_frame : VisionFrame) -> Tuple[List[int], List[int], List[int]]:
	# Single-face forward kept for back-compat. Delegates to the batched
	# entry point with N=1; bit-equal to the previous serial path.
	gender_ids, age_ids, race_ids = forward_batch(crop_vision_frame)
	return gender_ids, age_ids, race_ids


def forward_batch(crop_vision_frames : numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
	"""Run the ``fairface`` classifier over a stacked ``(N, 3, H, W)``
	batch.

	Returns a triple ``(gender_ids, age_ids, race_ids)``, each with a
	leading batch axis of size N. When the loaded ONNX model has a
	dynamic batch axis the entire batch runs in one ``session.run``;
	otherwise we fall back to ``run_session_looped_multi`` (bit-equal to
	the historical per-face path).

	NB: the upstream ``fairface`` model's output order is
	``(race, gender, age)``; we shuffle to ``(gender, age, race)`` to
	preserve the public API of ``forward`` and downstream call sites.
	"""
	face_classifier = get_inference_pool().get('face_classifier')

	with conditional_thread_semaphore():
		race_ids, gender_ids, age_ids = run_with_dynamic_batch_multi(face_classifier, {}, 'input', crop_vision_frames, output_indices = (0, 1, 2))

	return gender_ids, age_ids, race_ids


def categorize_gender(gender_id : int) -> Gender:
	if gender_id == 1:
		return 'female'
	return 'male'


def categorize_age(age_id : int) -> Age:
	if age_id == 0:
		return range(0, 2)
	if age_id == 1:
		return range(3, 9)
	if age_id == 2:
		return range(10, 19)
	if age_id == 3:
		return range(20, 29)
	if age_id == 4:
		return range(30, 39)
	if age_id == 5:
		return range(40, 49)
	if age_id == 6:
		return range(50, 59)
	if age_id == 7:
		return range(60, 69)
	return range(70, 100)


def categorize_race(race_id : int) -> Race:
	if race_id == 1:
		return 'black'
	if race_id == 2:
		return 'latino'
	if race_id == 3 or race_id == 4:
		return 'asian'
	if race_id == 5:
		return 'indian'
	if race_id == 6:
		return 'arabic'
	return 'white'
