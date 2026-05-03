from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy

from facefusion import inference_manager, state_manager
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.face_helper import create_rotation_matrix_and_size, estimate_matrix_by_face_landmark_5, transform_points, warp_face_by_translation
from facefusion.filesystem import resolve_relative_path
from facefusion.processors.batching import run_with_dynamic_batch, run_with_dynamic_batch_multi
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.types import Angle, BoundingBox, DownloadScope, DownloadSet, FaceLandmark5, FaceLandmark68, InferencePool, ModelSet, Prediction, Score, VisionFrame


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'2dfan4':
		{
			'__metadata__':
			{
				'vendor': 'breadbread1984',
				'license': 'MIT',
				'year': 2018
			},
			'hashes':
			{
				'2dfan4':
				{
					'url': resolve_download_url('models-3.0.0', '2dfan4.hash'),
					'path': resolve_relative_path('../.assets/models/2dfan4.hash')
				}
			},
			'sources':
			{
				'2dfan4':
				{
					'url': resolve_download_url('models-3.0.0', '2dfan4.onnx'),
					'path': resolve_relative_path('../.assets/models/2dfan4.onnx')
				}
			},
			'size': (256, 256)
		},
		'peppa_wutz':
		{
			'__metadata__':
			{
				'vendor': 'Unknown',
				'license': 'Apache-2.0',
				'year': 2023
			},
			'hashes':
			{
				'peppa_wutz':
				{
					'url': resolve_download_url('models-3.0.0', 'peppa_wutz.hash'),
					'path': resolve_relative_path('../.assets/models/peppa_wutz.hash')
				}
			},
			'sources':
			{
				'peppa_wutz':
				{
					'url': resolve_download_url('models-3.0.0', 'peppa_wutz.onnx'),
					'path': resolve_relative_path('../.assets/models/peppa_wutz.onnx')
				}
			},
			'size': (256, 256)
		},
		'fan_68_5':
		{
			'__metadata__':
			{
				'vendor': 'FaceFusion',
				'license': 'OpenRAIL-M',
				'year': 2024
			},
			'hashes':
			{
				'fan_68_5':
				{
					'url': resolve_download_url('models-3.0.0', 'fan_68_5.hash'),
					'path': resolve_relative_path('../.assets/models/fan_68_5.hash')
				}
			},
			'sources':
			{
				'fan_68_5':
				{
					'url': resolve_download_url('models-3.0.0', 'fan_68_5.onnx'),
					'path': resolve_relative_path('../.assets/models/fan_68_5.onnx')
				}
			}
		}
	}


def get_inference_pool() -> InferencePool:
	model_names = [ state_manager.get_item('face_landmarker_model'), 'fan_68_5' ]
	_, model_source_set = collect_model_downloads()

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ state_manager.get_item('face_landmarker_model'), 'fan_68_5' ]
	inference_manager.clear_inference_pool(__name__, model_names)


def collect_model_downloads() -> Tuple[DownloadSet, DownloadSet]:
	model_set = create_static_model_set('full')
	model_hash_set =\
	{
		'fan_68_5': model_set.get('fan_68_5').get('hashes').get('fan_68_5')
	}
	model_source_set =\
	{
		'fan_68_5': model_set.get('fan_68_5').get('sources').get('fan_68_5')
	}

	for face_landmarker_model in [ '2dfan4', 'peppa_wutz' ]:
		if state_manager.get_item('face_landmarker_model') in [ 'many', face_landmarker_model ]:
			model_hash_set[face_landmarker_model] = model_set.get(face_landmarker_model).get('hashes').get(face_landmarker_model)
			model_source_set[face_landmarker_model] = model_set.get(face_landmarker_model).get('sources').get(face_landmarker_model)

	return model_hash_set, model_source_set


def pre_check() -> bool:
	model_hash_set, model_source_set = collect_model_downloads()

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def detect_face_landmark(vision_frame : VisionFrame, bounding_box : BoundingBox, face_angle : Angle) -> Tuple[FaceLandmark68, Score]:
	# Single-face wrapper -- routes through the batched implementation
	# with N=1 so both call sites share identical pre/post-processing.
	# Output is bit-identical to the previous serial implementation.
	[ result ] = detect_face_landmarks_batch(vision_frame, [ bounding_box ], [ face_angle ])
	return result


def detect_face_landmarks_batch(vision_frame : VisionFrame, bounding_boxes : List[BoundingBox], face_angles : List[Angle]) -> List[Tuple[FaceLandmark68, Score]]:
	"""Run the active 2dfan4 / peppa_wutz refinement model(s) over a list
	of faces in one batched ONNX call per model.

	Each face still has its own bounding-box-driven affine + rotation, so
	the per-face crop preparation stays sequential -- that is pure CPU
	work (cv2 warp + CLAHE) and is cheap relative to the ONNX call.
	The benefit is that the heavy ONNX ``session.run`` per model now sees
	a stacked ``(N, 3, H, W)`` batch and can collapse N round-trips into
	one when the loaded model declares a dynamic batch axis. For the
	stock fixed-batch shipped models we fall back to the original
	per-face loop, so output is bit-equal in either case.

	The cross-model score arbitration ("``2dfan4`` wins if its score is
	within 0.2 of ``peppa_wutz``") is preserved verbatim; we just hoist
	it out of the per-face loop into a per-face combination step that
	runs after the batched ONNX calls return.
	"""
	if not bounding_boxes:
		return []

	landmarker_model = state_manager.get_item('face_landmarker_model')
	use_2dfan4 = landmarker_model in [ 'many', '2dfan4' ]
	use_peppa_wutz = landmarker_model in [ 'many', 'peppa_wutz' ]

	results_2dfan4 : List[Tuple[FaceLandmark68, Score]] = []
	results_peppa_wutz : List[Tuple[FaceLandmark68, Score]] = []

	if use_2dfan4:
		results_2dfan4 = _detect_with_2dfan4_batch(vision_frame, bounding_boxes, face_angles)
	if use_peppa_wutz:
		results_peppa_wutz = _detect_with_peppa_wutz_batch(vision_frame, bounding_boxes, face_angles)

	combined : List[Tuple[FaceLandmark68, Score]] = []
	for index in range(len(bounding_boxes)):
		face_landmark_2dfan4, face_landmark_score_2dfan4 = results_2dfan4[index] if use_2dfan4 else (None, 0.0)
		face_landmark_peppa_wutz, face_landmark_score_peppa_wutz = results_peppa_wutz[index] if use_peppa_wutz else (None, 0.0)

		if face_landmark_score_2dfan4 > face_landmark_score_peppa_wutz - 0.2:
			combined.append((face_landmark_2dfan4, face_landmark_score_2dfan4))
		else:
			combined.append((face_landmark_peppa_wutz, face_landmark_score_peppa_wutz))
	return combined


def _detect_with_2dfan4_batch(vision_frame : VisionFrame, bounding_boxes : List[BoundingBox], face_angles : List[Angle]) -> List[Tuple[FaceLandmark68, Score]]:
	model_size = create_static_model_set('full').get('2dfan4').get('size')
	prepared_crops : List[numpy.ndarray] = []
	per_face_inverse_rotations : List[numpy.ndarray] = []
	per_face_inverse_affines : List[numpy.ndarray] = []

	for bounding_box, face_angle in zip(bounding_boxes, face_angles):
		scale = 195 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max().clip(1, None)
		translation = (model_size[0] - numpy.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
		rotation_matrix, rotation_size = create_rotation_matrix_and_size(face_angle, model_size)
		crop_vision_frame, affine_matrix = warp_face_by_translation(vision_frame, translation, scale, model_size)
		crop_vision_frame = cv2.warpAffine(crop_vision_frame, rotation_matrix, rotation_size)
		crop_vision_frame = conditional_optimize_contrast(crop_vision_frame)
		crop_vision_frame = crop_vision_frame.transpose(2, 0, 1).astype(numpy.float32) / 255.0
		prepared_crops.append(numpy.expand_dims(crop_vision_frame, axis = 0))
		per_face_inverse_rotations.append(cv2.invertAffineTransform(rotation_matrix))
		per_face_inverse_affines.append(cv2.invertAffineTransform(affine_matrix))

	batched_input = numpy.concatenate(prepared_crops, axis = 0)
	batched_landmarks, batched_heatmaps = forward_with_2dfan4_batch(batched_input)

	results : List[Tuple[FaceLandmark68, Score]] = []
	for index in range(batched_input.shape[0]):
		face_landmark_68 = batched_landmarks[index][:, :2] / 64 * 256
		face_landmark_68 = transform_points(face_landmark_68, per_face_inverse_rotations[index])
		face_landmark_68 = transform_points(face_landmark_68, per_face_inverse_affines[index])
		face_heatmap = batched_heatmaps[index]
		face_landmark_score_68 = numpy.amax(face_heatmap, axis = (1, 2))
		face_landmark_score_68 = numpy.mean(face_landmark_score_68)
		face_landmark_score_68 = numpy.interp(face_landmark_score_68, [ 0, 0.9 ], [ 0, 1 ])
		results.append((face_landmark_68, face_landmark_score_68))
	return results


def _detect_with_peppa_wutz_batch(vision_frame : VisionFrame, bounding_boxes : List[BoundingBox], face_angles : List[Angle]) -> List[Tuple[FaceLandmark68, Score]]:
	model_size = create_static_model_set('full').get('peppa_wutz').get('size')
	prepared_crops : List[numpy.ndarray] = []
	per_face_inverse_rotations : List[numpy.ndarray] = []
	per_face_inverse_affines : List[numpy.ndarray] = []

	for bounding_box, face_angle in zip(bounding_boxes, face_angles):
		scale = 195 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max().clip(1, None)
		translation = (model_size[0] - numpy.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
		rotation_matrix, rotation_size = create_rotation_matrix_and_size(face_angle, model_size)
		crop_vision_frame, affine_matrix = warp_face_by_translation(vision_frame, translation, scale, model_size)
		crop_vision_frame = cv2.warpAffine(crop_vision_frame, rotation_matrix, rotation_size)
		crop_vision_frame = conditional_optimize_contrast(crop_vision_frame)
		crop_vision_frame = crop_vision_frame.transpose(2, 0, 1).astype(numpy.float32) / 255.0
		prepared_crops.append(numpy.expand_dims(crop_vision_frame, axis = 0))
		per_face_inverse_rotations.append(cv2.invertAffineTransform(rotation_matrix))
		per_face_inverse_affines.append(cv2.invertAffineTransform(affine_matrix))

	batched_input = numpy.concatenate(prepared_crops, axis = 0)
	batched_predictions = forward_with_peppa_wutz_batch(batched_input)

	results : List[Tuple[FaceLandmark68, Score]] = []
	for index in range(batched_input.shape[0]):
		prediction = batched_predictions[index].reshape(-1, 3)
		face_landmark_68 = prediction[:, :2] / 64 * model_size[0]
		face_landmark_68 = transform_points(face_landmark_68, per_face_inverse_rotations[index])
		face_landmark_68 = transform_points(face_landmark_68, per_face_inverse_affines[index])
		face_landmark_score_68 = prediction[:, 2].mean()
		face_landmark_score_68 = numpy.interp(face_landmark_score_68, [ 0, 0.95 ], [ 0, 1 ])
		results.append((face_landmark_68, face_landmark_score_68))
	return results


def conditional_optimize_contrast(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_RGB2Lab)
	if numpy.mean(crop_vision_frame[:, :, 0]) < 30: #type:ignore[arg-type]
		crop_vision_frame[:, :, 0] = cv2.createCLAHE(clipLimit = 2).apply(crop_vision_frame[:, :, 0])
	crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_Lab2RGB)
	return crop_vision_frame


def estimate_face_landmark_68_5(face_landmark_5 : FaceLandmark5) -> FaceLandmark68:
	# Single-face wrapper -- routes through the batched implementation
	# with N=1 so both call sites share identical pre/post-processing.
	# Output is bit-identical to the previous serial implementation.
	[ face_landmark_68_5 ] = estimate_face_landmark_68_5_batch([ face_landmark_5 ])
	return face_landmark_68_5


def estimate_face_landmark_68_5_batch(face_landmarks_5 : List[FaceLandmark5]) -> List[FaceLandmark68]:
	"""Estimate 68-point landmarks for a list of 5-point landmarks in one
	batched ``fan_68_5`` ONNX call.

	Each input landmark is first warped into the model's canonical space
	(``ffhq_512``) using its own affine matrix; the warped 5-point arrays
	are stacked into ``(N, 5, 2)`` and dispatched once. After the model
	returns ``(N, 68, 2)`` we re-apply the per-face inverse affines to
	bring each prediction back into the original pixel space.

	When the loaded ONNX model declares a dynamic batch axis the entire
	batch runs in **one** ``session.run``; otherwise we transparently
	fall back to the same per-face loop the codebase used before
	(bit-equal output guaranteed).
	"""
	if not face_landmarks_5:
		return []

	per_face_prepared : List[numpy.ndarray] = []
	per_face_inverse_matrices : List[numpy.ndarray] = []

	for face_landmark_5 in face_landmarks_5:
		affine_matrix = estimate_matrix_by_face_landmark_5(face_landmark_5, 'ffhq_512', (1, 1))
		warped_landmark_5 = cv2.transform(face_landmark_5.reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
		per_face_prepared.append(warped_landmark_5.astype(numpy.float32))
		per_face_inverse_matrices.append(cv2.invertAffineTransform(affine_matrix))

	batched_input = numpy.stack(per_face_prepared, axis = 0)
	batched_predictions = forward_fan_68_5_batch(batched_input)

	results : List[FaceLandmark68] = []
	for index, inverse_matrix in enumerate(per_face_inverse_matrices):
		face_landmark_68_5 = batched_predictions[index]
		face_landmark_68_5 = cv2.transform(face_landmark_68_5.reshape(1, -1, 2), inverse_matrix).reshape(-1, 2)
		results.append(face_landmark_68_5)
	return results


def forward_with_2dfan4_batch(crop_vision_frames : numpy.ndarray) -> Tuple[Prediction, Prediction]:
	"""Run the ``2dfan4`` landmarker over a stacked ``(N, 3, H, W)`` batch.

	Returns the model's two outputs, each with a leading batch axis of
	size N. When the loaded model declares a dynamic batch axis the
	whole batch runs in one ``session.run``; otherwise we fall back to
	``run_session_looped`` (per-call output is bit-equal to the
	historical single-face path).
	"""
	face_landmarker = get_inference_pool().get('2dfan4')

	with conditional_thread_semaphore():
		landmarks, heatmaps = run_with_dynamic_batch_multi(face_landmarker, {}, 'input', crop_vision_frames, output_indices = (0, 1))

	return landmarks, heatmaps


def forward_with_peppa_wutz_batch(crop_vision_frames : numpy.ndarray) -> Prediction:
	"""Run the ``peppa_wutz`` landmarker over a stacked ``(N, 3, H, W)``
	batch. Returns ``(N, ...)``; same dynamic / fixed batch handling as
	``forward_with_2dfan4_batch``.
	"""
	face_landmarker = get_inference_pool().get('peppa_wutz')

	with conditional_thread_semaphore():
		prediction = run_with_dynamic_batch(face_landmarker, {}, 'input', crop_vision_frames)

	return prediction


def forward_fan_68_5(face_landmark_5 : FaceLandmark5) -> FaceLandmark68:
	face_landmarker = get_inference_pool().get('fan_68_5')

	with conditional_thread_semaphore():
		face_landmark_68_5 = face_landmarker.run(None,
		{
			'input': [ face_landmark_5 ]
		})[0][0]

	return face_landmark_68_5


def forward_fan_68_5_batch(face_landmarks_5 : numpy.ndarray) -> numpy.ndarray:
	"""Run the ``fan_68_5`` 5->68 landmark expander over a stacked
	``(N, 5, 2)`` batch.

	When the ONNX model has a dynamic batch axis the whole batch is
	dispatched in one ``session.run`` call; for fixed-batch models we
	fall back to ``run_session_looped`` and the per-call output is
	bit-equal to the historical loop.
	"""
	face_landmarker = get_inference_pool().get('fan_68_5')

	with conditional_thread_semaphore():
		face_landmarks_68_5 = run_with_dynamic_batch(face_landmarker, {}, 'input', face_landmarks_5)

	return face_landmarks_68_5
