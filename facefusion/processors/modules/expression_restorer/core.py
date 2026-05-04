from argparse import ArgumentParser
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy

import facefusion.jobs.job_manager
import facefusion.jobs.job_store
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, inference_manager, logger, state_manager, translator, video_manager
from facefusion.common_helper import create_int_metavar
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.face_analyser import scale_face
from facefusion.face_helper import paste_back, warp_face_by_face_landmark_5
from facefusion.face_masker import create_box_mask, create_occlusion_mask
from facefusion.face_selector import select_faces
from facefusion.filesystem import in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from facefusion.processors.batching import supports_dynamic_batch
from facefusion.processors.live_portrait import create_rotation, limit_expression
from facefusion.processors.modules.expression_restorer import choices as expression_restorer_choices
from facefusion.processors.modules.expression_restorer.types import ExpressionRestorerInputs
from facefusion.processors.types import LivePortraitExpression, LivePortraitFeatureVolume, LivePortraitMotionPoints, LivePortraitPitch, LivePortraitRoll, LivePortraitScale, LivePortraitTranslation, LivePortraitYaw, ProcessorOutputs
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import conditional_thread_semaphore, thread_semaphore
from facefusion.types import ApplyStateItem, Args, BoundingBox, DownloadScope, Face, InferencePool, ModelOptions, ModelSet, ProcessMode, VisionFrame
from facefusion.vision import read_static_image, read_static_video_frame


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'live_portrait':
		{
			'__metadata__':
			{
				'vendor': 'KwaiVGI',
				'license': 'MIT',
				'year': 2024
			},
			'hashes':
			{
				'feature_extractor':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_feature_extractor.hash'),
					'path': resolve_relative_path('../.assets/models/live_portrait_feature_extractor.hash')
				},
				'motion_extractor':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_motion_extractor.hash'),
					'path': resolve_relative_path('../.assets/models/live_portrait_motion_extractor.hash')
				},
				'generator':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_generator.hash'),
					'path': resolve_relative_path('../.assets/models/live_portrait_generator.hash')
				}
			},
			'sources':
			{
				'feature_extractor':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_feature_extractor.onnx'),
					'path': resolve_relative_path('../.assets/models/live_portrait_feature_extractor.onnx')
				},
				'motion_extractor':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_motion_extractor.onnx'),
					'path': resolve_relative_path('../.assets/models/live_portrait_motion_extractor.onnx')
				},
				'generator':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_generator.onnx'),
					'path': resolve_relative_path('../.assets/models/live_portrait_generator.onnx')
				}
			},
			'template': 'arcface_128',
			'size': (512, 512)
		}
	}


def get_inference_pool() -> InferencePool:
	model_names = [ state_manager.get_item('expression_restorer_model') ]
	model_source_set = get_model_options().get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ state_manager.get_item('expression_restorer_model') ]
	inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options() -> ModelOptions:
	model_name = state_manager.get_item('expression_restorer_model')
	return create_static_model_set('full').get(model_name)


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--expression-restorer-model', help = translator.get('help.model', __package__), default = config.get_str_value('processors', 'expression_restorer_model', 'live_portrait'), choices = expression_restorer_choices.expression_restorer_models)
		group_processors.add_argument('--expression-restorer-factor', help = translator.get('help.factor', __package__), type = int, default = config.get_int_value('processors', 'expression_restorer_factor', '80'), choices = expression_restorer_choices.expression_restorer_factor_range, metavar = create_int_metavar(expression_restorer_choices.expression_restorer_factor_range))
		group_processors.add_argument('--expression-restorer-areas', help = translator.get('help.areas', __package__).format(choices = ', '.join(expression_restorer_choices.expression_restorer_areas)), default = config.get_str_list('processors', 'expression_restorer_areas', ' '.join(expression_restorer_choices.expression_restorer_areas)), choices = expression_restorer_choices.expression_restorer_areas, nargs = '+', metavar = 'EXPRESSION_RESTORER_AREAS')
		facefusion.jobs.job_store.register_step_keys([ 'expression_restorer_model', 'expression_restorer_factor', 'expression_restorer_areas' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('expression_restorer_model', args.get('expression_restorer_model'))
	apply_state_item('expression_restorer_factor', args.get('expression_restorer_factor'))
	apply_state_item('expression_restorer_areas', args.get('expression_restorer_areas'))


def pre_check() -> bool:
	model_hash_set = get_model_options().get('hashes')
	model_source_set = get_model_options().get('sources')

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def pre_process(mode : ProcessMode) -> bool:
	if mode == 'stream':
		logger.error(translator.get('stream_not_supported') + translator.get('exclamation_mark'), __name__)
		return False
	if mode in [ 'output', 'preview' ] and not is_image(state_manager.get_item('target_path')) and not is_video(state_manager.get_item('target_path')):
		logger.error(translator.get('choose_image_or_video_target') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
		logger.error(translator.get('specify_image_or_video_output') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not same_file_extension(state_manager.get_item('target_path'), state_manager.get_item('output_path')):
		logger.error(translator.get('match_target_and_output_extension') + translator.get('exclamation_mark'), __name__)
		return False
	return True


def post_process() -> None:
	read_static_image.cache_clear()
	read_static_video_frame.cache_clear()
	video_manager.clear_video_pool()
	if state_manager.get_item('video_memory_strategy') in [ 'strict', 'moderate' ]:
		clear_inference_pool()
	if state_manager.get_item('video_memory_strategy') == 'strict':
		content_analyser.clear_inference_pool()
		face_classifier.clear_inference_pool()
		face_detector.clear_inference_pool()
		face_landmarker.clear_inference_pool()
		face_masker.clear_inference_pool()
		face_recognizer.clear_inference_pool()


def restore_expressions(target_faces : List[Face], target_vision_frame : VisionFrame, temp_vision_frame : VisionFrame) -> VisionFrame:
	"""Restore expression for every face in ``target_faces`` against the
	same source frame.

	When the target faces' bounding boxes do not overlap, all crops are
	pushed through the LivePortrait stack -- ``feature_extractor``,
	``motion_extractor`` and ``generator`` -- in **batched** session.run
	calls (one per ONNX model, regardless of face count) and pasted back
	sequentially. Output is bit-equal to the per-face loop because no
	face's paste-back affects another face's warp region.

	If any pair of bounding boxes intersect, the second face's warp would
	sample pixels modified by the first face's paste-back, so we
	transparently fall back to the original per-face loop -- still
	bit-equal.
	"""
	if not target_faces:
		return temp_vision_frame
	if len(target_faces) == 1 or _faces_overlap([ face.bounding_box for face in target_faces ]):
		for target_face in target_faces:
			temp_vision_frame = restore_expression(target_face, target_vision_frame, temp_vision_frame)
		return temp_vision_frame

	model_template = get_model_options().get('template')
	model_size = get_model_options().get('size')
	expression_restorer_factor = float(numpy.interp(float(state_manager.get_item('expression_restorer_factor')), [ 0, 100 ], [ 0, 1.2 ]))

	prepared_target_crops : List[VisionFrame] = []
	prepared_temp_crops : List[VisionFrame] = []
	contexts : List[Tuple[numpy.ndarray, List[numpy.ndarray]]] = []

	for target_face in target_faces:
		target_crop_vision_frame, _ = warp_face_by_face_landmark_5(target_vision_frame, target_face.landmark_set.get('5/68'), model_template, model_size)
		temp_crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark_set.get('5/68'), model_template, model_size)

		box_mask = create_box_mask(temp_crop_vision_frame, state_manager.get_item('face_mask_blur'), (0, 0, 0, 0))
		crop_masks = [ box_mask ]
		if 'occlusion' in state_manager.get_item('face_mask_types'):
			crop_masks.append(create_occlusion_mask(temp_crop_vision_frame))

		prepared_target_crops.append(prepare_crop_frame(target_crop_vision_frame))
		prepared_temp_crops.append(prepare_crop_frame(temp_crop_vision_frame))
		contexts.append((affine_matrix, crop_masks))

	target_crops_stacked = numpy.concatenate(prepared_target_crops, axis = 0)
	temp_crops_stacked = numpy.concatenate(prepared_temp_crops, axis = 0)

	feature_volumes = forward_extract_feature_batch(temp_crops_stacked)
	target_motion = forward_extract_motion_batch(target_crops_stacked)
	temp_motion = forward_extract_motion_batch(temp_crops_stacked)

	pitches, yaws, rolls, scales, translations, temp_expressions, motion_points_batch = temp_motion
	target_expressions = target_motion[5]

	target_motion_points_list : List[numpy.ndarray] = []
	temp_motion_points_list : List[numpy.ndarray] = []

	for index in range(len(target_faces)):
		pitch = pitches[index : index + 1]
		yaw = yaws[index : index + 1]
		roll = rolls[index : index + 1]
		scale = scales[index : index + 1]
		translation = translations[index : index + 1]
		temp_expression = temp_expressions[index : index + 1]
		motion_points = motion_points_batch[index : index + 1]
		target_expression = target_expressions[index : index + 1]

		rotation = create_rotation(pitch, yaw, roll)
		target_expression = restrict_expression_areas(temp_expression, target_expression)
		target_expression = target_expression * expression_restorer_factor + temp_expression * (1 - expression_restorer_factor)
		target_expression = limit_expression(target_expression)
		target_motion_points = scale * (motion_points @ rotation.T + target_expression) + translation
		temp_motion_points = scale * (motion_points @ rotation.T + temp_expression) + translation
		target_motion_points_list.append(target_motion_points)
		temp_motion_points_list.append(temp_motion_points)

	target_motion_points_stacked = numpy.concatenate(target_motion_points_list, axis = 0)
	temp_motion_points_stacked = numpy.concatenate(temp_motion_points_list, axis = 0)
	generated_batch = forward_generate_frame_batch(feature_volumes, target_motion_points_stacked, temp_motion_points_stacked)

	for index, (affine_matrix, crop_masks) in enumerate(contexts):
		crop_vision_frame = normalize_crop_frame(generated_batch[index])
		crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
		temp_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
	return temp_vision_frame


def _faces_overlap(bounding_boxes : List[BoundingBox]) -> bool:
	"""Return True if any pair of bounding boxes intersect, with a small
	conservative expansion to cover the wider warp region used by
	``warp_face_by_face_landmark_5``. If in doubt the caller should fall
	back to the sequential loop, which is always bit-equal."""
	expansion = 0.25
	expanded : List[Tuple[float, float, float, float]] = []
	for box in bounding_boxes:
		left, top, right, bottom = float(box[0]), float(box[1]), float(box[2]), float(box[3])
		width = right - left
		height = bottom - top
		dx = width * expansion
		dy = height * expansion
		expanded.append((left - dx, top - dy, right + dx, bottom + dy))

	for index in range(len(expanded)):
		left_i, top_i, right_i, bottom_i = expanded[index]
		for other_index in range(index + 1, len(expanded)):
			left_j, top_j, right_j, bottom_j = expanded[other_index]
			if left_i < right_j and left_j < right_i and top_i < bottom_j and top_j < bottom_i:
				return True
	return False


def restore_expression(target_face : Face, target_vision_frame : VisionFrame, temp_vision_frame : VisionFrame) -> VisionFrame:
	model_template = get_model_options().get('template')
	model_size = get_model_options().get('size')
	expression_restorer_factor = float(numpy.interp(float(state_manager.get_item('expression_restorer_factor')), [ 0, 100 ], [ 0, 1.2 ]))
	target_crop_vision_frame, _ = warp_face_by_face_landmark_5(target_vision_frame, target_face.landmark_set.get('5/68'), model_template, model_size)
	temp_crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark_set.get('5/68'), model_template, model_size)
	box_mask = create_box_mask(temp_crop_vision_frame, state_manager.get_item('face_mask_blur'), (0, 0, 0, 0))
	crop_masks =\
	[
		box_mask
	]

	if 'occlusion' in state_manager.get_item('face_mask_types'):
		occlusion_mask = create_occlusion_mask(temp_crop_vision_frame)
		crop_masks.append(occlusion_mask)

	target_crop_vision_frame = prepare_crop_frame(target_crop_vision_frame)
	temp_crop_vision_frame = prepare_crop_frame(temp_crop_vision_frame)
	temp_crop_vision_frame = apply_restore(target_crop_vision_frame, temp_crop_vision_frame, expression_restorer_factor)
	temp_crop_vision_frame = normalize_crop_frame(temp_crop_vision_frame)
	crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
	paste_vision_frame = paste_back(temp_vision_frame, temp_crop_vision_frame, crop_mask, affine_matrix)
	return paste_vision_frame


def apply_restore(target_crop_vision_frame : VisionFrame, temp_crop_vision_frame : VisionFrame, expression_restorer_factor : float) -> VisionFrame:
	feature_volume = forward_extract_feature(temp_crop_vision_frame)
	target_expression = forward_extract_motion(target_crop_vision_frame)[5]
	pitch, yaw, roll, scale, translation, temp_expression, motion_points = forward_extract_motion(temp_crop_vision_frame)
	rotation = create_rotation(pitch, yaw, roll)
	target_expression = restrict_expression_areas(temp_expression, target_expression)
	target_expression = target_expression * expression_restorer_factor + temp_expression * (1 - expression_restorer_factor)
	target_expression = limit_expression(target_expression)
	target_motion_points = scale * (motion_points @ rotation.T + target_expression) + translation
	temp_motion_points = scale * (motion_points @ rotation.T + temp_expression) + translation
	crop_vision_frame = forward_generate_frame(feature_volume, target_motion_points, temp_motion_points)
	return crop_vision_frame


def restrict_expression_areas(temp_expression : LivePortraitExpression, target_expression : LivePortraitExpression) -> LivePortraitExpression:
	expression_restorer_areas = state_manager.get_item('expression_restorer_areas')

	if 'upper-face' not in expression_restorer_areas:
		target_expression[:, [ 1, 2, 6, 10, 11, 12, 13, 15, 16 ]] = temp_expression[:, [ 1, 2, 6, 10, 11, 12, 13, 15, 16 ]]

	if 'lower-face' not in expression_restorer_areas:
		target_expression[:, [ 3, 7, 14, 17, 18, 19, 20 ]] = temp_expression[:, [ 3, 7, 14, 17, 18, 19, 20 ]]

	target_expression[:, [ 0, 4, 5, 8, 9 ]] = temp_expression[:, [ 0, 4, 5, 8, 9 ]]
	return target_expression


def forward_extract_feature(crop_vision_frame : VisionFrame) -> LivePortraitFeatureVolume:
	feature_extractor = get_inference_pool().get('feature_extractor')

	with conditional_thread_semaphore():
		feature_volume = feature_extractor.run(None,
		{
			'input': crop_vision_frame
		})[0]

	return feature_volume


def forward_extract_motion(crop_vision_frame : VisionFrame) -> Tuple[LivePortraitPitch, LivePortraitYaw, LivePortraitRoll, LivePortraitScale, LivePortraitTranslation, LivePortraitExpression, LivePortraitMotionPoints]:
	motion_extractor = get_inference_pool().get('motion_extractor')

	with conditional_thread_semaphore():
		pitch, yaw, roll, scale, translation, expression, motion_points = motion_extractor.run(None,
		{
			'input': crop_vision_frame
		})

	return pitch, yaw, roll, scale, translation, expression, motion_points


def forward_generate_frame(feature_volume : LivePortraitFeatureVolume, target_motion_points : LivePortraitMotionPoints, temp_motion_points : LivePortraitMotionPoints) -> VisionFrame:
	generator = get_inference_pool().get('generator')

	with thread_semaphore():
		crop_vision_frame = generator.run(None,
		{
			'feature_volume': feature_volume,
			'source': target_motion_points,
			'target': temp_motion_points
		})[0][0]

	return crop_vision_frame


def forward_extract_feature_batch(crop_vision_frames : numpy.ndarray) -> numpy.ndarray:
	"""Run ``feature_extractor`` once across a stacked ``(N, ...)`` batch
	of crops. With a dynamic-batch ONNX export this collapses the per-face
	loop into a single ``session.run`` call; otherwise the helper falls
	back to N sequential calls so the per-face slice is bit-equal."""
	feature_extractor = get_inference_pool().get('feature_extractor')

	with conditional_thread_semaphore():
		if supports_dynamic_batch(feature_extractor, 'input'):
			try:
				return feature_extractor.run(None, { 'input': crop_vision_frames })[0]
			except Exception:  # pragma: no cover - defensive fall-through
				pass
		per_face : List[numpy.ndarray] = []
		for index in range(crop_vision_frames.shape[0]):
			single = feature_extractor.run(None, { 'input': crop_vision_frames[index : index + 1] })[0]
			per_face.append(single)
	return numpy.concatenate(per_face, axis = 0)


def forward_extract_motion_batch(crop_vision_frames : numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
	"""Run ``motion_extractor`` once across a stacked ``(N, ...)`` batch
	of crops. Returns the same 7-tuple of arrays as
	``forward_extract_motion`` but with a leading ``N`` axis on each."""
	motion_extractor = get_inference_pool().get('motion_extractor')

	with conditional_thread_semaphore():
		if supports_dynamic_batch(motion_extractor, 'input'):
			try:
				outputs = motion_extractor.run(None, { 'input': crop_vision_frames })
				return tuple(outputs[index] for index in range(7))  # type: ignore[return-value]
			except Exception:  # pragma: no cover - defensive fall-through
				pass
		per_face_outputs : List[List[numpy.ndarray]] = [ [] for _ in range(7) ]
		for index in range(crop_vision_frames.shape[0]):
			outputs = motion_extractor.run(None, { 'input': crop_vision_frames[index : index + 1] })
			for slot in range(7):
				per_face_outputs[slot].append(outputs[slot])
	return tuple(numpy.concatenate(buffer, axis = 0) for buffer in per_face_outputs)  # type: ignore[return-value]


def forward_generate_frame_batch(feature_volumes : numpy.ndarray, target_motion_points : numpy.ndarray, temp_motion_points : numpy.ndarray) -> numpy.ndarray:
	"""Run ``generator`` once across a stacked ``(N, ...)`` batch.
	Each input has its own dynamic-batch axis; if any input is fixed
	batch=1 we fall back to N sequential calls. Returns a (N, C, H, W)
	array."""
	generator = get_inference_pool().get('generator')

	with thread_semaphore():
		if (
			supports_dynamic_batch(generator, 'feature_volume')
			and supports_dynamic_batch(generator, 'source')
			and supports_dynamic_batch(generator, 'target')
		):
			try:
				return generator.run(None,
				{
					'feature_volume': feature_volumes,
					'source': target_motion_points,
					'target': temp_motion_points
				})[0]
			except Exception:  # pragma: no cover - defensive fall-through
				pass
		per_face : List[numpy.ndarray] = []
		for index in range(feature_volumes.shape[0]):
			generated = generator.run(None,
			{
				'feature_volume': feature_volumes[index : index + 1],
				'source': target_motion_points[index : index + 1],
				'target': temp_motion_points[index : index + 1]
			})[0]
			per_face.append(generated)
	return numpy.concatenate(per_face, axis = 0)


def prepare_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	model_size = get_model_options().get('size')
	prepare_size = (model_size[0] // 2, model_size[1] // 2)
	crop_vision_frame = cv2.resize(crop_vision_frame, prepare_size, interpolation = cv2.INTER_AREA)
	crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
	crop_vision_frame = numpy.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
	return crop_vision_frame


def normalize_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = crop_vision_frame.transpose(1, 2, 0).clip(0, 1)
	crop_vision_frame = crop_vision_frame * 255.0
	crop_vision_frame = crop_vision_frame.astype(numpy.uint8)[:, :, ::-1]
	return crop_vision_frame


def process_frame(inputs : ExpressionRestorerInputs) -> ProcessorOutputs:
	reference_vision_frame = inputs.get('reference_vision_frame')
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	target_faces = select_faces(reference_vision_frame, target_vision_frame)

	if target_faces:
		scaled_faces = [ scale_face(target_face, target_vision_frame, temp_vision_frame) for target_face in target_faces ]
		temp_vision_frame = restore_expressions(scaled_faces, target_vision_frame, temp_vision_frame)

	return temp_vision_frame, temp_vision_mask
