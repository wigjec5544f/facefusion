"""Portrait animator processor (Đợt 4.D1).

Animates a static source portrait so it follows the head pose and facial
expression of the faces detected in the target image / video. The processor
reuses the LivePortrait ONNX components that already ship with facefusion
for ``expression_restorer`` and ``face_editor`` -- ``feature_extractor``
(appearance volume), ``motion_extractor`` (pitch/yaw/roll/scale/translation
+ canonical 21-keypoint expression) and ``generator`` (warps the appearance
volume to a new keypoint configuration).

Pipeline summary (per detected target face):

1.  Source portrait is cropped to ffhq_512, fed through the feature and
    motion extractors **once** (cached); we keep its appearance volume +
    pose + scale + translation + expression + canonical keypoints.
2.  Each driving frame's face is cropped to ffhq_512 and only its motion
    is extracted (pitch, yaw, roll, expression).
3.  The target's pose / expression are blended onto the source's
    appearance + scale + translation according to two weight knobs
    (``--portrait-animator-pose-weight``, ``--portrait-animator-expression-weight``).
4.  The generator warps the cached source feature volume from the source
    rest-pose keypoints to the new "driven" keypoints.
5.  The animated 512x512 crop is pasted back into the driving frame, so
    the output keeps the driving frame's background and the source
    person's face is reanimated in place of the target face.

This produces an output where:

*   identity / appearance / lighting on the face come from the source
    portrait,
*   pose & expression come from the driving target,
*   background and face position come from the driving target.

That makes it complementary to ``face_swapper`` (which only transfers an
identity embedding) and ``expression_restorer`` (which keeps the target
identity but modulates expression).
"""
from argparse import ArgumentParser
from functools import lru_cache
from typing import List, Optional, Tuple

import cv2
import numpy

import facefusion.jobs.job_manager
import facefusion.jobs.job_store
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, inference_manager, logger, state_manager, translator, video_manager
from facefusion.common_helper import create_int_metavar
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.face_analyser import get_many_faces, get_one_face, scale_face
from facefusion.face_helper import paste_back, warp_face_by_face_landmark_5
from facefusion.face_masker import create_box_mask, create_occlusion_mask
from facefusion.face_selector import select_faces, sort_faces_by_order
from facefusion.filesystem import filter_image_paths, has_image, in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from facefusion.processors.batching import supports_dynamic_batch
from facefusion.processors.live_portrait import create_rotation, limit_expression
from facefusion.processors.modules.portrait_animator import choices as portrait_animator_choices
from facefusion.processors.modules.portrait_animator.types import PortraitAnimatorInputs
from facefusion.processors.types import LivePortraitExpression, LivePortraitFeatureVolume, LivePortraitMotionPoints, LivePortraitPitch, LivePortraitRoll, LivePortraitScale, LivePortraitTranslation, LivePortraitYaw, ProcessorOutputs
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import conditional_thread_semaphore, thread_semaphore
from facefusion.types import ApplyStateItem, Args, BoundingBox, DownloadScope, Face, InferencePool, ModelOptions, ModelSet, ProcessMode, VisionFrame
from facefusion.vision import read_static_image, read_static_images, read_static_video_frame

# Cache of the per-source-portrait "rest state" (appearance feature
# volume + canonical motion). The generator + extractors are deterministic
# functions of the cropped 256x256 source input so a small LRU keyed by the
# source path tuple is safe across frames within a job. Cleared on
# `post_process` together with the inference pool when memory pressure
# requires it.
_SOURCE_STATE_LRU_SIZE = 4


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
				'year': 2024,
				'paper': 'https://arxiv.org/abs/2407.03168',
				'upstream': 'https://github.com/KwaiVGI/LivePortrait'
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
			'template': 'ffhq_512',
			'size': (512, 512)
		}
	}


def get_inference_pool() -> InferencePool:
	model_names = [ state_manager.get_item('portrait_animator_model') ]
	model_source_set = get_model_options().get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ state_manager.get_item('portrait_animator_model') ]
	inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options() -> ModelOptions:
	model_name = state_manager.get_item('portrait_animator_model')
	return create_static_model_set('full').get(model_name)


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument(
			'--portrait-animator-model',
			help = translator.get('help.model', __package__),
			default = config.get_str_value('processors', 'portrait_animator_model', 'live_portrait'),
			choices = portrait_animator_choices.portrait_animator_models
		)
		group_processors.add_argument(
			'--portrait-animator-pose-weight',
			help = translator.get('help.pose_weight', __package__),
			type = int,
			default = config.get_int_value('processors', 'portrait_animator_pose_weight', '100'),
			choices = portrait_animator_choices.portrait_animator_pose_weight_range,
			metavar = create_int_metavar(portrait_animator_choices.portrait_animator_pose_weight_range)
		)
		group_processors.add_argument(
			'--portrait-animator-expression-weight',
			help = translator.get('help.expression_weight', __package__),
			type = int,
			default = config.get_int_value('processors', 'portrait_animator_expression_weight', '100'),
			choices = portrait_animator_choices.portrait_animator_expression_weight_range,
			metavar = create_int_metavar(portrait_animator_choices.portrait_animator_expression_weight_range)
		)
		facefusion.jobs.job_store.register_step_keys(
		[
			'portrait_animator_model',
			'portrait_animator_pose_weight',
			'portrait_animator_expression_weight'
		])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('portrait_animator_model', args.get('portrait_animator_model'))
	apply_state_item('portrait_animator_pose_weight', args.get('portrait_animator_pose_weight'))
	apply_state_item('portrait_animator_expression_weight', args.get('portrait_animator_expression_weight'))


def pre_check() -> bool:
	model_hash_set = get_model_options().get('hashes')
	model_source_set = get_model_options().get('sources')

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def pre_process(mode : ProcessMode) -> bool:
	if mode == 'stream':
		logger.error(translator.get('stream_not_supported') + translator.get('exclamation_mark'), __name__)
		return False
	if not has_image(state_manager.get_item('source_paths')):
		logger.error(translator.get('choose_image_source') + translator.get('exclamation_mark'), __name__)
		return False

	source_image_paths = filter_image_paths(state_manager.get_item('source_paths'))
	source_vision_frames = read_static_images(source_image_paths)
	source_faces = get_many_faces(source_vision_frames)

	if not get_one_face(source_faces):
		logger.error(translator.get('no_source_face_detected') + translator.get('exclamation_mark'), __name__)
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
	clear_source_state_cache()
	if state_manager.get_item('video_memory_strategy') in [ 'strict', 'moderate' ]:
		clear_inference_pool()
	if state_manager.get_item('video_memory_strategy') == 'strict':
		content_analyser.clear_inference_pool()
		face_classifier.clear_inference_pool()
		face_detector.clear_inference_pool()
		face_landmarker.clear_inference_pool()
		face_masker.clear_inference_pool()
		face_recognizer.clear_inference_pool()


def animate_portrait(target_face : Face, target_vision_frame : VisionFrame, temp_vision_frame : VisionFrame) -> VisionFrame:
	source_state = _resolve_source_state()

	if source_state is None:
		# pre_process should have rejected this, but never silently emit
		# garbage if a caller bypasses validation.
		return temp_vision_frame

	model_template = get_model_options().get('template')
	model_size = get_model_options().get('size')
	# Driving motion (pose / expression) MUST come from the original
	# target frame; using `temp_vision_frame` would feed the motion
	# extractor whatever the previous processor in the chain produced
	# (e.g. a face_swapper output), which is no longer the real driver.
	# `temp_vision_frame` is only used for the paste-back affine matrix
	# and the masks, so the animated face lands in the right place on
	# top of the chain's accumulated background. This mirrors the split
	# already used by `expression_restorer.restore_expression`.
	target_crop_vision_frame, _ = warp_face_by_face_landmark_5(target_vision_frame, target_face.landmark_set.get('5/68'), model_template, model_size)
	temp_crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark_set.get('5/68'), model_template, model_size)
	box_mask = create_box_mask(temp_crop_vision_frame, state_manager.get_item('face_mask_blur'), (0, 0, 0, 0))
	crop_masks = [ box_mask ]
	if 'occlusion' in (state_manager.get_item('face_mask_types') or []):
		crop_masks.append(create_occlusion_mask(temp_crop_vision_frame))

	target_input = prepare_crop_frame(target_crop_vision_frame)
	tgt_pitch, tgt_yaw, tgt_roll, _, _, tgt_expression, _ = forward_extract_motion(target_input)

	pose_weight = float(numpy.interp(float(state_manager.get_item('portrait_animator_pose_weight') or 0), [ 0, 100 ], [ 0.0, 1.0 ]))
	expression_weight = float(numpy.interp(float(state_manager.get_item('portrait_animator_expression_weight') or 0), [ 0, 100 ], [ 0.0, 1.0 ]))

	driven_pitch = source_state['pitch'] + (tgt_pitch - source_state['pitch']) * pose_weight
	driven_yaw = source_state['yaw'] + (tgt_yaw - source_state['yaw']) * pose_weight
	driven_roll = source_state['roll'] + (tgt_roll - source_state['roll']) * pose_weight
	driven_expression = source_state['expression'] + (tgt_expression - source_state['expression']) * expression_weight
	driven_expression = limit_expression(driven_expression)

	driven_rotation = create_rotation(driven_pitch, driven_yaw, driven_roll)
	driven_motion_points = source_state['scale'] * (source_state['motion_points'] @ driven_rotation.T + driven_expression) + source_state['translation']

	animated_crop = forward_generate_frame(source_state['feature_volume'], driven_motion_points, source_state['rest_motion_points'])
	animated_crop = normalize_crop_frame(animated_crop)
	animated_crop = cv2.resize(animated_crop, (temp_crop_vision_frame.shape[1], temp_crop_vision_frame.shape[0]), interpolation = cv2.INTER_CUBIC)

	crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
	return paste_back(temp_vision_frame, animated_crop, crop_mask, affine_matrix)


def animate_portraits(target_faces : List[Face], target_vision_frame : VisionFrame, temp_vision_frame : VisionFrame) -> VisionFrame:
	"""Animate every face in ``target_faces`` against the same source portrait.

	When the target faces' bounding boxes do not overlap, all crops are
	pushed through ``motion_extractor`` and ``generator`` in **batched**
	session.run calls (one per ONNX model, regardless of face count).
	The cached source ``feature_volume`` is broadcast across the batch
	via ``numpy.repeat``. Output is bit-equal to the per-face loop because
	no face's paste-back affects another face's warp region.

	If any pair of bounding boxes intersect (with a 25% conservative
	margin to cover the wider warp region) the second face's warp would
	sample pixels modified by the first face's paste-back, so we
	transparently fall back to the original per-face loop -- still
	bit-equal.
	"""
	if not target_faces:
		return temp_vision_frame
	if len(target_faces) == 1 or _faces_overlap([ face.bounding_box for face in target_faces ]):
		for target_face in target_faces:
			temp_vision_frame = animate_portrait(target_face, target_vision_frame, temp_vision_frame)
		return temp_vision_frame

	source_state = _resolve_source_state()
	if source_state is None:
		# pre_process should have rejected this, but never silently emit
		# garbage if a caller bypasses validation.
		return temp_vision_frame

	model_template = get_model_options().get('template')
	model_size = get_model_options().get('size')
	pose_weight = float(numpy.interp(float(state_manager.get_item('portrait_animator_pose_weight') or 0), [ 0, 100 ], [ 0.0, 1.0 ]))
	expression_weight = float(numpy.interp(float(state_manager.get_item('portrait_animator_expression_weight') or 0), [ 0, 100 ], [ 0.0, 1.0 ]))

	prepared_target_crops : List[VisionFrame] = []
	contexts : List[Tuple[numpy.ndarray, List[numpy.ndarray], Tuple[int, int]]] = []

	for target_face in target_faces:
		# Driving motion (pose / expression) MUST come from the original
		# target frame; mirrors the split used by ``animate_portrait``.
		target_crop_vision_frame, _ = warp_face_by_face_landmark_5(target_vision_frame, target_face.landmark_set.get('5/68'), model_template, model_size)
		temp_crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark_set.get('5/68'), model_template, model_size)

		box_mask = create_box_mask(temp_crop_vision_frame, state_manager.get_item('face_mask_blur'), (0, 0, 0, 0))
		crop_masks = [ box_mask ]
		if 'occlusion' in (state_manager.get_item('face_mask_types') or []):
			crop_masks.append(create_occlusion_mask(temp_crop_vision_frame))

		prepared_target_crops.append(prepare_crop_frame(target_crop_vision_frame))
		contexts.append((affine_matrix, crop_masks, (temp_crop_vision_frame.shape[1], temp_crop_vision_frame.shape[0])))

	target_crops_stacked = numpy.concatenate(prepared_target_crops, axis = 0)
	tgt_pitches, tgt_yaws, tgt_rolls, _, _, tgt_expressions, _ = forward_extract_motion_batch(target_crops_stacked)

	target_motion_points_list : List[numpy.ndarray] = []

	for index in range(len(target_faces)):
		tgt_pitch = tgt_pitches[index : index + 1]
		tgt_yaw = tgt_yaws[index : index + 1]
		tgt_roll = tgt_rolls[index : index + 1]
		tgt_expression = tgt_expressions[index : index + 1]

		driven_pitch = source_state['pitch'] + (tgt_pitch - source_state['pitch']) * pose_weight
		driven_yaw = source_state['yaw'] + (tgt_yaw - source_state['yaw']) * pose_weight
		driven_roll = source_state['roll'] + (tgt_roll - source_state['roll']) * pose_weight
		driven_expression = source_state['expression'] + (tgt_expression - source_state['expression']) * expression_weight
		driven_expression = limit_expression(driven_expression)

		driven_rotation = create_rotation(driven_pitch, driven_yaw, driven_roll)
		driven_motion_points = source_state['scale'] * (source_state['motion_points'] @ driven_rotation.T + driven_expression) + source_state['translation']
		target_motion_points_list.append(driven_motion_points)

	target_motion_points_stacked = numpy.concatenate(target_motion_points_list, axis = 0)
	# ``rest_motion_points`` is shared across all faces (depends only on
	# the cached source state); broadcast it to (N, ...) so the generator
	# receives matching batch axes for every input.
	rest_motion_points_stacked = numpy.repeat(source_state['rest_motion_points'], len(target_faces), axis = 0)
	feature_volumes_stacked = numpy.repeat(source_state['feature_volume'], len(target_faces), axis = 0)

	generated_batch = forward_generate_frame_batch(feature_volumes_stacked, target_motion_points_stacked, rest_motion_points_stacked)

	for index, (affine_matrix, crop_masks, target_size) in enumerate(contexts):
		animated_crop = normalize_crop_frame(generated_batch[index])
		animated_crop = cv2.resize(animated_crop, target_size, interpolation = cv2.INTER_CUBIC)
		crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
		temp_vision_frame = paste_back(temp_vision_frame, animated_crop, crop_mask, affine_matrix)
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


def forward_generate_frame(feature_volume : LivePortraitFeatureVolume, source_motion_points : LivePortraitMotionPoints, target_motion_points : LivePortraitMotionPoints) -> VisionFrame:
	generator = get_inference_pool().get('generator')

	with thread_semaphore():
		crop_vision_frame = generator.run(None,
		{
			'feature_volume': feature_volume,
			'source': source_motion_points,
			'target': target_motion_points
		})[0][0]

	return crop_vision_frame


def forward_extract_motion_batch(crop_vision_frames : numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
	"""Run ``motion_extractor`` once across a stacked ``(N, ...)`` batch
	of crops. Returns the same 7-tuple of arrays as
	``forward_extract_motion`` but with a leading ``N`` axis on each. With
	a dynamic-batch ONNX export this collapses the per-face loop into a
	single ``session.run`` call; otherwise the helper falls back to N
	sequential calls so the per-face slice is bit-equal."""
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


def forward_generate_frame_batch(feature_volumes : numpy.ndarray, source_motion_points : numpy.ndarray, target_motion_points : numpy.ndarray) -> numpy.ndarray:
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
					'source': source_motion_points,
					'target': target_motion_points
				})[0]
			except Exception:  # pragma: no cover - defensive fall-through
				pass
		per_face : List[numpy.ndarray] = []
		for index in range(feature_volumes.shape[0]):
			generated = generator.run(None,
			{
				'feature_volume': feature_volumes[index : index + 1],
				'source': source_motion_points[index : index + 1],
				'target': target_motion_points[index : index + 1]
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


def _resolve_source_state() -> Optional[dict]:
	source_paths = state_manager.get_item('source_paths') or []
	source_image_paths = tuple(filter_image_paths(source_paths))

	if not source_image_paths:
		return None
	state = _get_cached_source_state(source_image_paths)
	if state is not None:
		return state
	state = _build_source_state(source_image_paths)
	if state is not None:
		_store_cached_source_state(source_image_paths, state)
	return state


def _build_source_state(source_image_paths : Tuple[str, ...]) -> Optional[dict]:
	source_vision_frames = read_static_images(list(source_image_paths))
	source_faces = get_many_faces(source_vision_frames)
	source_faces = sort_faces_by_order(source_faces, 'large-small')
	source_face = get_one_face(source_faces)

	if source_face is None:
		return None

	# Pick the source frame that actually contained the chosen face.
	# `Face` is a namedtuple holding numpy arrays (bounding_box, embedding,
	# ...); using `in` / `==` on it would invoke numpy's element-wise
	# equality and raise "truth value of an array is ambiguous" whenever
	# two non-identical Face objects were compared. The `face_store`
	# memoises faces per-frame, so the chosen `source_face` is the very
	# same Python object that came out of one of the per-frame
	# `get_many_faces([frame])` calls -- compare by identity to avoid the
	# ambiguous-truth crash in the multi-source case.
	source_vision_frame = None
	for candidate_frame in source_vision_frames:
		candidate_faces = get_many_faces([ candidate_frame ])
		if any(source_face is candidate_face for candidate_face in candidate_faces):
			source_vision_frame = candidate_frame
			break
	if source_vision_frame is None:
		source_vision_frame = source_vision_frames[0]

	model_template = get_model_options().get('template')
	model_size = get_model_options().get('size')
	source_crop_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame, source_face.landmark_set.get('5/68'), model_template, model_size)
	source_input = prepare_crop_frame(source_crop_vision_frame)
	feature_volume = forward_extract_feature(source_input)
	pitch, yaw, roll, scale, translation, expression, motion_points = forward_extract_motion(source_input)
	rotation = create_rotation(pitch, yaw, roll)
	rest_motion_points = scale * (motion_points @ rotation.T + expression) + translation
	return\
	{
		'feature_volume': feature_volume,
		'motion_points': motion_points,
		'rest_motion_points': rest_motion_points,
		'pitch': pitch,
		'yaw': yaw,
		'roll': roll,
		'scale': scale,
		'translation': translation,
		'expression': expression
	}


# Module-level dict instead of @lru_cache because we need an explicit
# `clear_source_state_cache()` for `post_process` and tests.
_SOURCE_STATE_CACHE : "dict[Tuple[str, ...], dict]" = {}


def _get_cached_source_state(key : Tuple[str, ...]) -> Optional[dict]:
	return _SOURCE_STATE_CACHE.get(key)


def _store_cached_source_state(key : Tuple[str, ...], value : dict) -> None:
	while len(_SOURCE_STATE_CACHE) >= _SOURCE_STATE_LRU_SIZE:
		_SOURCE_STATE_CACHE.pop(next(iter(_SOURCE_STATE_CACHE)))
	_SOURCE_STATE_CACHE[key] = value


def clear_source_state_cache() -> None:
	_SOURCE_STATE_CACHE.clear()


def process_frame(inputs : PortraitAnimatorInputs) -> ProcessorOutputs:
	reference_vision_frame = inputs.get('reference_vision_frame')
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	target_faces = select_faces(reference_vision_frame, target_vision_frame)

	if target_faces:
		scaled_faces = [ scale_face(target_face, target_vision_frame, temp_vision_frame) for target_face in target_faces ]
		temp_vision_frame = animate_portraits(scaled_faces, target_vision_frame, temp_vision_frame)

	return temp_vision_frame, temp_vision_mask
