"""Frame interpolator processor module.

Unlike most facefusion processors -- which transform individual frames in
the per-frame pipeline (`process_frame`) -- frame interpolation has to look
at consecutive frames *together* to synthesise intermediate frames. There
is no useful per-frame operation: a single frame in isolation cannot be
interpolated.

For that reason this processor implements `process_frame` as a no-op
identity passthrough (so it satisfies the registry contract and can be
mixed freely with other processors), and does the real work in a new
optional hook `process_video(input_path, output_path)` that the video
workflow invokes once after `merge_frames`/`restore_audio` have produced
a finished MP4.

Activating this processor:

    python facefusion.py headless-run \\
        --processors frame_interpolator \\
        --frame-interpolator-target-fps 60

Backward-compatibility: the legacy top-level `--frame-interpolator-target-fps`
flag (registered in `facefusion.program.create_output_creation_program`)
keeps working without `--processors frame_interpolator`. When the
processor is registered, processor-level args take precedence, the
processor's `pre_check` ensures the model weights are downloaded, and
`post_process` participates in `video_memory_strategy` cleanup.
"""
from argparse import ArgumentParser
from functools import lru_cache
from typing import Optional

import facefusion.jobs.job_manager
import facefusion.jobs.job_store
from facefusion import config, frame_interpolator as frame_interpolator_engine, logger, state_manager, translator, video_manager
from facefusion.common_helper import create_int_metavar
from facefusion.processors.modules.frame_interpolator import choices as frame_interpolator_choices
from facefusion.processors.modules.frame_interpolator.types import FrameInterpolatorInputs
from facefusion.processors.types import ProcessorOutputs
from facefusion.program_helper import find_argument_group
from facefusion.types import ApplyStateItem, Args, DownloadScope, InferencePool, ModelOptions, ModelSet, ProcessMode
from facefusion.vision import read_static_image, read_static_video_frame


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	# Mirror the engine-level static set so that get_model_options() works when external
	# callers (e.g. UIs, doctor) introspect the processor without touching the engine.
	return frame_interpolator_engine.create_static_model_set(download_scope)


def get_inference_pool() -> InferencePool:
	return frame_interpolator_engine.get_inference_pool(_get_active_model())


def clear_inference_pool() -> None:
	frame_interpolator_engine.clear_inference_pool(_get_active_model())


def get_model_options() -> ModelOptions:
	return frame_interpolator_engine.get_model_options(_get_active_model())


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument(
			'--frame-interpolator-model',
			help = translator.get('help.model', __package__),
			default = config.get_str_value('processors', 'frame_interpolator_model', 'rife_4_9'),
			choices = frame_interpolator_choices.frame_interpolator_models
		)
		group_processors.add_argument(
			'--frame-interpolator-multiplier',
			help = translator.get('help.multiplier', __package__),
			type = int,
			default = config.get_int_value('processors', 'frame_interpolator_multiplier'),
			choices = frame_interpolator_choices.frame_interpolator_multiplier_range,
			metavar = create_int_metavar(frame_interpolator_choices.frame_interpolator_multiplier_range)
		)
		facefusion.jobs.job_store.register_step_keys([ 'frame_interpolator_model', 'frame_interpolator_multiplier' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('frame_interpolator_model', args.get('frame_interpolator_model'))
	apply_state_item('frame_interpolator_multiplier', args.get('frame_interpolator_multiplier'))


def pre_check() -> bool:
	model_name = _get_active_model()
	return frame_interpolator_engine.pre_check(model_name)


def pre_process(mode : ProcessMode) -> bool:
	# The processor is video-only. In image / preview modes it would never
	# fire `process_video`, so we just allow it through (process_frame is a
	# no-op anyway). The workflow itself decides when to call us.
	if mode == 'output' and not _has_target_or_multiplier_configured():
		logger.warn(translator.get('frame_interpolator_no_config'), __name__)
	return True


def post_process() -> None:
	read_static_image.cache_clear()
	read_static_video_frame.cache_clear()
	video_manager.clear_video_pool()
	if state_manager.get_item('video_memory_strategy') in [ 'strict', 'moderate' ]:
		clear_inference_pool()


def process_frame(inputs : FrameInterpolatorInputs) -> ProcessorOutputs:
	"""Identity passthrough.

	Frame interpolation is a video-level operation that needs adjacent
	frames to synthesise intermediates; there is no useful single-frame
	transform. The actual interpolation runs in `process_video`.
	"""
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	return temp_vision_frame, temp_vision_mask


def process_video(input_path : str, output_path : str, source_fps : Optional[float] = None) -> int:
	"""Run RIFE on the rendered video at ``input_path`` and write the
	interpolated result to ``output_path`` (caller is responsible for any
	in-place rename). Returns 0 on success, non-zero on failure.

	The function is invoked once per workflow run by
	`facefusion.workflows.image_to_video.interpolate_output_video` when
	this processor is in `state_manager.get_item('processors')`.
	"""
	multiplier = resolve_multiplier(source_fps)
	if multiplier is None:
		logger.warn(translator.get('frame_interpolator_no_config'), __name__)
		return 0
	if multiplier < 2:
		logger.debug(f'frame_interpolator: multiplier resolved to {multiplier}; skipping', __name__)
		return 0

	rc = frame_interpolator_engine.interpolate_video_file(
		input_path = input_path,
		output_path = output_path,
		multiplier = multiplier,
		model_name = _get_active_model(),
		video_encoder = state_manager.get_item('output_video_encoder'),
		video_quality = state_manager.get_item('output_video_quality'),
		video_preset = state_manager.get_item('output_video_preset')
	)
	return rc


def resolve_multiplier(source_fps : Optional[float]) -> Optional[int]:
	"""Reconcile `--frame-interpolator-multiplier` and
	`--frame-interpolator-target-fps`. The multiplier flag, when set,
	wins; otherwise we derive ``round(target_fps / source_fps)`` (capped
	at 2). Returns None when neither is configured.
	"""
	explicit_multiplier = state_manager.get_item('frame_interpolator_multiplier')
	if explicit_multiplier is not None and explicit_multiplier > 0:
		return max(2, int(explicit_multiplier))

	target_fps = state_manager.get_item('frame_interpolator_target_fps')
	if not target_fps:
		return None
	if not source_fps or source_fps <= 0:
		logger.warn('frame_interpolator: source fps is unknown; cannot derive multiplier', __name__)
		return None
	if target_fps <= source_fps:
		logger.warn(f'frame_interpolator: target_fps ({target_fps}) <= source ({source_fps}); skipping', __name__)
		return None
	return max(2, int(round(target_fps / source_fps)))


def is_active() -> bool:
	"""Whether the user has supplied enough configuration to actually
	interpolate. Used by the workflow to decide between this processor
	and the legacy output_creation flag path."""
	return _has_target_or_multiplier_configured()


def _has_target_or_multiplier_configured() -> bool:
	target_fps = state_manager.get_item('frame_interpolator_target_fps')
	multiplier = state_manager.get_item('frame_interpolator_multiplier')
	return bool(target_fps) or bool(multiplier)


def _get_active_model() -> str:
	return state_manager.get_item('frame_interpolator_model') or 'rife_4_9'
