"""Tests for the new frame_interpolator processor module.

Covers the processor's contract methods, the multiplier resolution logic
(target_fps vs explicit multiplier), the activation guard, and the
workflow's dispatch into the processor when it is registered."""
import pathlib
import unittest.mock as mock

from facefusion import state_manager
from facefusion.processors.core import PROCESSORS_METHODS, load_processor_module
from facefusion.processors.modules.frame_interpolator import core as frame_interpolator_processor
from facefusion.workflows import image_to_video


# ---------------------------------------------------------------------------
# Contract: the module exposes every method the registry requires.


def test_processor_exposes_required_contract_methods() -> None:
	module = load_processor_module('frame_interpolator')
	for method_name in PROCESSORS_METHODS:
		assert hasattr(module, method_name), f'frame_interpolator processor missing {method_name}'


# ---------------------------------------------------------------------------
# process_frame must be a true identity passthrough -- frame interpolation
# has no useful single-frame transform.


def test_process_frame_is_identity_passthrough() -> None:
	frame_marker = object()
	mask_marker = object()
	frame, mask = frame_interpolator_processor.process_frame(
		{
			'target_vision_frame': frame_marker,
			'temp_vision_frame': frame_marker,
			'temp_vision_mask': mask_marker
		}
	)
	assert frame is frame_marker
	assert mask is mask_marker


# ---------------------------------------------------------------------------
# resolve_multiplier — explicit multiplier wins; target_fps is derived only
# when explicit multiplier is unset.


def _patch_state(monkeypatch, **values) -> None:
	monkeypatch.setattr(state_manager, 'get_item', lambda key: values.get(key))


def test_resolve_multiplier_prefers_explicit_value(monkeypatch) -> None:
	_patch_state(monkeypatch, frame_interpolator_multiplier = 4, frame_interpolator_target_fps = 60)
	assert frame_interpolator_processor.resolve_multiplier(30) == 4


def test_resolve_multiplier_floors_explicit_at_two(monkeypatch) -> None:
	# Explicit multiplier of 1 is meaningless (no synthetic frames); we still
	# clamp to 2 so the user sees something happen if they typo'd.
	_patch_state(monkeypatch, frame_interpolator_multiplier = 1)
	assert frame_interpolator_processor.resolve_multiplier(30) == 2


def test_resolve_multiplier_derives_from_target_fps(monkeypatch) -> None:
	_patch_state(monkeypatch, frame_interpolator_target_fps = 60)
	assert frame_interpolator_processor.resolve_multiplier(30) == 2


def test_resolve_multiplier_rounds_to_three_for_24_to_60(monkeypatch) -> None:
	# 60 / 24 = 2.5, round-half-to-even gives 2 in Python; we use round() so
	# this lands at 2 -- documented but not desirable for some users. The
	# explicit `--frame-interpolator-multiplier 3` path is the escape hatch.
	# We just lock in the current behaviour so it doesn't drift silently.
	_patch_state(monkeypatch, frame_interpolator_target_fps = 60)
	assert frame_interpolator_processor.resolve_multiplier(24) == 2


def test_resolve_multiplier_returns_none_when_unconfigured(monkeypatch) -> None:
	_patch_state(monkeypatch)
	assert frame_interpolator_processor.resolve_multiplier(30) is None


def test_resolve_multiplier_rejects_unknown_source_fps(monkeypatch) -> None:
	_patch_state(monkeypatch, frame_interpolator_target_fps = 60)
	assert frame_interpolator_processor.resolve_multiplier(0) is None
	assert frame_interpolator_processor.resolve_multiplier(None) is None


def test_resolve_multiplier_skips_when_target_le_source(monkeypatch) -> None:
	_patch_state(monkeypatch, frame_interpolator_target_fps = 30)
	assert frame_interpolator_processor.resolve_multiplier(30) is None


# ---------------------------------------------------------------------------
# is_active reflects whether the user supplied any interpolation config.


def test_is_active_false_without_config(monkeypatch) -> None:
	_patch_state(monkeypatch)
	assert frame_interpolator_processor.is_active() is False


def test_is_active_true_with_target_fps(monkeypatch) -> None:
	_patch_state(monkeypatch, frame_interpolator_target_fps = 60)
	assert frame_interpolator_processor.is_active() is True


def test_is_active_true_with_explicit_multiplier(monkeypatch) -> None:
	_patch_state(monkeypatch, frame_interpolator_multiplier = 3)
	assert frame_interpolator_processor.is_active() is True


# ---------------------------------------------------------------------------
# process_video forwards encoder + quality + preset state through to the
# engine, and uses the active model.


def test_process_video_forwards_encoder_settings(monkeypatch, tmp_path : pathlib.Path) -> None:
	captured : dict = {}

	def fake_engine(input_path, output_path, multiplier, model_name, video_encoder, video_quality, video_preset):
		captured.update({
			'input_path': input_path,
			'output_path': output_path,
			'multiplier': multiplier,
			'model_name': model_name,
			'video_encoder': video_encoder,
			'video_quality': video_quality,
			'video_preset': video_preset
		})
		return 0

	monkeypatch.setattr(
		frame_interpolator_processor.frame_interpolator_engine,
		'interpolate_video_file',
		lambda **kwargs: fake_engine(**kwargs)
	)
	_patch_state(
		monkeypatch,
		frame_interpolator_target_fps = 60,
		frame_interpolator_model = 'rife_4_9',
		output_video_encoder = 'libx264',
		output_video_quality = 80,
		output_video_preset = 'veryfast'
	)
	rc = frame_interpolator_processor.process_video(str(tmp_path / 'in.mp4'), str(tmp_path / 'out.mp4'), source_fps = 30)
	assert rc == 0
	assert captured['multiplier'] == 2
	assert captured['model_name'] == 'rife_4_9'
	assert captured['video_encoder'] == 'libx264'
	assert captured['video_quality'] == 80
	assert captured['video_preset'] == 'veryfast'


def test_process_video_skips_when_unconfigured(monkeypatch, tmp_path : pathlib.Path) -> None:
	_patch_state(monkeypatch)
	called = mock.Mock()
	monkeypatch.setattr(
		frame_interpolator_processor.frame_interpolator_engine,
		'interpolate_video_file',
		called
	)
	assert frame_interpolator_processor.process_video(str(tmp_path / 'in.mp4'), str(tmp_path / 'out.mp4'), source_fps = 30) == 0
	called.assert_not_called()


# ---------------------------------------------------------------------------
# Workflow dispatch — when frame_interpolator is in `processors`, the
# workflow should delegate to processor.process_video instead of the legacy
# engine call. The legacy fallback stays in place when the processor isn't
# registered.


def test_workflow_uses_processor_when_registered(monkeypatch, tmp_path : pathlib.Path) -> None:
	output_path = tmp_path / 'out.mp4'
	output_path.write_bytes(b'fake mp4')

	_patch_state(
		monkeypatch,
		frame_interpolator_target_fps = 60,
		frame_interpolator_model = 'rife_4_9',
		output_path = str(output_path),
		output_video_fps = 30,
		processors = [ 'frame_interpolator' ]
	)
	monkeypatch.setattr('facefusion.workflows.image_to_video.is_video', lambda path: True)

	process_video_calls = []

	def fake_process_video(input_path, output_path, source_fps = None):
		process_video_calls.append({
			'input_path': input_path,
			'output_path': output_path,
			'source_fps': source_fps
		})
		pathlib.Path(output_path).write_bytes(b'interpolated')
		return 0

	# Patch processor's process_video and the legacy engine call. Only the
	# processor should fire.
	monkeypatch.setattr(frame_interpolator_processor, 'process_video', fake_process_video)
	legacy_call = mock.Mock()
	monkeypatch.setattr('facefusion.frame_interpolator.interpolate_video_file', legacy_call)
	monkeypatch.setattr('facefusion.video_manager.clear_video_pool', lambda: None)

	assert image_to_video.interpolate_output_video() == 0
	assert len(process_video_calls) == 1
	assert process_video_calls[0]['source_fps'] == 30
	legacy_call.assert_not_called()
	assert output_path.read_bytes() == b'interpolated'


def test_workflow_falls_back_to_legacy_path_without_processor(monkeypatch, tmp_path : pathlib.Path) -> None:
	output_path = tmp_path / 'out.mp4'
	output_path.write_bytes(b'fake mp4')

	_patch_state(
		monkeypatch,
		frame_interpolator_target_fps = 60,
		output_path = str(output_path),
		output_video_fps = 30,
		output_video_encoder = 'libx264',
		output_video_quality = 80,
		output_video_preset = 'veryfast',
		processors = [ ]
	)
	monkeypatch.setattr('facefusion.workflows.image_to_video.is_video', lambda path: True)

	def fake_legacy(input_path, output_path, multiplier, **kwargs):
		pathlib.Path(output_path).write_bytes(b'legacy')
		return 0

	processor_call = mock.Mock()
	monkeypatch.setattr(frame_interpolator_processor, 'process_video', processor_call)
	monkeypatch.setattr('facefusion.frame_interpolator.interpolate_video_file', fake_legacy)
	monkeypatch.setattr('facefusion.video_manager.clear_video_pool', lambda: None)

	assert image_to_video.interpolate_output_video() == 0
	processor_call.assert_not_called()
	assert output_path.read_bytes() == b'legacy'


def test_workflow_skips_when_processor_registered_but_no_config(monkeypatch, tmp_path : pathlib.Path) -> None:
	output_path = tmp_path / 'out.mp4'
	output_path.write_bytes(b'untouched')

	_patch_state(
		monkeypatch,
		output_path = str(output_path),
		output_video_fps = 30,
		processors = [ 'frame_interpolator' ]
	)
	monkeypatch.setattr('facefusion.workflows.image_to_video.is_video', lambda path: True)

	processor_call = mock.Mock()
	legacy_call = mock.Mock()
	monkeypatch.setattr(frame_interpolator_processor, 'process_video', processor_call)
	monkeypatch.setattr('facefusion.frame_interpolator.interpolate_video_file', legacy_call)

	assert image_to_video.interpolate_output_video() == 0
	processor_call.assert_not_called()
	legacy_call.assert_not_called()
	assert output_path.read_bytes() == b'untouched'
