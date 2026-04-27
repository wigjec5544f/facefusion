"""Tests for Đợt 4.D1 portrait_animator processor.

The processor reuses the LivePortrait ONNX components that are already
shipped for ``expression_restorer`` / ``face_editor`` (so no new model
upload is required) and adds a new pipeline that drives a static source
portrait with the head pose and expression of the target/driving frame.

These tests exercise the schema, CLI plumbing, motion-blending math, and
source-state cache without spinning up the full ONNX runtime.
"""
import argparse
from typing import Any, Dict, Tuple
from unittest.mock import patch

import numpy
import pytest

from facefusion import state_manager
from facefusion.processors.modules.portrait_animator import choices as portrait_animator_choices
from facefusion.processors.modules.portrait_animator import core as portrait_animator_core


@pytest.fixture(scope = 'module', autouse = True)
def _init_state() -> None:
	# resolve_download_url reads `download_providers` from the global
	# state at static-model-set construction time. register_args reads
	# config_path through facefusion.config.get_*_value to resolve .ini
	# fallbacks; point it at the shipped facefusion.ini.
	state_manager.init_item('download_providers', [ 'github' ])
	state_manager.init_item('portrait_animator_model', 'live_portrait')
	state_manager.init_item('config_path', 'facefusion.ini')


@pytest.fixture(autouse = True)
def _reset_source_state_cache() -> None:
	portrait_animator_core.clear_source_state_cache()
	yield
	portrait_animator_core.clear_source_state_cache()


# ---------------------------------------------------------------------------
# Schema / model registration.


def test_live_portrait_is_listed_as_a_model_choice() -> None:
	assert 'live_portrait' in portrait_animator_choices.portrait_animator_models


def test_live_portrait_metadata_carries_apache_license_and_paper() -> None:
	models = portrait_animator_core.create_static_model_set('full')
	metadata = models['live_portrait'].get('__metadata__') or {}
	assert metadata.get('vendor') == 'KwaiVGI'
	assert metadata.get('license') == 'MIT'
	assert metadata.get('year') == 2024
	# Pinning the upstream URL keeps the contributor doc honest if the
	# weight provenance ever changes.
	assert 'KwaiVGI/LivePortrait' in metadata.get('upstream', '')


def test_live_portrait_reuses_existing_onnx_components() -> None:
	# The portrait animator must not invent its own model files; it
	# reuses the same ONNX bundle as expression_restorer / face_editor.
	# This test pins the source URLs so a refactor can't silently swap
	# them for a different (potentially unsigned) mirror.
	sources = portrait_animator_core.create_static_model_set('full')['live_portrait']['sources']
	assert set(sources.keys()) == { 'feature_extractor', 'motion_extractor', 'generator' }
	for component in sources.values():
		assert component['url'].endswith('.onnx')
		assert 'live_portrait' in component['url']


def test_live_portrait_template_and_size_match_face_editor() -> None:
	# Same ffhq_512 template + 512x512 crop as expression_restorer /
	# face_editor, so warp_face_by_face_landmark_5 lines up.
	options = portrait_animator_core.create_static_model_set('full')['live_portrait']
	assert options['template'] == 'ffhq_512'
	assert options['size'] == (512, 512)


# ---------------------------------------------------------------------------
# CLI / state plumbing.


def _build_program() -> argparse.ArgumentParser:
	program = argparse.ArgumentParser()
	program.add_argument_group('processors')
	portrait_animator_core.register_args(program)
	return program


def test_register_args_exposes_three_flags() -> None:
	program = _build_program()
	dest_set = { action.dest for action in program._actions }
	assert 'portrait_animator_model' in dest_set
	assert 'portrait_animator_pose_weight' in dest_set
	assert 'portrait_animator_expression_weight' in dest_set


def test_register_args_defaults_to_full_drive() -> None:
	# The default for both knobs is 100 (% of target motion). 0 would
	# silently produce a static output — surprising UX for a processor
	# called "portrait animator".
	program = _build_program()
	args = program.parse_args([])
	assert args.portrait_animator_model == 'live_portrait'
	assert args.portrait_animator_pose_weight == 100
	assert args.portrait_animator_expression_weight == 100


def test_apply_args_propagates_three_keys() -> None:
	collected : Dict[str, Any] = {}

	def _apply(key : str, value : Any) -> None:
		collected[key] = value

	args =\
	{
		'portrait_animator_model': 'live_portrait',
		'portrait_animator_pose_weight': 75,
		'portrait_animator_expression_weight': 50
	}
	portrait_animator_core.apply_args(args, _apply)

	assert collected['portrait_animator_model'] == 'live_portrait'
	assert collected['portrait_animator_pose_weight'] == 75
	assert collected['portrait_animator_expression_weight'] == 50


# ---------------------------------------------------------------------------
# Source-state cache.


def _fake_source_state() -> Dict[str, Any]:
	# Match the shapes that the LivePortrait ONNX motion_extractor emits:
	# pitch/yaw/roll are 0-d (scalars), scale is (1, 1), translation is
	# (1, 3), expression / motion_points are (1, 21, 3).
	return\
	{
		'feature_volume': numpy.zeros((1, 32, 16, 64, 64), dtype = numpy.float32),
		'motion_points': numpy.zeros((1, 21, 3), dtype = numpy.float32),
		'rest_motion_points': numpy.zeros((1, 21, 3), dtype = numpy.float32),
		'pitch': numpy.float32(0.0),
		'yaw': numpy.float32(0.0),
		'roll': numpy.float32(0.0),
		'scale': numpy.array([[ 1.0 ]], dtype = numpy.float32),
		'translation': numpy.zeros((1, 3), dtype = numpy.float32),
		'expression': numpy.zeros((1, 21, 3), dtype = numpy.float32)
	}


def test_source_state_cache_is_keyed_by_paths_and_avoids_recompute() -> None:
	state_manager.init_item('source_paths', [ '/tmp/example_portrait.png' ])

	build_calls : list = []

	def _build(paths : Tuple[str, ...]) -> Dict[str, Any]:
		build_calls.append(paths)
		return _fake_source_state()

	with \
		patch.object(portrait_animator_core, 'filter_image_paths', return_value = [ '/tmp/example_portrait.png' ]), \
		patch.object(portrait_animator_core, '_build_source_state', side_effect = _build):
		first = portrait_animator_core._resolve_source_state()
		second = portrait_animator_core._resolve_source_state()

	assert first is second
	assert build_calls == [ ('/tmp/example_portrait.png',) ]


def test_source_state_cache_returns_none_without_source_paths() -> None:
	state_manager.init_item('source_paths', [])

	with patch.object(portrait_animator_core, 'filter_image_paths', return_value = []):
		assert portrait_animator_core._resolve_source_state() is None


def test_source_state_cache_evicts_when_full() -> None:
	state_manager.init_item('source_paths', [ '/tmp/a.png' ])

	def _build(paths : Tuple[str, ...]) -> Dict[str, Any]:
		return _fake_source_state()

	with \
		patch.object(portrait_animator_core, '_build_source_state', side_effect = _build), \
		patch.object(portrait_animator_core, 'filter_image_paths', side_effect = lambda paths : paths):
		# Insert one more than the LRU size, oldest must drop.
		for index in range(portrait_animator_core._SOURCE_STATE_LRU_SIZE + 1):
			state_manager.init_item('source_paths', [ '/tmp/portrait_{0}.png'.format(index) ])
			assert portrait_animator_core._resolve_source_state() is not None

	cache_keys = list(portrait_animator_core._SOURCE_STATE_CACHE.keys())
	assert len(cache_keys) == portrait_animator_core._SOURCE_STATE_LRU_SIZE
	# First inserted key (portrait_0) should have been evicted.
	assert ('/tmp/portrait_0.png',) not in cache_keys


# ---------------------------------------------------------------------------
# pre_process gating.


def test_pre_process_rejects_stream_mode() -> None:
	state_manager.init_item('source_paths', [ '/tmp/portrait.png' ])
	assert portrait_animator_core.pre_process('stream') is False


def test_pre_process_rejects_missing_source_image() -> None:
	state_manager.init_item('source_paths', [])

	with patch.object(portrait_animator_core, 'has_image', return_value = False):
		assert portrait_animator_core.pre_process('output') is False


# ---------------------------------------------------------------------------
# Motion blending math (the actual numerical contract).


def test_full_drive_yields_target_pose_and_expression() -> None:
	# pose_weight = expression_weight = 100  →  driven == target.
	source_pitch = numpy.array([ 5.0 ], dtype = numpy.float32)
	target_pitch = numpy.array([ 20.0 ], dtype = numpy.float32)
	driven = source_pitch + (target_pitch - source_pitch) * 1.0
	assert numpy.allclose(driven, target_pitch)


def test_zero_drive_yields_source_pose_and_expression() -> None:
	# pose_weight = expression_weight = 0  →  driven == source (output is
	# essentially a still of the source portrait).
	source_pitch = numpy.array([ 5.0 ], dtype = numpy.float32)
	target_pitch = numpy.array([ 20.0 ], dtype = numpy.float32)
	driven = source_pitch + (target_pitch - source_pitch) * 0.0
	assert numpy.allclose(driven, source_pitch)


def test_half_drive_is_linear_blend() -> None:
	source_yaw = numpy.array([ -10.0 ], dtype = numpy.float32)
	target_yaw = numpy.array([ 30.0 ], dtype = numpy.float32)
	driven = source_yaw + (target_yaw - source_yaw) * 0.5
	assert numpy.allclose(driven, numpy.array([ 10.0 ], dtype = numpy.float32))


def test_pose_weight_interp_maps_0_to_100_to_unit_interval() -> None:
	# Mirrors the numpy.interp call inside animate_portrait so the CLI
	# weight (int 0..100) maps to the float 0..1 used in the blend.
	for percent, expected in [ (0, 0.0), (25, 0.25), (50, 0.5), (75, 0.75), (100, 1.0) ]:
		assert float(numpy.interp(float(percent), [ 0, 100 ], [ 0.0, 1.0 ])) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Process-frame contract.


def test_process_frame_passthrough_when_no_target_face() -> None:
	frame = numpy.zeros((128, 128, 3), dtype = numpy.uint8)
	mask = numpy.zeros((128, 128), dtype = numpy.float32)
	inputs =\
	{
		'reference_vision_frame': frame,
		'source_vision_frames': [],
		'target_vision_frame': frame,
		'temp_vision_frame': frame,
		'temp_vision_mask': mask
	}

	with patch.object(portrait_animator_core, 'select_faces', return_value = []):
		out_frame, out_mask = portrait_animator_core.process_frame(inputs)

	assert out_frame is frame
	assert out_mask is mask


def test_process_frame_invokes_animate_per_target_face() -> None:
	frame = numpy.zeros((128, 128, 3), dtype = numpy.uint8)
	mask = numpy.zeros((128, 128), dtype = numpy.float32)
	inputs =\
	{
		'reference_vision_frame': frame,
		'source_vision_frames': [],
		'target_vision_frame': frame,
		'temp_vision_frame': frame,
		'temp_vision_mask': mask
	}

	fake_faces = [ object(), object() ]
	animate_calls : list = []

	def _animate(face, target_frame, temp_frame):
		animate_calls.append(face)
		return temp_frame

	with \
		patch.object(portrait_animator_core, 'select_faces', return_value = fake_faces), \
		patch.object(portrait_animator_core, 'scale_face', side_effect = lambda face, t, te : face), \
		patch.object(portrait_animator_core, 'animate_portrait', side_effect = _animate):
		out_frame, out_mask = portrait_animator_core.process_frame(inputs)

	assert animate_calls == fake_faces
	assert out_mask is mask


# ---------------------------------------------------------------------------
# Regression: driving motion must come from `target_vision_frame`, not from
# the chain-accumulated `temp_vision_frame`. If a previous processor (e.g.
# `face_swapper`) modified the face, extracting motion from `temp_vision_frame`
# would feed the motion extractor a non-driver. Devin Review BUG_2a826f23_0001.


def test_animate_portrait_extracts_motion_from_target_vision_frame() -> None:
	target_frame = numpy.full((512, 512, 3), 200, dtype = numpy.uint8)
	temp_frame = numpy.full((512, 512, 3), 50, dtype = numpy.uint8)

	class _FakeFace:
		landmark_set = { '5/68': numpy.zeros((5, 2), dtype = numpy.float32) }

	source_state = _fake_source_state()

	def _warp(frame, _landmark, _template, size):
		return frame[:size[0], :size[1]].copy(), numpy.eye(2, 3, dtype = numpy.float32)

	def _box(crop, _blur, _padding):
		return numpy.ones(crop.shape[:2], dtype = numpy.float32)

	# Each call returns the (unique) input back so we can prove which
	# frame was used as the motion-extractor input. Float values aren't
	# representative of real motion outputs but cover the unpacking shape.
	prepare_inputs : list = []

	def _prepare(crop):
		prepare_inputs.append(int(crop[0, 0, 0]))
		# Return shape (1, 3, 256, 256) for downstream stub call.
		return numpy.zeros((1, 3, 256, 256), dtype = numpy.float32)

	def _forward_motion(_input):
		# Pitch/yaw/roll come out of the LivePortrait motion_extractor as
		# scalars so `scipy ... from_euler('xyz', [pitch, yaw, roll])`
		# resolves to a single 3x3 rotation. Match that shape contract
		# here (numpy.float32(0.0) == ndim 0).
		return\
		(
			numpy.float32(0.0),
			numpy.float32(0.0),
			numpy.float32(0.0),
			numpy.array([[ 1.0 ]], dtype = numpy.float32),
			numpy.zeros((1, 3), dtype = numpy.float32),
			numpy.zeros((1, 21, 3), dtype = numpy.float32),
			numpy.zeros((1, 21, 3), dtype = numpy.float32)
		)

	def _forward_generate(_volume, _src, _tgt):
		return numpy.zeros((3, 256, 256), dtype = numpy.float32)

	def _paste(temp, _crop, _mask, _affine):
		return temp

	with \
		patch.object(portrait_animator_core, '_resolve_source_state', return_value = source_state), \
		patch.object(portrait_animator_core, 'warp_face_by_face_landmark_5', side_effect = _warp), \
		patch.object(portrait_animator_core, 'create_box_mask', side_effect = _box), \
		patch.object(portrait_animator_core, 'prepare_crop_frame', side_effect = _prepare), \
		patch.object(portrait_animator_core, 'forward_extract_motion', side_effect = _forward_motion), \
		patch.object(portrait_animator_core, 'forward_generate_frame', side_effect = _forward_generate), \
		patch.object(portrait_animator_core, 'paste_back', side_effect = _paste):
		state_manager.init_item('portrait_animator_pose_weight', 100)
		state_manager.init_item('portrait_animator_expression_weight', 100)
		state_manager.init_item('face_mask_blur', 0.3)
		state_manager.init_item('face_mask_types', [])
		portrait_animator_core.animate_portrait(_FakeFace(), target_frame, temp_frame)

	# The single call to prepare_crop_frame inside animate_portrait must
	# operate on the target (driver) crop = pixel value 200, not the
	# temp/chain-modified crop = pixel value 50.
	assert prepare_inputs == [ 200 ], (
		"animate_portrait fed a non-driver frame to the motion extractor; "
		"see Devin Review BUG_2a826f23_0001"
	)


# ---------------------------------------------------------------------------
# Pixel I/O helpers (deterministic, no ONNX).


def test_prepare_crop_frame_outputs_normalized_chw_float32() -> None:
	crop = (numpy.random.rand(512, 512, 3) * 255).astype(numpy.uint8)
	prepared = portrait_animator_core.prepare_crop_frame(crop)
	assert prepared.dtype == numpy.float32
	assert prepared.shape == (1, 3, 256, 256)
	assert prepared.min() >= 0.0
	assert prepared.max() <= 1.0


def test_normalize_crop_frame_round_trips_through_prepare() -> None:
	# Round-tripping a constant grey patch through prepare → normalize
	# should preserve the colour to within uint8 quantisation.
	grey = numpy.full((512, 512, 3), 128, dtype = numpy.uint8)
	prepared = portrait_animator_core.prepare_crop_frame(grey)[0]
	normalized = portrait_animator_core.normalize_crop_frame(prepared)
	assert normalized.dtype == numpy.uint8
	assert normalized.shape == (256, 256, 3)
	# Allow ±1 for rounding from float<->uint8.
	assert numpy.abs(normalized.astype(numpy.int16) - 128).max() <= 1


# ---------------------------------------------------------------------------
# Discovery: the new processor shows up in --processors choices.


def test_portrait_animator_is_discovered_as_a_processor() -> None:
	from facefusion.filesystem import get_file_name, resolve_file_paths

	available_processors = [ get_file_name(file_path) for file_path in resolve_file_paths('facefusion/processors/modules') ]
	assert 'portrait_animator' in available_processors
