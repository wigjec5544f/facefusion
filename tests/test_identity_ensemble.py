"""Tests for multi-source identity fusion (Đợt 1.C3, PR #23).

The ``face_swapper`` module historically averaged every source face's
ArcFace embedding with a naive ``numpy.mean``. PR #23 introduces four
fusion modes -- ``mean`` (legacy, bit-equal), ``weighted`` (per-face
quality weights), ``slerp`` (spherical interpolation on the unit
sphere), and ``robust`` (outlier rejection + weighted mean) -- exposed
through ``--source-fusion-mode``. These tests cover:

* The mathematical helpers (``_compute_face_quality_weights``,
  ``_weighted_mean``, ``_slerp_pair``, ``_reject_outliers``).
* ``get_fused_face`` dispatch through every mode.
* ``get_average_face`` back-compat -- it must produce the exact bytes
  the legacy ``numpy.mean`` did.
* ``extract_source_face`` plumbing -- the state-driven mode is read at
  call time and routed to ``get_fused_face`` without touching the ONNX
  recogniser.
* ``register_args`` + ``apply_args`` -- the new CLI flags reach the
  state manager unchanged.

All tests are stub-based; no ONNX weights, no network access.
"""
import argparse
from typing import List

import numpy
import pytest

from facefusion import face_analyser, state_manager
from facefusion.face_analyser import (
	_compute_face_quality_weights,
	_reject_outliers,
	_slerp_pair,
	_weighted_mean,
	get_average_face,
	get_fused_face,
)
from facefusion.processors.modules.face_swapper import core as face_swapper_core
from facefusion.types import Face


@pytest.fixture(scope = 'module', autouse = True)
def _init_state_manager() -> None:
	# `register_args` resolves CLI defaults via `facefusion.config.get_*_value`,
	# which in turn reads the config_path from the global state manager.
	# `face_swapper_core.create_static_model_set` also reads
	# `download_providers` at static-model-set construction time. Tests
	# in this module never download anything, but we still need both
	# items present so the import / dispatch paths don't blow up.
	state_manager.init_item('config_path', 'facefusion.ini')
	state_manager.init_item('download_providers', [ 'github' ])


# ---------------------------------------------------------------------------
# Helpers.


def _make_face(
	embedding : numpy.ndarray,
	*,
	detector_score : float = 0.9,
	landmarker_score : float = 0.8,
	bbox : numpy.ndarray = None
) -> Face:
	if bbox is None:
		bbox = numpy.array([ 0.0, 0.0, 100.0, 100.0 ], dtype = numpy.float32)
	embedding = numpy.asarray(embedding, dtype = numpy.float64)
	embedding_norm = embedding / max(float(numpy.linalg.norm(embedding)), 1e-12)
	landmark_set = {
		'5': numpy.zeros((5, 2), dtype = numpy.float32),
		'5/68': numpy.zeros((5, 2), dtype = numpy.float32),
		'68': numpy.zeros((68, 2), dtype = numpy.float32),
		'68/5': numpy.zeros((68, 2), dtype = numpy.float32)
	}
	score_set = { 'detector': float(detector_score), 'landmarker': float(landmarker_score) }
	return Face(
		bounding_box = bbox,
		score_set = score_set,
		landmark_set = landmark_set,
		angle = 0,
		embedding = embedding,
		embedding_norm = embedding_norm,
		gender = 'female',
		age = range(20, 30),
		race = 'white'
	)


def _direction(*coords : float) -> numpy.ndarray:
	# Lift a 2D / 3D direction to a 512-dim vector for shape parity with
	# real ArcFace embeddings while keeping the maths human-readable.
	embedding = numpy.zeros(512, dtype = numpy.float64)
	for index, value in enumerate(coords):
		embedding[index] = float(value)
	return embedding


# ---------------------------------------------------------------------------
# _compute_face_quality_weights.


def test_quality_weights_reflect_detector_score() -> None:
	face_high = _make_face(_direction(1.0, 0.0), detector_score = 0.95, landmarker_score = 0.9)
	face_low = _make_face(_direction(1.0, 0.0), detector_score = 0.10, landmarker_score = 0.9)
	weights = _compute_face_quality_weights([ face_high, face_low ])

	assert weights[0] > weights[1]
	assert weights[0] > 0.0
	assert weights[1] > 0.0


def test_quality_weights_reward_larger_bbox() -> None:
	small_bbox = numpy.array([ 0.0, 0.0, 50.0, 50.0 ], dtype = numpy.float32)
	large_bbox = numpy.array([ 0.0, 0.0, 400.0, 400.0 ], dtype = numpy.float32)
	face_small = _make_face(_direction(1.0, 0.0), bbox = small_bbox)
	face_large = _make_face(_direction(1.0, 0.0), bbox = large_bbox)
	weights = _compute_face_quality_weights([ face_small, face_large ])

	assert weights[1] > weights[0]


def test_quality_weights_apply_landmarker_floor_when_disabled() -> None:
	# When the landmarker is gated off (`face_landmarker_score == 0`),
	# every face's `landmarker` score is 0; we still need usable weights.
	face_a = _make_face(_direction(1.0, 0.0), landmarker_score = 0.0)
	face_b = _make_face(_direction(0.0, 1.0), landmarker_score = 0.0)
	weights = _compute_face_quality_weights([ face_a, face_b ])

	assert numpy.all(weights > 0.0)


def test_quality_weights_fall_back_to_uniform_when_all_zero() -> None:
	face_a = _make_face(_direction(1.0, 0.0), detector_score = 0.0, landmarker_score = 0.0)
	face_b = _make_face(_direction(0.0, 1.0), detector_score = 0.0, landmarker_score = 0.0)
	weights = _compute_face_quality_weights([ face_a, face_b ])

	# detector_score == 0 zeroes the per-face weight; the helper must
	# fall back to a uniform vector so the caller doesn't divide by 0.
	numpy.testing.assert_allclose(weights, numpy.ones(2))


# ---------------------------------------------------------------------------
# _weighted_mean.


def test_weighted_mean_matches_explicit_formula() -> None:
	values = [ numpy.array([ 1.0, 0.0 ]), numpy.array([ 0.0, 1.0 ]) ]
	weights = numpy.array([ 3.0, 1.0 ])
	expected = (3.0 * values[0] + 1.0 * values[1]) / 4.0
	result = _weighted_mean(values, weights)

	numpy.testing.assert_allclose(result, expected)


def test_weighted_mean_collapses_to_arithmetic_mean_for_uniform_weights() -> None:
	values = [ numpy.array([ 2.0, 4.0 ]), numpy.array([ 4.0, 6.0 ]), numpy.array([ 6.0, 8.0 ]) ]
	weights = numpy.array([ 1.0, 1.0, 1.0 ])
	result = _weighted_mean(values, weights)

	numpy.testing.assert_allclose(result, numpy.mean(numpy.stack(values, axis = 0), axis = 0))


# ---------------------------------------------------------------------------
# _slerp_pair.


def test_slerp_pair_at_t0_returns_a() -> None:
	a = numpy.array([ 1.0, 0.0, 0.0 ])
	b = numpy.array([ 0.0, 1.0, 0.0 ])
	result = _slerp_pair(a, b, 0.0)

	numpy.testing.assert_allclose(result, a, atol = 1e-9)


def test_slerp_pair_at_t1_returns_b() -> None:
	a = numpy.array([ 1.0, 0.0, 0.0 ])
	b = numpy.array([ 0.0, 1.0, 0.0 ])
	result = _slerp_pair(a, b, 1.0)

	numpy.testing.assert_allclose(result, b, atol = 1e-9)


def test_slerp_pair_output_is_unit_norm() -> None:
	a = numpy.array([ 1.0, 0.0, 0.0 ])
	b = numpy.array([ 0.0, 1.0, 0.0 ])
	for t in [ 0.0, 0.25, 0.5, 0.75, 1.0 ]:
		result = _slerp_pair(a, b, t)
		assert abs(float(numpy.linalg.norm(result)) - 1.0) < 1e-9


def test_slerp_pair_handles_collinear_inputs() -> None:
	# Vectors very close to each other -> the helper must fall back to
	# linear interpolation + renormalisation instead of dividing by sin(0).
	a = numpy.array([ 1.0, 0.0, 0.0 ])
	b = numpy.array([ 1.0, 1e-8, 0.0 ])
	result = _slerp_pair(a, b, 0.5)

	assert abs(float(numpy.linalg.norm(result)) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# _reject_outliers.


def test_reject_outliers_drops_far_face() -> None:
	# Four sources clustered around (1, 0), one outlier flipped to (-1, 0).
	consistent = [ _make_face(_direction(1.0, 0.05 * index)) for index in range(4) ]
	outlier = _make_face(_direction(-1.0, 0.0))
	survivors = _reject_outliers(consistent + [ outlier ], 0.65)

	# `in` triggers numpy element-wise equality on `Face` namedtuples,
	# so we compare by object identity instead.
	assert len(survivors) == 4
	assert all(id(face) != id(outlier) for face in survivors)


def test_reject_outliers_keeps_all_consistent_sources() -> None:
	faces = [ _make_face(_direction(1.0, 0.05 * index)) for index in range(4) ]
	survivors = _reject_outliers(faces, 0.65)

	assert len(survivors) == 4


def test_reject_outliers_keeps_closest_when_all_flagged() -> None:
	# Two pairs pointing in opposite directions -- their centroid falls
	# at the origin, every face fails the threshold. The helper must
	# still return at least one face (the one closest to the centroid).
	faces = [
		_make_face(_direction(1.0, 0.0)),
		_make_face(_direction(-1.0, 0.0)),
		_make_face(_direction(0.0, 1.0))
	]
	survivors = _reject_outliers(faces, 0.99)

	assert len(survivors) >= 1


def test_reject_outliers_short_circuits_for_single_face() -> None:
	face = _make_face(_direction(1.0, 0.0))
	survivors = _reject_outliers([ face ], 0.65)

	assert survivors == [ face ]


# ---------------------------------------------------------------------------
# get_fused_face dispatch.


def test_get_fused_face_returns_none_for_empty_input() -> None:
	assert get_fused_face([], 'mean', None) is None
	assert get_fused_face([], 'weighted', None) is None
	assert get_fused_face([], 'slerp', None) is None
	assert get_fused_face([], 'robust', 0.65) is None


def test_get_fused_face_mean_is_bit_equal_with_legacy_numpy_mean() -> None:
	# The legacy implementation built `embedding` as
	# `numpy.mean(face_embeddings, axis = 0)` and `embedding_norm` as
	# `numpy.mean(face_embeddings_norm, axis = 0)`. Mode `mean` must
	# reproduce both arrays bit-for-bit.
	faces = [ _make_face(_direction(1.0, 0.2 * index, 0.1)) for index in range(3) ]
	expected_embedding = numpy.mean(numpy.stack([ face.embedding for face in faces ]), axis = 0)
	expected_norm = numpy.mean(numpy.stack([ face.embedding_norm for face in faces ]), axis = 0)
	fused = get_fused_face(faces, 'mean', None)

	assert numpy.array_equal(fused.embedding, expected_embedding)
	assert numpy.array_equal(fused.embedding_norm, expected_norm)


def test_get_fused_face_single_face_short_circuits_to_mean() -> None:
	# With only one source, every mode must collapse to the single
	# face's embedding -- there is nothing to fuse.
	face = _make_face(_direction(1.0, 0.0))
	for mode in [ 'mean', 'weighted', 'slerp', 'robust' ]:
		fused = get_fused_face([ face ], mode, 0.65)
		numpy.testing.assert_allclose(fused.embedding, face.embedding)
		numpy.testing.assert_allclose(fused.embedding_norm, face.embedding_norm)


def test_get_fused_face_preserves_first_face_metadata() -> None:
	face_a = _make_face(_direction(1.0, 0.0))
	face_b = _make_face(_direction(0.0, 1.0))
	fused = get_fused_face([ face_a, face_b ], 'weighted', None)

	# bounding_box / score_set / landmark_set / age etc. always come
	# from the first source, mirroring the legacy `get_average_face`.
	assert fused.bounding_box is face_a.bounding_box
	assert fused.score_set is face_a.score_set
	assert fused.landmark_set is face_a.landmark_set


def test_get_fused_face_weighted_prefers_high_score() -> None:
	# A high-quality (1, 0) source plus a low-quality (0, 1) source. The
	# weighted fusion must pull the result towards (1, 0).
	face_high = _make_face(_direction(1.0, 0.0), detector_score = 0.99, landmarker_score = 0.99)
	face_low = _make_face(_direction(0.0, 1.0), detector_score = 0.10, landmarker_score = 0.10)
	fused = get_fused_face([ face_high, face_low ], 'weighted', None)

	assert fused.embedding[0] > fused.embedding[1]


def test_get_fused_face_slerp_returns_unit_norm() -> None:
	face_a = _make_face(_direction(1.0, 0.0))
	face_b = _make_face(_direction(0.0, 1.0))
	fused = get_fused_face([ face_a, face_b ], 'slerp', None)

	assert abs(float(numpy.linalg.norm(fused.embedding_norm)) - 1.0) < 1e-9


def test_get_fused_face_slerp_midpoint_is_45_degrees() -> None:
	# Two orthogonal unit vectors -> spherical midpoint is the diagonal
	# at 45 degrees, components ~ 1/sqrt(2).
	face_a = _make_face(_direction(1.0, 0.0))
	face_b = _make_face(_direction(0.0, 1.0))
	fused = get_fused_face([ face_a, face_b ], 'slerp', None)

	expected = 1.0 / numpy.sqrt(2.0)
	assert abs(float(fused.embedding_norm[0]) - expected) < 1e-6
	assert abs(float(fused.embedding_norm[1]) - expected) < 1e-6


def test_get_fused_face_robust_rejects_outlier() -> None:
	# Four consistent sources at (1, eps) plus one outlier at (-1, 0).
	# The robust fuser must drop the outlier so the result still points
	# towards (1, 0) instead of being pulled to the origin.
	consistent = [ _make_face(_direction(1.0, 0.01 * index), detector_score = 0.95) for index in range(4) ]
	outlier = _make_face(_direction(-1.0, 0.0), detector_score = 0.95)
	fused = get_fused_face(consistent + [ outlier ], 'robust', 0.65)

	assert fused.embedding[0] > 0.5


def test_get_fused_face_robust_preserves_consistent_sources() -> None:
	# When no source is an outlier, robust must reduce to weighted on the
	# entire set.
	faces = [ _make_face(_direction(1.0, 0.01 * index), detector_score = 0.95) for index in range(4) ]
	fused_robust = get_fused_face(faces, 'robust', 0.65)
	fused_weighted = get_fused_face(faces, 'weighted', None)

	numpy.testing.assert_allclose(fused_robust.embedding, fused_weighted.embedding)
	numpy.testing.assert_allclose(fused_robust.embedding_norm, fused_weighted.embedding_norm)


def test_get_fused_face_rejects_unknown_mode() -> None:
	face = _make_face(_direction(1.0, 0.0))
	with pytest.raises(ValueError):
		get_fused_face([ face, face ], 'mystery', None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_average_face back-compat.


def test_get_average_face_remains_bit_equal_after_refactor() -> None:
	# This is the contract the refactor must keep: callers of the public
	# `get_average_face` API must observe the historical bytes.
	faces = [ _make_face(_direction(1.0, 0.2 * index, 0.1, 0.05 * index)) for index in range(4) ]
	expected_embedding = numpy.mean(numpy.stack([ face.embedding for face in faces ]), axis = 0)
	expected_norm = numpy.mean(numpy.stack([ face.embedding_norm for face in faces ]), axis = 0)
	fused = get_average_face(faces)

	assert numpy.array_equal(fused.embedding, expected_embedding)
	assert numpy.array_equal(fused.embedding_norm, expected_norm)


def test_get_average_face_returns_none_for_empty_list() -> None:
	assert get_average_face([]) is None


# ---------------------------------------------------------------------------
# extract_source_face plumbing.


def _patch_get_many_faces(monkeypatch, faces_per_frame : List[List[Face]]) -> None:
	# `extract_source_face` calls `get_many_faces([source_vision_frame])`
	# once per source path. We replace it with a queue-pop stub so each
	# call returns the next pre-built face list without touching the
	# detector / recogniser ONNX models.
	queue = list(faces_per_frame)

	def _fake(_vision_frames):  # noqa: ARG001
		return queue.pop(0) if queue else []

	monkeypatch.setattr(face_swapper_core, 'get_many_faces', _fake)


def test_extract_source_face_routes_default_mode_through_mean(monkeypatch) -> None:
	# Default state: no `source_fusion_mode` set anywhere -> the helper
	# falls back to 'mean'.
	state_manager.clear_item('source_fusion_mode')
	state_manager.clear_item('source_fusion_outlier_threshold')

	face_a = _make_face(_direction(1.0, 0.0))
	face_b = _make_face(_direction(0.0, 1.0))
	_patch_get_many_faces(monkeypatch, [ [ face_a ], [ face_b ] ])

	captured = {}

	def _fake_get_fused_face(faces, mode, threshold):  # noqa: ARG001
		captured['mode'] = mode
		captured['threshold'] = threshold
		captured['count'] = len(faces)
		return get_fused_face(faces, mode, threshold)

	monkeypatch.setattr(face_swapper_core, 'get_fused_face', _fake_get_fused_face)

	source_vision_frames = [ numpy.zeros((10, 10, 3), dtype = numpy.uint8), numpy.zeros((10, 10, 3), dtype = numpy.uint8) ]
	face_swapper_core.extract_source_face(source_vision_frames)

	assert captured['mode'] == 'mean'
	assert captured['threshold'] is None
	assert captured['count'] == 2


def test_extract_source_face_reads_state_for_robust_mode(monkeypatch) -> None:
	state_manager.init_item('source_fusion_mode', 'robust')
	state_manager.init_item('source_fusion_outlier_threshold', 0.50)

	face_a = _make_face(_direction(1.0, 0.0))
	face_b = _make_face(_direction(0.0, 1.0))
	_patch_get_many_faces(monkeypatch, [ [ face_a ], [ face_b ] ])

	captured = {}

	def _fake_get_fused_face(faces, mode, threshold):  # noqa: ARG001
		captured['mode'] = mode
		captured['threshold'] = threshold
		return get_fused_face(faces, mode, threshold)

	monkeypatch.setattr(face_swapper_core, 'get_fused_face', _fake_get_fused_face)

	source_vision_frames = [ numpy.zeros((10, 10, 3), dtype = numpy.uint8), numpy.zeros((10, 10, 3), dtype = numpy.uint8) ]
	face_swapper_core.extract_source_face(source_vision_frames)

	assert captured['mode'] == 'robust'
	assert captured['threshold'] == 0.50

	# Tidy up so the next test starts from a clean state slot.
	state_manager.clear_item('source_fusion_mode')
	state_manager.clear_item('source_fusion_outlier_threshold')


def test_extract_source_face_returns_none_when_no_faces_detected(monkeypatch) -> None:
	state_manager.clear_item('source_fusion_mode')
	state_manager.clear_item('source_fusion_outlier_threshold')

	_patch_get_many_faces(monkeypatch, [ [], [] ])
	source_vision_frames = [ numpy.zeros((10, 10, 3), dtype = numpy.uint8), numpy.zeros((10, 10, 3), dtype = numpy.uint8) ]
	result = face_swapper_core.extract_source_face(source_vision_frames)

	assert result is None


# ---------------------------------------------------------------------------
# register_args + apply_args.


def test_register_args_adds_source_fusion_flags() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--command')  # placeholder so the parser has a base.
	processors_group = parser.add_argument_group('processors')
	# `find_argument_group` looks the group up by title.
	face_swapper_core.register_args(parser)

	help_strings = [ action.option_strings for action in processors_group._group_actions ]
	flat = [ flag for group in help_strings for flag in group ]
	assert '--source-fusion-mode' in flat
	assert '--source-fusion-outlier-threshold' in flat


def test_apply_args_pushes_fusion_state() -> None:
	captured = {}

	def _capture(key, value):
		captured[key] = value

	args = {
		'face_swapper_model': 'inswapper_128',
		'face_swapper_pixel_boost': '128x128',
		'face_swapper_weight': 0.5,
		'source_fusion_mode': 'weighted',
		'source_fusion_outlier_threshold': 0.42
	}
	face_swapper_core.apply_args(args, _capture)

	assert captured['source_fusion_mode'] == 'weighted'
	assert captured['source_fusion_outlier_threshold'] == 0.42


# ---------------------------------------------------------------------------
# Module sanity.


def test_module_exports_fusion_helpers() -> None:
	# The processor module must be able to import the new public helper
	# alongside the legacy entrypoint without side effects.
	assert callable(face_analyser.get_average_face)
	assert callable(face_analyser.get_fused_face)
	assert callable(face_swapper_core.extract_source_face)
