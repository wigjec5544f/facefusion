"""Tests for facefusion.processors.batching.

These tests intentionally use plain stub session objects (not real ONNX
sessions) so they're fast and don't require any model downloads.
"""
from typing import Any, Dict, List
from unittest import mock

import numpy
import pytest

from facefusion.processors import batching


class _StubInput:

	def __init__(self, name : str, shape : List[Any]) -> None:
		self.name = name
		self.shape = shape


class _StubSession:
	"""Minimal stand-in for an onnxruntime InferenceSession."""

	def __init__(self, inputs : List[_StubInput], run_handler : Any) -> None:
		self._inputs = inputs
		self._run_handler = run_handler
		self.run_calls : List[Dict[str, numpy.ndarray]] = []

	def get_inputs(self) -> List[_StubInput]:
		return self._inputs

	def run(self, output_names : Any, feed : Dict[str, numpy.ndarray]) -> List[numpy.ndarray]:
		self.run_calls.append({ key: value.copy() for key, value in feed.items() })
		return self._run_handler(feed)


def _identity_handler(input_name : str = 'target'):
	"""Return a handler that echoes the named input back as the only output.
	Useful for proving batched/looped paths produce identical results."""

	def handler(feed : Dict[str, numpy.ndarray]) -> List[numpy.ndarray]:
		return [ feed[input_name].copy() ]
	return handler


def test_supports_dynamic_batch_with_string_dim() -> None:
	session = _StubSession([ _StubInput('target', [ 'batch', 3, 64, 64 ]) ], _identity_handler())
	assert batching.supports_dynamic_batch(session, 'target') is True


def test_supports_dynamic_batch_with_none_dim() -> None:
	session = _StubSession([ _StubInput('target', [ None, 3, 64, 64 ]) ], _identity_handler())
	assert batching.supports_dynamic_batch(session, 'target') is True


def test_supports_dynamic_batch_with_negative_or_zero_dim() -> None:
	session_neg = _StubSession([ _StubInput('target', [ -1, 3, 64, 64 ]) ], _identity_handler())
	session_zero = _StubSession([ _StubInput('target', [ 0, 3, 64, 64 ]) ], _identity_handler())
	assert batching.supports_dynamic_batch(session_neg, 'target') is True
	assert batching.supports_dynamic_batch(session_zero, 'target') is True


def test_supports_dynamic_batch_with_fixed_one() -> None:
	session = _StubSession([ _StubInput('target', [ 1, 3, 64, 64 ]) ], _identity_handler())
	assert batching.supports_dynamic_batch(session, 'target') is False


def test_supports_dynamic_batch_with_missing_input() -> None:
	session = _StubSession([ _StubInput('source', [ 1, 512 ]) ], _identity_handler())
	assert batching.supports_dynamic_batch(session, 'target') is False


def test_run_session_batched_passes_full_batch() -> None:
	session = _StubSession([ _StubInput('target', [ 'batch', 3, 4, 4 ]) ], _identity_handler())
	batched_input = numpy.arange(2 * 3 * 4 * 4, dtype = numpy.float32).reshape(2, 3, 4, 4)
	source = numpy.zeros((1, 512), dtype = numpy.float32)
	result = batching.run_session_batched(session, { 'source': source }, 'target', batched_input)

	assert len(session.run_calls) == 1
	assert numpy.array_equal(session.run_calls[0]['target'], batched_input)
	assert numpy.array_equal(session.run_calls[0]['source'], source)
	assert result.shape == batched_input.shape
	assert numpy.array_equal(result, batched_input)


def test_run_session_looped_calls_session_per_element() -> None:
	session = _StubSession([ _StubInput('target', [ 1, 3, 4, 4 ]) ], _identity_handler())
	batched_input = numpy.arange(3 * 3 * 4 * 4, dtype = numpy.float32).reshape(3, 3, 4, 4)
	result = batching.run_session_looped(session, {}, 'target', batched_input)

	assert len(session.run_calls) == 3
	for index, call in enumerate(session.run_calls):
		assert call['target'].shape == (1, 3, 4, 4)
		assert numpy.array_equal(call['target'][0], batched_input[index])
	assert result.shape == batched_input.shape
	assert numpy.array_equal(result, batched_input)


def test_run_with_dynamic_batch_uses_batched_path_when_supported() -> None:
	session = _StubSession([ _StubInput('target', [ 'batch', 3, 4, 4 ]) ], _identity_handler())
	batched_input = numpy.ones((4, 3, 4, 4), dtype = numpy.float32)
	result = batching.run_with_dynamic_batch(session, {}, 'target', batched_input)

	assert len(session.run_calls) == 1, 'should batch into a single session.run call'
	assert numpy.array_equal(result, batched_input)


def test_run_with_dynamic_batch_falls_back_when_fixed_batch_one() -> None:
	session = _StubSession([ _StubInput('target', [ 1, 3, 4, 4 ]) ], _identity_handler())
	batched_input = numpy.ones((4, 3, 4, 4), dtype = numpy.float32)
	result = batching.run_with_dynamic_batch(session, {}, 'target', batched_input)

	assert len(session.run_calls) == 4, 'should fall back to per-tile calls'
	assert numpy.array_equal(result, batched_input)


def test_run_with_dynamic_batch_falls_back_on_runtime_failure() -> None:
	"""When supports_dynamic_batch reports True but the runtime rejects the
	batched call (e.g. exporter mis-tagged), we should still produce output
	by re-running per element and notify the caller via on_fallback."""
	rejecting_handler_calls = { 'count': 0 }

	def rejecting_handler(feed : Dict[str, numpy.ndarray]) -> List[numpy.ndarray]:
		rejecting_handler_calls['count'] += 1
		if feed['target'].shape[0] != 1:
			raise RuntimeError('fixed batch=1 enforced by runtime')
		return [ feed['target'].copy() ]

	session = _StubSession([ _StubInput('target', [ 'batch', 3, 4, 4 ]) ], rejecting_handler)
	batched_input = numpy.ones((3, 3, 4, 4), dtype = numpy.float32)
	captured : List[Exception] = []
	result = batching.run_with_dynamic_batch(
		session, {}, 'target', batched_input, on_fallback = captured.append
	)

	assert len(captured) == 1
	assert isinstance(captured[0], RuntimeError)
	# 1 failed batched call + 3 successful per-tile calls.
	assert rejecting_handler_calls['count'] == 4
	assert numpy.array_equal(result, batched_input)


def test_stack_prepared_frames_concatenates_along_batch_axis() -> None:
	tiles = [
		numpy.zeros((1, 3, 4, 4), dtype = numpy.float32),
		numpy.ones((1, 3, 4, 4), dtype = numpy.float32),
		numpy.full((1, 3, 4, 4), 2.0, dtype = numpy.float32)
	]
	stacked = batching.stack_prepared_frames(tiles)
	assert stacked.shape == (3, 3, 4, 4)
	assert numpy.array_equal(stacked[0], tiles[0][0])
	assert numpy.array_equal(stacked[2], tiles[2][0])


def test_stack_prepared_frames_rejects_empty_input() -> None:
	with pytest.raises(ValueError):
		batching.stack_prepared_frames([])


def test_batched_and_looped_paths_produce_equivalent_outputs() -> None:
	"""The whole point of the helper: callers can switch between paths with
	no observable change. We use a non-trivial handler (sum across spatial
	axes per channel, then broadcast) to avoid trivial pass-through."""
	def handler(feed : Dict[str, numpy.ndarray]) -> List[numpy.ndarray]:
		target = feed['target']
		# Output shape == input shape, value = per-sample-per-channel mean.
		means = target.mean(axis = (2, 3), keepdims = True)
		return [ numpy.broadcast_to(means, target.shape).copy() ]

	dynamic_session = _StubSession([ _StubInput('target', [ 'batch', 3, 4, 4 ]) ], handler)
	fixed_session = _StubSession([ _StubInput('target', [ 1, 3, 4, 4 ]) ], handler)

	batched_input = numpy.random.RandomState(0).rand(5, 3, 4, 4).astype(numpy.float32)

	dynamic_result = batching.run_with_dynamic_batch(dynamic_session, {}, 'target', batched_input)
	fixed_result = batching.run_with_dynamic_batch(fixed_session, {}, 'target', batched_input)

	assert len(dynamic_session.run_calls) == 1
	assert len(fixed_session.run_calls) == 5
	numpy.testing.assert_allclose(dynamic_result, fixed_result)


def test_face_swapper_batched_path_invokes_helper(monkeypatch : pytest.MonkeyPatch) -> None:
	"""Smoke test that face_swapper.swap_face wires through the batched
	helper rather than calling forward_swap_face N times. We don't run the
	real model -- we just confirm the helper sees the stacked tiles."""
	from facefusion.processors.modules.face_swapper import core as face_swapper_core
	captured : Dict[str, Any] = {}

	def fake_forward_batch(source_face : Any, target_face : Any, prepared : List[numpy.ndarray]) -> numpy.ndarray:
		captured['count'] = len(prepared)
		captured['shape'] = prepared[0].shape
		# Return a dummy (N, 3, 8, 8) batch -- swap_face only cares about
		# the leading axis count; downstream normalize_crop_frame is mocked.
		return numpy.zeros((len(prepared), 3, 8, 8), dtype = numpy.float32)

	monkeypatch.setattr(face_swapper_core, 'forward_swap_face_batch', fake_forward_batch)
	monkeypatch.setattr(face_swapper_core, 'normalize_crop_frame', lambda frame: numpy.zeros((8, 8, 3), dtype = numpy.uint8))
	monkeypatch.setattr(face_swapper_core, 'prepare_crop_frame', lambda frame: numpy.zeros((1, 3, 8, 8), dtype = numpy.float32))
	monkeypatch.setattr(face_swapper_core, 'warp_face_by_face_landmark_5', lambda *args, **kwargs: (numpy.zeros((16, 16, 3), dtype = numpy.uint8), numpy.eye(2, 3, dtype = numpy.float32)))
	monkeypatch.setattr(face_swapper_core, 'paste_back', lambda temp, crop, mask, matrix: temp)
	monkeypatch.setattr(face_swapper_core, 'implode_pixel_boost', lambda crop, total, size: [ numpy.zeros((8, 8, 3), dtype = numpy.uint8) ] * (total * total))
	monkeypatch.setattr(face_swapper_core, 'explode_pixel_boost', lambda frames, total, model_size, boost_size: numpy.zeros((boost_size[0], boost_size[1], 3), dtype = numpy.uint8))
	monkeypatch.setattr(face_swapper_core, 'unpack_resolution', lambda value: (16, 16))
	monkeypatch.setattr(face_swapper_core, 'create_box_mask', lambda *args, **kwargs: numpy.ones((16, 16), dtype = numpy.float32))
	monkeypatch.setattr(face_swapper_core, 'get_model_options', lambda: { 'template': 'arcface_128', 'size': (8, 8), 'type': 'inswapper' })

	with mock.patch.object(face_swapper_core, 'state_manager') as state:
		state.get_item.side_effect = lambda key: {
			'face_swapper_pixel_boost': '16x16',
			'face_mask_blur': 0.3,
			'face_mask_padding': (0, 0, 0, 0),
			'face_mask_types': [ 'box' ]
		}.get(key)

		fake_source = mock.MagicMock()
		fake_target = mock.MagicMock()
		fake_target.landmark_set = { '5/68': numpy.zeros((5, 2), dtype = numpy.float32) }
		fake_target.embedding = numpy.zeros((512,), dtype = numpy.float32)
		face_swapper_core.swap_face(fake_source, fake_target, numpy.zeros((32, 32, 3), dtype = numpy.uint8))

	# 16x16 boost / 8x8 model = 2x2 = 4 tiles
	assert captured['count'] == 4
	assert captured['shape'] == (1, 3, 8, 8)
