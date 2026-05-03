"""Tests for face_landmarker fan_68_5 dynamic batching (Đợt 1.G2 widest, PR #17).

Pattern mirrors ``tests/test_face_recognizer_batching.py``:

* `forward_fan_68_5_batch` dispatch is exercised against a stub
  `_FakeSession` shaped like an ONNX Runtime InferenceSession; we verify
  it issues exactly one ``session.run`` for dynamic-batch models and
  falls back to N calls for fixed-batch models.
* `estimate_face_landmark_68_5_batch` plumbing is tested by patching the
  affine helpers and recogniser pool; we verify that the batched
  entrypoint preserves the same per-face output as the historical
  ``estimate_face_landmark_68_5``.
* Empty input is a no-op.
* The single-face helper still exists and round-trips through the batch
  path bit-equal.
"""
import numpy
import pytest

from facefusion import face_landmarker, state_manager


@pytest.fixture(scope = 'module', autouse = True)
def _init_state_manager() -> None:
	# `create_static_model_set` resolves download URLs at import time and
	# reads `download_providers` from the global state manager. Tests in
	# this module never actually download anything, but the resolver
	# still needs the state manager populated so URL construction
	# doesn't blow up.
	state_manager.init_item('download_providers', [ 'github' ])
	state_manager.init_item('face_landmarker_model', 'fan_68_5')


# ---------------------------------------------------------------------------
# Stub ONNX Runtime session.


class _FakeInput:

	def __init__(self, name, shape):
		self.name = name
		self.shape = shape


class _FakeSession:

	def __init__(self, batch_dim, output_handler = None):
		self.batch_dim = batch_dim
		self.output_handler = output_handler
		self.calls = []

	def get_inputs(self):
		return [ _FakeInput('input', [ self.batch_dim, 5, 2 ]) ]

	def run(self, output_names, feed):  # noqa: ARG002
		batch_input = feed['input']
		# `forward_fan_68_5_batch` always reshapes the inputs into a 3-D
		# ``(N, 5, 2)`` array before calling session.run; ``run_session_looped``
		# slices that batch with ``arr[i:i+1]`` so each per-call frame is
		# also 3-D. Any other rank means the caller is feeding garbage --
		# fail loudly so the tests catch shape regressions.
		assert batch_input.ndim == 3, f'expected 3-D batch input, got shape {batch_input.shape}'
		self.calls.append({ 'input_shape': tuple(batch_input.shape) })
		if self.output_handler is not None:
			return [ self.output_handler(batch_input) ]
		# Default: deterministic landmark prediction = (input * 2) repeated
		# across 68 landmarks -- enough variation so we can verify
		# per-row identity through the batched path.
		batch = batch_input.shape[0]
		base = batch_input.sum(axis = (1, 2)).reshape(batch, 1, 1)
		landmarks = numpy.tile(base, (1, 68, 2)).astype(numpy.float32)
		return [ landmarks ]


def _patch_session(monkeypatch, session):
	monkeypatch.setattr(face_landmarker, 'get_inference_pool', lambda: { 'fan_68_5': session })


# ---------------------------------------------------------------------------
# forward_fan_68_5_batch dispatch.


def test_forward_fan_68_5_batch_uses_single_call_for_dynamic_batch(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session)
	stacked = numpy.zeros((4, 5, 2), dtype = numpy.float32)

	output = face_landmarker.forward_fan_68_5_batch(stacked)

	assert len(session.calls) == 1
	assert session.calls[0]['input_shape'] == (4, 5, 2)
	assert output.shape == (4, 68, 2)


def test_forward_fan_68_5_batch_falls_back_to_loop_for_fixed_batch(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 1)
	_patch_session(monkeypatch, session)
	stacked = numpy.zeros((3, 5, 2), dtype = numpy.float32)

	output = face_landmarker.forward_fan_68_5_batch(stacked)

	assert len(session.calls) == 3
	assert all(call['input_shape'] == (1, 5, 2) for call in session.calls)
	assert output.shape == (3, 68, 2)


def test_forward_fan_68_5_batch_dynamic_and_fixed_paths_produce_identical_output(monkeypatch) -> None:
	stacked = numpy.stack([
		numpy.full((5, 2), value, dtype = numpy.float32)
		for value in (1.0, 2.0, 3.0, 4.0)
	])

	session_dynamic = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session_dynamic)
	dynamic_output = face_landmarker.forward_fan_68_5_batch(stacked.copy())

	session_fixed = _FakeSession(batch_dim = 1)
	_patch_session(monkeypatch, session_fixed)
	fixed_output = face_landmarker.forward_fan_68_5_batch(stacked.copy())

	numpy.testing.assert_array_equal(dynamic_output, fixed_output)


# ---------------------------------------------------------------------------
# estimate_face_landmark_68_5_batch plumbing.


def _patch_affine_helpers(monkeypatch):
	# Stub the affine helpers so the test doesn't depend on cv2's
	# arc-template numerical kernel. We pick an identity matrix so the
	# batched warp / inverse-warp passthrough is observable.
	monkeypatch.setattr(
		face_landmarker,
		'estimate_matrix_by_face_landmark_5',
		lambda face_landmark_5, template, size: numpy.eye(2, 3, dtype = numpy.float32)  # noqa: ARG005
	)


def test_estimate_face_landmark_68_5_batch_returns_empty_for_empty_input(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session)
	_patch_affine_helpers(monkeypatch)

	result = face_landmarker.estimate_face_landmark_68_5_batch([])

	assert result == []
	# Crucially the recogniser must NOT be called for an empty list.
	assert session.calls == []


def test_estimate_face_landmark_68_5_batch_preserves_input_order(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session)
	_patch_affine_helpers(monkeypatch)

	# Three landmarks whose first-point x is 100, 200, 300; with the
	# identity affine each landmark warps to itself. The fake session's
	# fingerprint sums all (5, 2) values per row, so each face gets a
	# distinct prediction value.
	landmarks = [
		numpy.array([[ value, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ]], dtype = numpy.float32)
		for value in (100.0, 200.0, 300.0)
	]

	predictions = face_landmarker.estimate_face_landmark_68_5_batch(landmarks)

	assert len(predictions) == 3
	# Single batched session.run shaped (3, 5, 2).
	assert len(session.calls) == 1
	assert session.calls[0]['input_shape'] == (3, 5, 2)

	# Per-row distinctness preserved through the inverse-affine
	# transform (identity, so values are unchanged).
	prediction_sums = [ float(prediction.sum()) for prediction in predictions ]
	assert prediction_sums[0] < prediction_sums[1] < prediction_sums[2]
	# Each prediction has shape (68, 2).
	for prediction in predictions:
		assert prediction.shape == (68, 2)


def test_estimate_face_landmark_68_5_single_helper_matches_batch_path(monkeypatch) -> None:
	# The legacy single-face helper now delegates to the batch
	# entrypoint with N=1; verify the bytes are identical to a fresh
	# batch call with the same input.
	session_one = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session_one)
	_patch_affine_helpers(monkeypatch)

	landmark = numpy.array([[ 175.0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ]], dtype = numpy.float32)
	single_pred = face_landmarker.estimate_face_landmark_68_5(landmark)

	session_two = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session_two)
	_patch_affine_helpers(monkeypatch)
	[ batch_pred ] = face_landmarker.estimate_face_landmark_68_5_batch([ landmark ])

	numpy.testing.assert_array_equal(single_pred, batch_pred)


def test_estimate_face_landmark_68_5_batch_falls_back_for_fixed_batch(monkeypatch) -> None:
	# Same input through dynamic-batch path vs fixed-batch loop must
	# produce bit-identical outputs.
	_patch_affine_helpers(monkeypatch)
	landmarks = [
		numpy.array([[ value, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ]], dtype = numpy.float32)
		for value in (50.0, 150.0)
	]

	session_dynamic = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session_dynamic)
	dyn_results = face_landmarker.estimate_face_landmark_68_5_batch(landmarks)

	session_fixed = _FakeSession(batch_dim = 1)
	_patch_session(monkeypatch, session_fixed)
	fix_results = face_landmarker.estimate_face_landmark_68_5_batch(landmarks)

	assert len(dyn_results) == len(fix_results) == 2
	for dyn_pred, fix_pred in zip(dyn_results, fix_results):
		numpy.testing.assert_array_equal(dyn_pred, fix_pred)
