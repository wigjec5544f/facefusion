"""Tests for face_recognizer dynamic batching (Đợt 1.G2 widest, PR #16).

Pattern mirrors ``tests/test_face_enhancer_batching.py``:

* `forward_batch` dispatch is exercised against a stub `_FakeSession` shaped
  like an ONNX Runtime InferenceSession; we verify it issues exactly one
  ``session.run`` for dynamic-batch models and falls back to N calls for
  fixed-batch models.
* `calculate_face_embeddings` plumbing is tested by patching the warp helper
  and the recogniser pool; we verify that the new batch entrypoint preserves
  bit-equality with the historical single-face ``calculate_face_embedding``.
* Empty input is a no-op (returns an empty list, never touches the ONNX
  session).
* The single-face helper still exists and round-trips through the batch path.
"""
import numpy
import pytest

from facefusion import face_recognizer, state_manager


@pytest.fixture(scope = 'module', autouse = True)
def _init_state_manager() -> None:
	# `create_static_model_set` resolves download URLs at import time and
	# reads `download_providers` from the global state manager. Tests in
	# this module never actually download anything, but we still need the
	# state manager populated so the resolver doesn't blow up while the
	# model URLs are being constructed.
	state_manager.init_item('download_providers', [ 'github' ])


# ---------------------------------------------------------------------------
# Stub ONNX Runtime session.


class _FakeInput:

	def __init__(self, name, shape):
		self.name = name
		self.shape = shape


class _FakeSession:

	def __init__(self, batch_dim, embedding_dim = 512, output_handler = None):
		self.batch_dim = batch_dim
		self.embedding_dim = embedding_dim
		self.output_handler = output_handler
		self.calls = []

	def get_inputs(self):
		return [ _FakeInput('input', [ self.batch_dim, 3, 112, 112 ]) ]

	def run(self, output_names, feed):  # noqa: ARG002
		batch_input = feed['input']
		self.calls.append({ 'input_shape': batch_input.shape })
		if self.output_handler is not None:
			return [ self.output_handler(batch_input) ]
		# Default: return a deterministic fingerprint of each input row so
		# we can assert that batched and looped outputs are identical.
		batch = batch_input.shape[0]
		fingerprint = batch_input.reshape(batch, -1).sum(axis = 1, keepdims = True)
		return [ numpy.tile(fingerprint, (1, self.embedding_dim)).astype(numpy.float32) ]


def _patch_session(monkeypatch, session):
	monkeypatch.setattr(face_recognizer, 'get_inference_pool', lambda: { 'face_recognizer': session })


# ---------------------------------------------------------------------------
# forward_batch dispatch.


def test_forward_batch_uses_single_call_for_dynamic_batch(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session)
	stacked = numpy.zeros((3, 3, 112, 112), dtype = numpy.float32)

	output = face_recognizer.forward_batch(stacked)

	assert len(session.calls) == 1
	assert session.calls[0]['input_shape'] == (3, 3, 112, 112)
	assert output.shape == (3, 512)


def test_forward_batch_falls_back_to_loop_for_fixed_batch(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 1)
	_patch_session(monkeypatch, session)
	stacked = numpy.zeros((3, 3, 112, 112), dtype = numpy.float32)

	output = face_recognizer.forward_batch(stacked)

	# 3 separate session.run calls each shaped (1, 3, 112, 112).
	assert len(session.calls) == 3
	assert all(call['input_shape'] == (1, 3, 112, 112) for call in session.calls)
	assert output.shape == (3, 512)


def test_forward_batch_dynamic_and_fixed_paths_produce_identical_output(monkeypatch) -> None:
	# Different per-row pixel values so the fingerprint output is unique
	# per row, then assert the dynamic-batch path and the fixed-batch
	# fallback produce byte-identical results.
	stacked = numpy.stack([
		numpy.full((3, 112, 112), value, dtype = numpy.float32)
		for value in (0.1, 0.5, 0.9)
	])

	session_dynamic = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session_dynamic)
	dynamic_output = face_recognizer.forward_batch(stacked.copy())

	session_fixed = _FakeSession(batch_dim = 1)
	_patch_session(monkeypatch, session_fixed)
	fixed_output = face_recognizer.forward_batch(stacked.copy())

	numpy.testing.assert_array_equal(dynamic_output, fixed_output)


# ---------------------------------------------------------------------------
# calculate_face_embeddings plumbing.


def _patch_warp(monkeypatch):
	# warp_face_by_face_landmark_5 takes (frame, landmark_5, template,
	# size) -> (warped_crop, affine_matrix). Stub it to return a synthetic
	# crop whose pixel values fingerprint the landmark, so we can assert
	# the per-face crops feed the recogniser in the correct order.
	def _stub_warp(temp_vision_frame, face_landmark_5, model_template, model_size):  # noqa: ARG001
		marker = float(face_landmark_5[0][0])
		warped = numpy.full(model_size + (3,), marker, dtype = numpy.float32)
		affine = numpy.eye(2, 3, dtype = numpy.float32)
		return warped, affine

	monkeypatch.setattr(face_recognizer, 'warp_face_by_face_landmark_5', _stub_warp)


def test_calculate_face_embeddings_returns_empty_for_empty_input(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session)
	_patch_warp(monkeypatch)

	result = face_recognizer.calculate_face_embeddings(numpy.zeros((512, 512, 3), dtype = numpy.uint8), [])

	assert result == []
	# Crucially the recogniser must NOT be called for an empty face list.
	assert session.calls == []


def test_calculate_face_embeddings_preserves_input_order(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session)
	_patch_warp(monkeypatch)

	# Three landmarks whose first-point x is 100, 200, 300 -- the warp
	# stub turns those into crops filled with 100/127.5-1, 200/127.5-1,
	# 300/127.5-1 respectively. The fake session's fingerprint output is a
	# per-row sum, so each face gets a distinct embedding vector.
	landmarks = [
		numpy.array([[ value, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ]], dtype = numpy.float32)
		for value in (100.0, 200.0, 300.0)
	]

	embeddings = face_recognizer.calculate_face_embeddings(numpy.zeros((512, 512, 3), dtype = numpy.uint8), landmarks)

	assert len(embeddings) == 3
	# Single batched session.run shaped (3, 3, 112, 112).
	assert len(session.calls) == 1
	assert session.calls[0]['input_shape'] == (3, 3, 112, 112)

	# Ordering: face 0's embedding fingerprint < face 1's < face 2's
	# (because crop fill values are monotonically increasing).
	embedding_sums = [ float(emb[0].sum()) for emb in embeddings ]
	assert embedding_sums[0] < embedding_sums[1] < embedding_sums[2]

	# `embedding_norm` is unit-length.
	for _, embedding_norm in embeddings:
		numpy.testing.assert_allclose(numpy.linalg.norm(embedding_norm), 1.0, rtol = 1e-5)


def test_calculate_face_embedding_single_helper_matches_batch_path(monkeypatch) -> None:
	# The legacy single-face helper now delegates to the batch entrypoint
	# with N=1; assert the bytes are identical to a fresh batch call.
	session_one = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session_one)
	_patch_warp(monkeypatch)

	frame = numpy.zeros((512, 512, 3), dtype = numpy.uint8)
	landmark = numpy.array([[ 175.0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ]], dtype = numpy.float32)
	single_emb, single_norm = face_recognizer.calculate_face_embedding(frame, landmark)

	session_two = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session_two)
	_patch_warp(monkeypatch)
	[ (batch_emb, batch_norm) ] = face_recognizer.calculate_face_embeddings(frame, [ landmark ])

	numpy.testing.assert_array_equal(single_emb, batch_emb)
	numpy.testing.assert_array_equal(single_norm, batch_norm)


def test_calculate_face_embeddings_falls_back_to_loop_for_fixed_batch(monkeypatch) -> None:
	# Same input through dynamic-batch path vs fixed-batch loop must
	# produce bit-identical embedding lists.
	_patch_warp(monkeypatch)
	frame = numpy.zeros((512, 512, 3), dtype = numpy.uint8)
	landmarks = [
		numpy.array([[ value, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ]], dtype = numpy.float32)
		for value in (50.0, 150.0)
	]

	session_dynamic = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session_dynamic)
	dyn_results = face_recognizer.calculate_face_embeddings(frame, landmarks)

	session_fixed = _FakeSession(batch_dim = 1)
	_patch_session(monkeypatch, session_fixed)
	fix_results = face_recognizer.calculate_face_embeddings(frame, landmarks)

	assert len(dyn_results) == len(fix_results) == 2
	for (d_emb, d_norm), (f_emb, f_norm) in zip(dyn_results, fix_results):
		numpy.testing.assert_array_equal(d_emb, f_emb)
		numpy.testing.assert_array_equal(d_norm, f_norm)
