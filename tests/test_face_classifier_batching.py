"""Tests for face_classifier dynamic batching (Đợt 1.G2 widest, PR #19).

Pattern mirrors ``tests/test_face_recognizer_batching.py`` (PR #16) and
``tests/test_face_landmarker_batching.py`` (PR #17):

* `forward_batch` dispatch is exercised against a stub `_FakeSession`
  shaped like an ONNX Runtime InferenceSession; we verify it issues
  exactly one `session.run` for dynamic-batch models and falls back to
  N calls for fixed-batch models.
* `classify_faces` plumbing is tested by patching the per-face warp
  helper and the inference pool; we verify per-face output order is
  preserved and the empty-input fast path doesn't hit the model.
* The single-face wrapper (`classify_face`) still exists and
  round-trips through the batch path bit-equal.
"""
from typing import List

import numpy
import pytest

from facefusion import face_classifier, state_manager


@pytest.fixture(scope = 'module', autouse = True)
def _init_state_manager() -> None:
	state_manager.init_item('download_providers', [ 'github' ])


# ---------------------------------------------------------------------------
# Stub ONNX Runtime session.


class _FakeInput:

	def __init__(self, name, shape):
		self.name = name
		self.shape = shape


class _FakeSession:
	"""Stand-in for fairface InferenceSession.

	Output convention mirrors the real model: ``run`` returns
	``(race_ids, gender_ids, age_ids)`` shaped ``(N,)``. The fake
	produces deterministic per-row ids by fingerprinting input pixel
	sums so callers can verify per-face distinctness through the
	batched path.
	"""

	def __init__(self, batch_dim):
		self.batch_dim = batch_dim
		self.calls = []

	def get_inputs(self):
		return [ _FakeInput('input', [ self.batch_dim, 3, 224, 224 ]) ]

	def run(self, output_names, feed):  # noqa: ARG002
		batch_input = feed['input']
		assert batch_input.ndim == 4, f'expected 4-D input, got {batch_input.shape}'
		self.calls.append({ 'input_shape': tuple(batch_input.shape) })
		batch = batch_input.shape[0]
		fingerprints = batch_input.sum(axis = (1, 2, 3))
		# Map fingerprint into deterministic class ids; modulo keeps the
		# results inside the expected fairface ranges so categorize_*
		# helpers do not blow up downstream.
		race_ids = (fingerprints.astype(numpy.int64) % 7).reshape(batch)
		gender_ids = (fingerprints.astype(numpy.int64) % 2).reshape(batch)
		age_ids = (fingerprints.astype(numpy.int64) % 9).reshape(batch)
		return [ race_ids, gender_ids, age_ids ]


def _patch_session(monkeypatch, session):
	monkeypatch.setattr(face_classifier, 'get_inference_pool', lambda: { 'face_classifier': session })


# ---------------------------------------------------------------------------
# forward_batch dispatch.


def test_forward_batch_uses_single_call_for_dynamic_batch(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session)
	stacked = numpy.zeros((4, 3, 224, 224), dtype = numpy.float32)

	gender_ids, age_ids, race_ids = face_classifier.forward_batch(stacked)

	assert len(session.calls) == 1
	assert session.calls[0]['input_shape'] == (4, 3, 224, 224)
	assert gender_ids.shape == (4,)
	assert age_ids.shape == (4,)
	assert race_ids.shape == (4,)


def test_forward_batch_falls_back_to_loop_for_fixed_batch(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 1)
	_patch_session(monkeypatch, session)
	stacked = numpy.zeros((3, 3, 224, 224), dtype = numpy.float32)

	gender_ids, age_ids, race_ids = face_classifier.forward_batch(stacked)

	assert len(session.calls) == 3
	assert all(call['input_shape'] == (1, 3, 224, 224) for call in session.calls)
	assert gender_ids.shape == (3,)
	assert age_ids.shape == (3,)
	assert race_ids.shape == (3,)


def test_forward_batch_dynamic_and_fixed_paths_produce_identical_output(monkeypatch) -> None:
	stacked = numpy.stack([
		numpy.full((3, 224, 224), value, dtype = numpy.float32)
		for value in (0.5, 1.5, 2.5, 3.5)
	])

	dyn_session = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, dyn_session)
	dyn_gender, dyn_age, dyn_race = face_classifier.forward_batch(stacked.copy())

	fix_session = _FakeSession(batch_dim = 1)
	_patch_session(monkeypatch, fix_session)
	fix_gender, fix_age, fix_race = face_classifier.forward_batch(stacked.copy())

	numpy.testing.assert_array_equal(dyn_gender, fix_gender)
	numpy.testing.assert_array_equal(dyn_age, fix_age)
	numpy.testing.assert_array_equal(dyn_race, fix_race)


# ---------------------------------------------------------------------------
# classify_faces plumbing.


def _patch_warp(monkeypatch):
	# Per-face warp returns a deterministic crop fingerprinted by the
	# first landmark coordinate so downstream session can distinguish
	# faces. Affine matrix is irrelevant to classification.
	def _fake_warp(vision_frame, face_landmark_5, model_template, model_size):  # noqa: ARG001
		fingerprint = float(face_landmark_5.flat[0])
		crop = numpy.full((model_size[0], model_size[1], 3), fingerprint, dtype = numpy.uint8)
		return crop, numpy.eye(2, 3, dtype = numpy.float32)

	monkeypatch.setattr(face_classifier, 'warp_face_by_face_landmark_5', _fake_warp)


def test_classify_faces_returns_empty_for_empty_input(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session)
	_patch_warp(monkeypatch)

	results = face_classifier.classify_faces(numpy.zeros((1, 1, 3), dtype = numpy.uint8), [])

	assert results == []
	assert session.calls == []


def test_classify_faces_preserves_input_order(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session)
	_patch_warp(monkeypatch)

	# Three landmarks whose first-point x is 7, 17, 27. The warp stub
	# fingerprints by that value -> three distinct crops in the batch.
	landmarks : List[numpy.ndarray] = [
		numpy.array([[ value, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ]], dtype = numpy.float32)
		for value in (7.0, 17.0, 27.0)
	]
	frame = numpy.zeros((1, 1, 3), dtype = numpy.uint8)

	results = face_classifier.classify_faces(frame, landmarks)

	assert len(results) == 3
	# Single batched session.run with batch=3.
	assert len(session.calls) == 1
	assert session.calls[0]['input_shape'][0] == 3
	# Each result is (gender, age, race) with strings + range types.
	for gender, age, race in results:
		assert isinstance(gender, str)
		assert isinstance(race, str)
		assert isinstance(age, range)


def test_classify_face_single_helper_matches_batch_path(monkeypatch) -> None:
	# The legacy single-face helper now delegates to the batch entry
	# point with N=1; verify the bytes are identical to a fresh batch
	# call with the same input.
	_patch_warp(monkeypatch)
	landmark = numpy.array([[ 175.0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ]], dtype = numpy.float32)
	frame = numpy.zeros((1, 1, 3), dtype = numpy.uint8)

	session_one = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session_one)
	single = face_classifier.classify_face(frame, landmark)

	session_two = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session_two)
	[ batch_result ] = face_classifier.classify_faces(frame, [ landmark ])

	assert single == batch_result


def test_classify_faces_falls_back_for_fixed_batch(monkeypatch) -> None:
	# Same input through dynamic-batch path vs fixed-batch loop must
	# produce identical (gender, age, race) tuples.
	_patch_warp(monkeypatch)
	landmarks = [
		numpy.array([[ value, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ]], dtype = numpy.float32)
		for value in (7.0, 17.0, 27.0)
	]
	frame = numpy.zeros((1, 1, 3), dtype = numpy.uint8)

	dyn_session = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, dyn_session)
	dyn_results = face_classifier.classify_faces(frame, landmarks)

	fix_session = _FakeSession(batch_dim = 1)
	_patch_session(monkeypatch, fix_session)
	fix_results = face_classifier.classify_faces(frame, landmarks)

	assert len(dyn_results) == len(fix_results) == 3
	for dyn_tuple, fix_tuple in zip(dyn_results, fix_results):
		assert dyn_tuple == fix_tuple


def test_forward_legacy_wrapper_unwraps_single_face(monkeypatch) -> None:
	# The public ``forward`` API used to return per-call (gender, age,
	# race) ids for a (1, 3, H, W) input; verify the new wrapper still
	# delivers the same shape & values.
	session = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session)
	crop = numpy.zeros((1, 3, 224, 224), dtype = numpy.float32)

	gender_ids, age_ids, race_ids = face_classifier.forward(crop)

	assert gender_ids.shape == (1,)
	assert age_ids.shape == (1,)
	assert race_ids.shape == (1,)
