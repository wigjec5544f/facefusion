"""Tests for face_landmarker 2dfan4 / peppa_wutz dynamic batching
(Đợt 1.G2 widest, PR #18).

Pattern mirrors ``tests/test_face_landmarker_batching.py`` (PR #17):

* ``forward_with_2dfan4_batch`` and ``forward_with_peppa_wutz_batch``
  dispatch are exercised against a stub ``_FakeSession`` shaped like an
  ONNX Runtime InferenceSession; we verify they issue exactly **one**
  ``session.run`` per refinement model for dynamic-batch sessions and
  fall back to N calls for fixed-batch sessions.
* ``detect_face_landmarks_batch`` is tested by monkey-patching the two
  per-model batched helpers; we verify it (a) no-ops on empty input,
  (b) routes correctly for ``many`` / ``2dfan4`` / ``peppa_wutz``, and
  (c) preserves the score-arbitration rule from the original
  ``detect_face_landmark``.
* The single-face wrapper still exists and is bit-equal to the batch
  path with N=1.
"""
import numpy
import pytest

from facefusion import face_landmarker, state_manager


@pytest.fixture(scope = 'module', autouse = True)
def _init_state_manager() -> None:
	state_manager.init_item('download_providers', [ 'github' ])
	state_manager.init_item('face_landmarker_model', 'many')


@pytest.fixture(autouse = True)
def _reset_landmarker_model() -> None:
	# A few tests clobber this -- restore a sensible default before each.
	state_manager.set_item('face_landmarker_model', 'many')


# ---------------------------------------------------------------------------
# Stub ONNX Runtime session.


class _FakeInput:

	def __init__(self, name, shape):
		self.name = name
		self.shape = shape


class _Fake2dfan4Session:

	def __init__(self, batch_dim):
		self.batch_dim = batch_dim
		self.calls = []

	def get_inputs(self):
		return [ _FakeInput('input', [ self.batch_dim, 3, 256, 256 ]) ]

	def run(self, output_names, feed):  # noqa: ARG002
		batch_input = feed['input']
		assert batch_input.ndim == 4, f'expected 4-D input, got {batch_input.shape}'
		self.calls.append({ 'input_shape': tuple(batch_input.shape) })
		batch = batch_input.shape[0]
		# Deterministic per-row outputs so callers can verify per-face
		# distinctness through the batch path.
		fingerprint = batch_input.sum(axis = (1, 2, 3)).reshape(batch, 1, 1)
		landmarks = numpy.broadcast_to(fingerprint, (batch, 68, 3)).astype(numpy.float32)
		heatmaps = numpy.broadcast_to(fingerprint.reshape(batch, 1, 1, 1), (batch, 68, 64, 64)).astype(numpy.float32)
		return [ landmarks.copy(), heatmaps.copy() ]


class _FakePeppaSession:

	def __init__(self, batch_dim):
		self.batch_dim = batch_dim
		self.calls = []

	def get_inputs(self):
		return [ _FakeInput('input', [ self.batch_dim, 3, 256, 256 ]) ]

	def run(self, output_names, feed):  # noqa: ARG002
		batch_input = feed['input']
		assert batch_input.ndim == 4, f'expected 4-D input, got {batch_input.shape}'
		self.calls.append({ 'input_shape': tuple(batch_input.shape) })
		batch = batch_input.shape[0]
		fingerprint = batch_input.sum(axis = (1, 2, 3)).reshape(batch, 1, 1)
		predictions = numpy.broadcast_to(fingerprint, (batch, 68, 3)).astype(numpy.float32)
		return [ predictions.copy() ]


def _patch_session(monkeypatch, key, session):
	monkeypatch.setattr(face_landmarker, 'get_inference_pool', lambda: { key: session })


# ---------------------------------------------------------------------------
# forward_with_2dfan4_batch dispatch.


def test_forward_with_2dfan4_batch_uses_single_call_for_dynamic_batch(monkeypatch) -> None:
	session = _Fake2dfan4Session(batch_dim = 'batch')
	_patch_session(monkeypatch, '2dfan4', session)
	stacked = numpy.zeros((4, 3, 256, 256), dtype = numpy.float32)

	landmarks, heatmaps = face_landmarker.forward_with_2dfan4_batch(stacked)

	assert len(session.calls) == 1
	assert session.calls[0]['input_shape'] == (4, 3, 256, 256)
	assert landmarks.shape == (4, 68, 3)
	assert heatmaps.shape == (4, 68, 64, 64)


def test_forward_with_2dfan4_batch_falls_back_to_loop_for_fixed_batch(monkeypatch) -> None:
	session = _Fake2dfan4Session(batch_dim = 1)
	_patch_session(monkeypatch, '2dfan4', session)
	stacked = numpy.zeros((3, 3, 256, 256), dtype = numpy.float32)

	landmarks, heatmaps = face_landmarker.forward_with_2dfan4_batch(stacked)

	assert len(session.calls) == 3
	assert all(call['input_shape'] == (1, 3, 256, 256) for call in session.calls)
	assert landmarks.shape == (3, 68, 3)
	assert heatmaps.shape == (3, 68, 64, 64)


def test_forward_with_2dfan4_batch_dynamic_and_fixed_paths_produce_identical_output(monkeypatch) -> None:
	stacked = numpy.stack([
		numpy.full((3, 256, 256), value, dtype = numpy.float32)
		for value in (0.1, 0.2, 0.3)
	])

	dyn_session = _Fake2dfan4Session(batch_dim = 'batch')
	_patch_session(monkeypatch, '2dfan4', dyn_session)
	dyn_landmarks, dyn_heatmaps = face_landmarker.forward_with_2dfan4_batch(stacked.copy())

	fix_session = _Fake2dfan4Session(batch_dim = 1)
	_patch_session(monkeypatch, '2dfan4', fix_session)
	fix_landmarks, fix_heatmaps = face_landmarker.forward_with_2dfan4_batch(stacked.copy())

	numpy.testing.assert_array_equal(dyn_landmarks, fix_landmarks)
	numpy.testing.assert_array_equal(dyn_heatmaps, fix_heatmaps)


# ---------------------------------------------------------------------------
# forward_with_peppa_wutz_batch dispatch.


def test_forward_with_peppa_wutz_batch_uses_single_call_for_dynamic_batch(monkeypatch) -> None:
	session = _FakePeppaSession(batch_dim = 'batch')
	_patch_session(monkeypatch, 'peppa_wutz', session)
	stacked = numpy.zeros((4, 3, 256, 256), dtype = numpy.float32)

	prediction = face_landmarker.forward_with_peppa_wutz_batch(stacked)

	assert len(session.calls) == 1
	assert session.calls[0]['input_shape'] == (4, 3, 256, 256)
	assert prediction.shape == (4, 68, 3)


def test_forward_with_peppa_wutz_batch_falls_back_to_loop_for_fixed_batch(monkeypatch) -> None:
	session = _FakePeppaSession(batch_dim = 1)
	_patch_session(monkeypatch, 'peppa_wutz', session)
	stacked = numpy.zeros((3, 3, 256, 256), dtype = numpy.float32)

	prediction = face_landmarker.forward_with_peppa_wutz_batch(stacked)

	assert len(session.calls) == 3
	assert all(call['input_shape'] == (1, 3, 256, 256) for call in session.calls)
	assert prediction.shape == (3, 68, 3)


# ---------------------------------------------------------------------------
# detect_face_landmarks_batch routing & arbitration.


def _make_canned_2dfan4(monkeypatch, results):
	calls = []

	def _fake(vision_frame, bboxes, angles):
		calls.append((tuple(bboxes), tuple(angles)))
		assert len(bboxes) == len(results), f'expected {len(results)} bboxes, got {len(bboxes)}'
		return list(results)

	monkeypatch.setattr(face_landmarker, '_detect_with_2dfan4_batch', _fake)
	return calls


def _make_canned_peppa(monkeypatch, results):
	calls = []

	def _fake(vision_frame, bboxes, angles):
		calls.append((tuple(bboxes), tuple(angles)))
		assert len(bboxes) == len(results), f'expected {len(results)} bboxes, got {len(bboxes)}'
		return list(results)

	monkeypatch.setattr(face_landmarker, '_detect_with_peppa_wutz_batch', _fake)
	return calls


def _stub_landmark(value):
	return numpy.full((68, 2), value, dtype = numpy.float32)


def test_detect_face_landmarks_batch_returns_empty_for_empty_input(monkeypatch) -> None:
	# No model session should be touched at all.
	calls_2dfan4 = _make_canned_2dfan4(monkeypatch, [])
	calls_peppa = _make_canned_peppa(monkeypatch, [])

	results = face_landmarker.detect_face_landmarks_batch(numpy.zeros((1, 1, 3), dtype = numpy.uint8), [], [])

	assert results == []
	assert calls_2dfan4 == []
	assert calls_peppa == []


def test_detect_face_landmarks_batch_routes_2dfan4_only(monkeypatch) -> None:
	state_manager.set_item('face_landmarker_model', '2dfan4')
	canned_2dfan4 = [
		(_stub_landmark(1.0), 0.9),
		(_stub_landmark(2.0), 0.8)
	]
	calls_2dfan4 = _make_canned_2dfan4(monkeypatch, canned_2dfan4)
	calls_peppa = _make_canned_peppa(monkeypatch, [])

	bboxes = [ numpy.array([ 0, 0, 100, 100 ]), numpy.array([ 50, 50, 150, 150 ]) ]
	angles = [ 0, 90 ]
	results = face_landmarker.detect_face_landmarks_batch(numpy.zeros((1, 1, 3), dtype = numpy.uint8), bboxes, angles)

	assert len(calls_2dfan4) == 1
	assert calls_peppa == []  # peppa_wutz model not active -> not invoked
	assert len(results) == 2
	assert results[0][1] == pytest.approx(0.9)
	assert results[1][1] == pytest.approx(0.8)


def test_detect_face_landmarks_batch_routes_peppa_wutz_only(monkeypatch) -> None:
	state_manager.set_item('face_landmarker_model', 'peppa_wutz')
	canned_peppa = [
		(_stub_landmark(3.0), 0.7),
		(_stub_landmark(4.0), 0.6)
	]
	calls_2dfan4 = _make_canned_2dfan4(monkeypatch, [])
	calls_peppa = _make_canned_peppa(monkeypatch, canned_peppa)

	bboxes = [ numpy.array([ 0, 0, 100, 100 ]), numpy.array([ 50, 50, 150, 150 ]) ]
	angles = [ 0, 90 ]
	results = face_landmarker.detect_face_landmarks_batch(numpy.zeros((1, 1, 3), dtype = numpy.uint8), bboxes, angles)

	assert calls_2dfan4 == []  # 2dfan4 not active
	assert len(calls_peppa) == 1
	assert results[0][1] == pytest.approx(0.7)
	assert results[1][1] == pytest.approx(0.6)


def test_detect_face_landmarks_batch_runs_both_for_many(monkeypatch) -> None:
	state_manager.set_item('face_landmarker_model', 'many')
	canned_2dfan4 = [ (_stub_landmark(10.0), 0.95) ]
	canned_peppa = [ (_stub_landmark(20.0), 0.50) ]
	calls_2dfan4 = _make_canned_2dfan4(monkeypatch, canned_2dfan4)
	calls_peppa = _make_canned_peppa(monkeypatch, canned_peppa)

	bboxes = [ numpy.array([ 0, 0, 100, 100 ]) ]
	angles = [ 0 ]
	results = face_landmarker.detect_face_landmarks_batch(numpy.zeros((1, 1, 3), dtype = numpy.uint8), bboxes, angles)

	assert len(calls_2dfan4) == 1
	assert len(calls_peppa) == 1
	# 2dfan4 score 0.95 > 0.50 - 0.2 -> 2dfan4 wins.
	numpy.testing.assert_array_equal(results[0][0], _stub_landmark(10.0))
	assert results[0][1] == pytest.approx(0.95)


def test_detect_face_landmarks_batch_arbitration_picks_peppa_wutz_when_winning(monkeypatch) -> None:
	state_manager.set_item('face_landmarker_model', 'many')
	canned_2dfan4 = [ (_stub_landmark(10.0), 0.30) ]
	canned_peppa = [ (_stub_landmark(20.0), 0.80) ]
	_make_canned_2dfan4(monkeypatch, canned_2dfan4)
	_make_canned_peppa(monkeypatch, canned_peppa)

	bboxes = [ numpy.array([ 0, 0, 100, 100 ]) ]
	angles = [ 0 ]
	results = face_landmarker.detect_face_landmarks_batch(numpy.zeros((1, 1, 3), dtype = numpy.uint8), bboxes, angles)

	# 0.30 > 0.80 - 0.2 = 0.60 ? No -> peppa_wutz wins.
	numpy.testing.assert_array_equal(results[0][0], _stub_landmark(20.0))
	assert results[0][1] == pytest.approx(0.80)


def test_detect_face_landmark_single_helper_matches_batch_path(monkeypatch) -> None:
	state_manager.set_item('face_landmarker_model', '2dfan4')
	canned = [ (_stub_landmark(7.0), 0.75) ]
	_make_canned_2dfan4(monkeypatch, canned)
	_make_canned_peppa(monkeypatch, [])

	bbox = numpy.array([ 5, 6, 105, 106 ])
	angle = 12
	frame = numpy.zeros((1, 1, 3), dtype = numpy.uint8)

	# Single-face wrapper.
	single_landmark, single_score = face_landmarker.detect_face_landmark(frame, bbox, angle)

	# Batched call (re-patch since the previous call consumed the canned list).
	_make_canned_2dfan4(monkeypatch, canned)
	_make_canned_peppa(monkeypatch, [])
	[ (batch_landmark, batch_score) ] = face_landmarker.detect_face_landmarks_batch(frame, [ bbox ], [ angle ])

	numpy.testing.assert_array_equal(single_landmark, batch_landmark)
	assert single_score == pytest.approx(batch_score)
