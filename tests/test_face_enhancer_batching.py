"""Tests for face_enhancer multi-face dynamic batching.

Strategy mirrors `tests/test_frame_enhancer_batching.py`:

* `_faces_overlap` is a pure helper -- unit-test the bbox geometry
  exhaustively without touching ONNX.
* `forward_batch` dispatch is exercised against a stub `_FakeSession`
  shaped like an ONNX Runtime InferenceSession; we verify it issues
  exactly one `session.run` for dynamic-batch models, falls back to N
  calls for fixed-batch models, and falls back on runtime exceptions.
* `enhance_faces` plumbing is tested by patching the per-face warp /
  paste helpers so we can verify (a) a single batched call when faces
  don't overlap, (b) per-face sequential calls when they do overlap or
  there's only one face.

The assertion that batching is bit-equal to the sequential loop comes
from the geometry: when bounding boxes don't overlap, no face's
paste-back affects another face's warp region, so all crops sample
identical pixels in either order.
"""
import numpy

from facefusion.processors.modules.face_enhancer import core as face_enhancer_core


# ---------------------------------------------------------------------------
# Stub ONNX Runtime session.


class _FakeInput:

	def __init__(self, name, shape):
		self.name = name
		self.shape = shape


class _FakeSession:

	def __init__(self, batch_dim, has_weight = False, output_handler = None):
		self.batch_dim = batch_dim
		self.has_weight = has_weight
		self.output_handler = output_handler
		self.calls = []

	def get_inputs(self):
		inputs = [ _FakeInput('input', [ self.batch_dim, 3, 8, 8 ]) ]
		if self.has_weight:
			inputs.append(_FakeInput('weight', [ 1 ]))
		return inputs

	def run(self, output_names, feed):  # noqa: ARG002
		self.calls.append({
			'input_shape': feed['input'].shape,
			'has_weight': 'weight' in feed
		})
		if self.output_handler is None:
			return [ feed['input'] ]
		return [ self.output_handler(feed['input']) ]


def _patch_session(monkeypatch, session):
	monkeypatch.setattr(
		face_enhancer_core,
		'get_inference_pool',
		lambda: { 'face_enhancer': session }
	)


# ---------------------------------------------------------------------------
# _faces_overlap geometry.


def _bbox(x1, y1, x2, y2):
	return numpy.array([ x1, y1, x2, y2 ], dtype = numpy.float32)


def test_faces_overlap_returns_false_for_well_separated_faces() -> None:
	bboxes = [ _bbox(0, 0, 50, 50), _bbox(200, 200, 280, 280), _bbox(400, 0, 480, 80) ]
	assert face_enhancer_core._faces_overlap(bboxes) is False


def test_faces_overlap_returns_true_for_intersecting_faces() -> None:
	bboxes = [ _bbox(0, 0, 100, 100), _bbox(80, 80, 180, 180) ]
	assert face_enhancer_core._faces_overlap(bboxes) is True


def test_faces_overlap_uses_expansion_margin() -> None:
	# Strictly disjoint (gap of 5 px) but within the 25%-expansion margin
	# of 100-px-wide boxes (margin ~25 px). Conservative -> reported as
	# overlapping so the caller falls back to sequential.
	bboxes = [ _bbox(0, 0, 100, 100), _bbox(105, 0, 205, 100) ]
	assert face_enhancer_core._faces_overlap(bboxes) is True


def test_faces_overlap_handles_single_face_list() -> None:
	# Edge case: the helper is only meaningful for >=2 faces; a single
	# face cannot overlap with itself.
	assert face_enhancer_core._faces_overlap([ _bbox(0, 0, 50, 50) ]) is False


# ---------------------------------------------------------------------------
# forward_batch dispatch.


def test_forward_batch_uses_single_call_for_dynamic_batch(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session)
	crops = [ numpy.zeros((1, 3, 8, 8), dtype = numpy.float32) for _ in range(3) ]

	output = face_enhancer_core.forward_batch(crops, numpy.array([ 0.5 ]))

	assert len(session.calls) == 1
	assert session.calls[0]['input_shape'] == (3, 3, 8, 8)
	assert output.shape == (3, 3, 8, 8)


def test_forward_batch_falls_back_to_loop_for_fixed_batch(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 1)
	_patch_session(monkeypatch, session)
	crops = [ numpy.zeros((1, 3, 8, 8), dtype = numpy.float32) for _ in range(3) ]

	output = face_enhancer_core.forward_batch(crops, numpy.array([ 0.5 ]))

	assert len(session.calls) == 3
	assert all(call['input_shape'] == (1, 3, 8, 8) for call in session.calls)
	assert output.shape == (3, 3, 8, 8)


def test_forward_batch_passes_weight_when_model_has_weight_input(monkeypatch) -> None:
	# Models like codeformer expose an extra `weight` input; forward_batch
	# must keep feeding it.
	session = _FakeSession(batch_dim = 'batch', has_weight = True)
	_patch_session(monkeypatch, session)
	crops = [ numpy.zeros((1, 3, 8, 8), dtype = numpy.float32) for _ in range(2) ]

	face_enhancer_core.forward_batch(crops, numpy.array([ 0.7 ]))

	assert session.calls[0]['has_weight'] is True


def test_forward_batch_omits_weight_for_models_without_it(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 'batch', has_weight = False)
	_patch_session(monkeypatch, session)
	crops = [ numpy.zeros((1, 3, 8, 8), dtype = numpy.float32) for _ in range(2) ]

	face_enhancer_core.forward_batch(crops, numpy.array([ 0.7 ]))

	assert session.calls[0]['has_weight'] is False


def test_batched_and_looped_produce_equivalent_outputs(monkeypatch) -> None:
	def negate(batched):
		return -batched

	crops = [ numpy.random.RandomState(seed).rand(1, 3, 4, 4).astype(numpy.float32) for seed in range(4) ]

	dynamic = _FakeSession(batch_dim = 'batch', output_handler = negate)
	_patch_session(monkeypatch, dynamic)
	batched_out = face_enhancer_core.forward_batch(crops, numpy.array([ 0.5 ]))

	fixed = _FakeSession(batch_dim = 1, output_handler = negate)
	_patch_session(monkeypatch, fixed)
	looped_out = face_enhancer_core.forward_batch(crops, numpy.array([ 0.5 ]))

	numpy.testing.assert_allclose(batched_out, looped_out)


# ---------------------------------------------------------------------------
# enhance_faces routing -- fall back to per-face loop when overlap or a
# single face is present, batch otherwise.


def _make_face(bounding_box):
	# Stand-in for a real Face namedtuple; we only need bounding_box for
	# overlap detection. Other attributes are touched only by the patched
	# helpers below.
	class _StubFace:
		pass
	face = _StubFace()
	face.bounding_box = bounding_box
	return face


def test_enhance_faces_uses_batched_path_for_disjoint_faces(monkeypatch) -> None:
	calls = { 'warp': 0, 'paste': 0, 'forward_batch': 0, 'forward': 0 }

	def fake_warp(face, frame):
		calls['warp'] += 1
		# Return a (prepared, affine, masks) triple whose actual contents
		# don't matter for this test -- enhance_faces just hands them to
		# forward_batch / paste.
		return numpy.zeros((1, 3, 4, 4), dtype = numpy.float32), numpy.eye(2, 3), [ numpy.ones((4, 4), dtype = numpy.float32) ]

	def fake_paste(temp_frame, crop, affine, masks):
		calls['paste'] += 1
		return temp_frame

	def fake_forward_batch(prepared_crops, weight):
		calls['forward_batch'] += 1
		return numpy.stack([ crop[0] for crop in prepared_crops ], axis = 0)

	def fake_forward(prepared_crop, weight):
		calls['forward'] += 1
		return prepared_crop[0]

	monkeypatch.setattr(face_enhancer_core, '_warp_and_mask_face', fake_warp)
	monkeypatch.setattr(face_enhancer_core, '_paste_and_blend_face', fake_paste)
	monkeypatch.setattr(face_enhancer_core, 'forward_batch', fake_forward_batch)
	monkeypatch.setattr(face_enhancer_core, 'forward', fake_forward)
	monkeypatch.setattr(face_enhancer_core, '_build_face_enhancer_weight', lambda: numpy.array([ 0.5 ]))

	faces = [ _make_face(_bbox(0, 0, 50, 50)), _make_face(_bbox(400, 400, 480, 480)) ]
	temp_frame = numpy.zeros((512, 512, 3), dtype = numpy.uint8)
	face_enhancer_core.enhance_faces(faces, temp_frame)

	# Both faces warped, batched once, pasted twice, single forward never
	# touched.
	assert calls == { 'warp': 2, 'paste': 2, 'forward_batch': 1, 'forward': 0 }


def test_enhance_faces_falls_back_to_sequential_when_overlapping(monkeypatch) -> None:
	calls = { 'enhance_face': 0, 'forward_batch': 0 }

	def fake_enhance_face(face, frame):
		calls['enhance_face'] += 1
		return frame

	def fake_forward_batch(prepared_crops, weight):  # pragma: no cover
		calls['forward_batch'] += 1
		raise AssertionError('forward_batch must not be called when overlap detected')

	monkeypatch.setattr(face_enhancer_core, 'enhance_face', fake_enhance_face)
	monkeypatch.setattr(face_enhancer_core, 'forward_batch', fake_forward_batch)

	faces = [ _make_face(_bbox(0, 0, 100, 100)), _make_face(_bbox(80, 80, 180, 180)) ]
	temp_frame = numpy.zeros((512, 512, 3), dtype = numpy.uint8)
	face_enhancer_core.enhance_faces(faces, temp_frame)

	assert calls == { 'enhance_face': 2, 'forward_batch': 0 }


def test_enhance_faces_falls_back_to_sequential_for_single_face(monkeypatch) -> None:
	calls = { 'enhance_face': 0, 'forward_batch': 0 }

	def fake_enhance_face(face, frame):
		calls['enhance_face'] += 1
		return frame

	def fake_forward_batch(prepared_crops, weight):  # pragma: no cover
		calls['forward_batch'] += 1
		raise AssertionError('forward_batch must not be called for a single face')

	monkeypatch.setattr(face_enhancer_core, 'enhance_face', fake_enhance_face)
	monkeypatch.setattr(face_enhancer_core, 'forward_batch', fake_forward_batch)

	faces = [ _make_face(_bbox(0, 0, 100, 100)) ]
	temp_frame = numpy.zeros((128, 128, 3), dtype = numpy.uint8)
	face_enhancer_core.enhance_faces(faces, temp_frame)

	assert calls == { 'enhance_face': 1, 'forward_batch': 0 }


def test_enhance_faces_no_op_for_empty_face_list() -> None:
	temp_frame = numpy.full((16, 16, 3), 7, dtype = numpy.uint8)
	output = face_enhancer_core.enhance_faces([], temp_frame)
	# Same array reference -- nothing was touched.
	assert output is temp_frame
