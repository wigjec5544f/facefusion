"""Tests for expression_restorer multi-face dynamic batching.

Strategy mirrors `tests/test_face_enhancer_batching.py`:

* `_faces_overlap` is a pure helper -- unit-test the bbox geometry
  exhaustively without touching ONNX.
* The three batched ``forward_*_batch`` helpers are exercised against
  stub ONNX Runtime sessions; we verify each issues exactly one
  ``session.run`` for dynamic-batch models, falls back to N calls for
  fixed-batch models, and still produces a per-face slice that is
  bit-equal to the per-face loop.
* ``restore_expressions`` plumbing is tested by patching the inner
  helpers so we can verify (a) a single batched dispatch when faces
  don't overlap, (b) per-face sequential calls when they do overlap or
  there's only one face.

The assertion that batching is bit-equal to the sequential loop comes
from the geometry: when bounding boxes don't overlap, no face's
paste-back affects another face's warp region, so all crops sample
identical pixels in either order.
"""
import numpy

from facefusion.processors.modules.expression_restorer import core as expression_restorer_core


# ---------------------------------------------------------------------------
# Stub ONNX Runtime session.


class _FakeInput:

	def __init__(self, name, shape):
		self.name = name
		self.shape = shape


class _FakeFeatureSession:

	def __init__(self, batch_dim, output_handler = None):
		self.batch_dim = batch_dim
		self.output_handler = output_handler
		self.calls = []

	def get_inputs(self):
		return [ _FakeInput('input', [ self.batch_dim, 3, 8, 8 ]) ]

	def run(self, output_names, feed):  # noqa: ARG002
		self.calls.append({ 'input_shape': feed['input'].shape })
		if self.output_handler is None:
			# Project (N, 3, H, W) -> (N, 32, H/2, W/2) so the per-face slice
			# has a non-trivial shape independent of the input crop.
			n = feed['input'].shape[0]
			return [ numpy.full((n, 32, 4, 4), n, dtype = numpy.float32) ]
		return [ self.output_handler(feed['input']) ]


class _FakeMotionSession:

	def __init__(self, batch_dim):
		self.batch_dim = batch_dim
		self.calls = []

	def get_inputs(self):
		return [ _FakeInput('input', [ self.batch_dim, 3, 8, 8 ]) ]

	def run(self, output_names, feed):  # noqa: ARG002
		self.calls.append({ 'input_shape': feed['input'].shape })
		n = feed['input'].shape[0]
		# 7 outputs. Encode the per-row index as the first element of each
		# output so we can verify slice ordering downstream. Shapes
		# loosely match the LivePortrait motion-extractor outputs.
		row_seed = numpy.arange(n, dtype = numpy.float32).reshape(n, 1)
		pitch = row_seed + 0.1
		yaw = row_seed + 0.2
		roll = row_seed + 0.3
		scale = row_seed + 0.4
		translation = numpy.tile(row_seed + 0.5, (1, 3))
		expression = numpy.tile(row_seed + 0.6, (1, 21, 3))
		motion_points = numpy.tile(row_seed + 0.7, (1, 21, 3))
		return [ pitch, yaw, roll, scale, translation, expression, motion_points ]


class _FakeGeneratorSession:

	def __init__(self, feature_batch, source_batch, target_batch):
		self.batch_dims = (feature_batch, source_batch, target_batch)
		self.calls = []

	def get_inputs(self):
		return\
		[
			_FakeInput('feature_volume', [ self.batch_dims[0], 32, 4, 4 ]),
			_FakeInput('source', [ self.batch_dims[1], 21, 3 ]),
			_FakeInput('target', [ self.batch_dims[2], 21, 3 ])
		]

	def run(self, output_names, feed):  # noqa: ARG002
		self.calls.append({
			'feature_shape': feed['feature_volume'].shape,
			'source_shape': feed['source'].shape,
			'target_shape': feed['target'].shape
		})
		n = feed['feature_volume'].shape[0]
		# Encode the per-face index as the first pixel so the test can
		# cross-check that paste-back gets the right slice.
		output = numpy.zeros((n, 3, 8, 8), dtype = numpy.float32)
		output[:, 0, 0, 0] = numpy.arange(n)
		return [ output ]


def _patch_pool(monkeypatch, **sessions):
	monkeypatch.setattr(
		expression_restorer_core,
		'get_inference_pool',
		lambda: dict(sessions)
	)


# ---------------------------------------------------------------------------
# _faces_overlap geometry.


def _bbox(x1, y1, x2, y2):
	return numpy.array([ x1, y1, x2, y2 ], dtype = numpy.float32)


def test_faces_overlap_returns_false_for_well_separated_faces() -> None:
	bboxes = [ _bbox(0, 0, 50, 50), _bbox(200, 200, 280, 280), _bbox(400, 0, 480, 80) ]
	assert expression_restorer_core._faces_overlap(bboxes) is False


def test_faces_overlap_returns_true_for_intersecting_faces() -> None:
	bboxes = [ _bbox(0, 0, 100, 100), _bbox(80, 80, 180, 180) ]
	assert expression_restorer_core._faces_overlap(bboxes) is True


def test_faces_overlap_uses_expansion_margin() -> None:
	# Strictly disjoint by 5 px but inside the 25%-expansion margin of
	# 100-px-wide bounding boxes -> conservative report of overlap.
	bboxes = [ _bbox(0, 0, 100, 100), _bbox(105, 0, 205, 100) ]
	assert expression_restorer_core._faces_overlap(bboxes) is True


def test_faces_overlap_handles_single_face_list() -> None:
	assert expression_restorer_core._faces_overlap([ _bbox(0, 0, 50, 50) ]) is False


# ---------------------------------------------------------------------------
# forward_*_batch dispatch.


def test_forward_extract_feature_batch_uses_single_call_for_dynamic_batch(monkeypatch) -> None:
	session = _FakeFeatureSession(batch_dim = 'batch')
	_patch_pool(monkeypatch, feature_extractor = session)
	crops = numpy.zeros((4, 3, 8, 8), dtype = numpy.float32)

	output = expression_restorer_core.forward_extract_feature_batch(crops)

	assert len(session.calls) == 1
	assert session.calls[0]['input_shape'] == (4, 3, 8, 8)
	assert output.shape == (4, 32, 4, 4)


def test_forward_extract_feature_batch_falls_back_for_fixed_batch(monkeypatch) -> None:
	session = _FakeFeatureSession(batch_dim = 1)
	_patch_pool(monkeypatch, feature_extractor = session)
	crops = numpy.zeros((3, 3, 8, 8), dtype = numpy.float32)

	output = expression_restorer_core.forward_extract_feature_batch(crops)

	assert len(session.calls) == 3
	assert all(call['input_shape'] == (1, 3, 8, 8) for call in session.calls)
	assert output.shape == (3, 32, 4, 4)


def test_forward_extract_motion_batch_uses_single_call_for_dynamic_batch(monkeypatch) -> None:
	session = _FakeMotionSession(batch_dim = 'batch')
	_patch_pool(monkeypatch, motion_extractor = session)
	crops = numpy.zeros((3, 3, 8, 8), dtype = numpy.float32)

	outputs = expression_restorer_core.forward_extract_motion_batch(crops)

	assert len(session.calls) == 1
	assert len(outputs) == 7
	# pitch[i, 0] == i + 0.1 for the batched stub above
	numpy.testing.assert_allclose(outputs[0][:, 0], [ 0.1, 1.1, 2.1 ])


def test_forward_extract_motion_batch_falls_back_for_fixed_batch(monkeypatch) -> None:
	session = _FakeMotionSession(batch_dim = 1)
	_patch_pool(monkeypatch, motion_extractor = session)
	crops = numpy.zeros((3, 3, 8, 8), dtype = numpy.float32)

	outputs = expression_restorer_core.forward_extract_motion_batch(crops)

	assert len(session.calls) == 3
	assert outputs[0].shape == (3, 1)


def test_forward_generate_frame_batch_uses_single_call_for_dynamic_batch(monkeypatch) -> None:
	session = _FakeGeneratorSession(feature_batch = 'batch', source_batch = 'batch', target_batch = 'batch')
	_patch_pool(monkeypatch, generator = session)
	feature_volumes = numpy.zeros((3, 32, 4, 4), dtype = numpy.float32)
	source = numpy.zeros((3, 21, 3), dtype = numpy.float32)
	target = numpy.zeros((3, 21, 3), dtype = numpy.float32)

	output = expression_restorer_core.forward_generate_frame_batch(feature_volumes, source, target)

	assert len(session.calls) == 1
	assert output.shape == (3, 3, 8, 8)
	# Each row carries its own index in (0, 0, 0) -- verifies slice mapping.
	numpy.testing.assert_array_equal(output[:, 0, 0, 0], [ 0, 1, 2 ])


def test_forward_generate_frame_batch_falls_back_when_any_input_fixed(monkeypatch) -> None:
	# `target` is fixed batch=1: forward_generate_frame_batch must take
	# the per-face fallback regardless of the other two inputs.
	session = _FakeGeneratorSession(feature_batch = 'batch', source_batch = 'batch', target_batch = 1)
	_patch_pool(monkeypatch, generator = session)
	feature_volumes = numpy.zeros((3, 32, 4, 4), dtype = numpy.float32)
	source = numpy.zeros((3, 21, 3), dtype = numpy.float32)
	target = numpy.zeros((3, 21, 3), dtype = numpy.float32)

	output = expression_restorer_core.forward_generate_frame_batch(feature_volumes, source, target)

	assert len(session.calls) == 3
	assert all(call['feature_shape'] == (1, 32, 4, 4) for call in session.calls)
	assert output.shape == (3, 3, 8, 8)


# ---------------------------------------------------------------------------
# restore_expressions routing -- batched path vs. sequential fallback.


def _make_face(bounding_box):
	# Stand-in for a real Face namedtuple; only the bounding_box and a
	# minimal landmark_set are exercised by the patched helpers below.
	class _StubFace:
		pass
	face = _StubFace()
	face.bounding_box = bounding_box
	face.landmark_set = { '5/68': numpy.zeros((5, 2), dtype = numpy.float32) }
	return face


def test_restore_expressions_uses_batched_path_for_disjoint_faces(monkeypatch) -> None:
	calls = { 'feature_batch': 0, 'motion_batch': 0, 'generate_batch': 0, 'restore_expression': 0 }

	def fake_warp(*args, **kwargs):  # noqa: ARG001
		return numpy.zeros((8, 8, 3), dtype = numpy.uint8), numpy.eye(2, 3, dtype = numpy.float32)

	def fake_box_mask(*args, **kwargs):  # noqa: ARG001
		return numpy.ones((8, 8), dtype = numpy.float32)

	def fake_prepare(crop):  # noqa: ARG001
		return numpy.zeros((1, 3, 8, 8), dtype = numpy.float32)

	def fake_feature_batch(stack):
		calls['feature_batch'] += 1
		return numpy.zeros((stack.shape[0], 32, 4, 4), dtype = numpy.float32)

	def fake_motion_batch(stack):
		calls['motion_batch'] += 1
		n = stack.shape[0]
		return\
		(
			numpy.zeros((n, 1), dtype = numpy.float32),
			numpy.zeros((n, 1), dtype = numpy.float32),
			numpy.zeros((n, 1), dtype = numpy.float32),
			numpy.ones((n, 1), dtype = numpy.float32),
			numpy.zeros((n, 3), dtype = numpy.float32),
			numpy.zeros((n, 21, 3), dtype = numpy.float32),
			numpy.zeros((n, 21, 3), dtype = numpy.float32)
		)

	def fake_generate_batch(features, source, target):
		calls['generate_batch'] += 1
		return numpy.zeros((features.shape[0], 3, 8, 8), dtype = numpy.float32)

	def fake_normalize(crop):  # noqa: ARG001
		return numpy.zeros((8, 8, 3), dtype = numpy.uint8)

	def fake_paste_back(temp_frame, *args, **kwargs):  # noqa: ARG001
		return temp_frame

	def fake_create_rotation(pitch, yaw, roll):  # noqa: ARG001
		return numpy.eye(3, dtype = numpy.float32)

	def fake_limit_expression(value):
		return value

	def fake_restore_expression(face, target_frame, temp_frame):  # noqa: ARG001
		calls['restore_expression'] += 1
		return temp_frame

	def fake_get_model_options():
		return { 'template': 'arcface_128_v2', 'size': (256, 256) }

	def fake_state_get(name):
		if name == 'expression_restorer_factor':
			return 50
		if name == 'face_mask_blur':
			return 0.3
		if name == 'face_mask_types':
			return [ 'box' ]
		if name == 'expression_restorer_areas':
			return [ 'upper-face', 'lower-face' ]
		return None

	monkeypatch.setattr(expression_restorer_core, 'warp_face_by_face_landmark_5', fake_warp)
	monkeypatch.setattr(expression_restorer_core, 'create_box_mask', fake_box_mask)
	monkeypatch.setattr(expression_restorer_core, 'create_occlusion_mask', fake_box_mask)
	monkeypatch.setattr(expression_restorer_core, 'prepare_crop_frame', fake_prepare)
	monkeypatch.setattr(expression_restorer_core, 'forward_extract_feature_batch', fake_feature_batch)
	monkeypatch.setattr(expression_restorer_core, 'forward_extract_motion_batch', fake_motion_batch)
	monkeypatch.setattr(expression_restorer_core, 'forward_generate_frame_batch', fake_generate_batch)
	monkeypatch.setattr(expression_restorer_core, 'normalize_crop_frame', fake_normalize)
	monkeypatch.setattr(expression_restorer_core, 'paste_back', fake_paste_back)
	monkeypatch.setattr(expression_restorer_core, 'create_rotation', fake_create_rotation)
	monkeypatch.setattr(expression_restorer_core, 'limit_expression', fake_limit_expression)
	monkeypatch.setattr(expression_restorer_core, 'restore_expression', fake_restore_expression)
	monkeypatch.setattr(expression_restorer_core, 'get_model_options', fake_get_model_options)
	monkeypatch.setattr(expression_restorer_core.state_manager, 'get_item', fake_state_get)

	faces = [ _make_face(_bbox(0, 0, 50, 50)), _make_face(_bbox(400, 400, 480, 480)) ]
	temp_frame = numpy.zeros((512, 512, 3), dtype = numpy.uint8)
	expression_restorer_core.restore_expressions(faces, temp_frame, temp_frame)

	# Each ONNX model is dispatched once for the entire batch:
	#   feature_extractor: 1 batched call
	#   motion_extractor:  2 batched calls (target crops + temp crops)
	#   generator:         1 batched call
	# Per-face fallback is never used.
	assert calls == { 'feature_batch': 1, 'motion_batch': 2, 'generate_batch': 1, 'restore_expression': 0 }


def test_restore_expressions_falls_back_to_sequential_when_overlapping(monkeypatch) -> None:
	calls = { 'restore_expression': 0, 'feature_batch': 0 }

	def fake_restore_expression(face, target_frame, temp_frame):  # noqa: ARG001
		calls['restore_expression'] += 1
		return temp_frame

	def fake_feature_batch(stack):  # pragma: no cover
		calls['feature_batch'] += 1
		raise AssertionError('forward_extract_feature_batch must not run when overlap detected')

	monkeypatch.setattr(expression_restorer_core, 'restore_expression', fake_restore_expression)
	monkeypatch.setattr(expression_restorer_core, 'forward_extract_feature_batch', fake_feature_batch)

	faces = [ _make_face(_bbox(0, 0, 100, 100)), _make_face(_bbox(80, 80, 180, 180)) ]
	temp_frame = numpy.zeros((512, 512, 3), dtype = numpy.uint8)
	expression_restorer_core.restore_expressions(faces, temp_frame, temp_frame)

	assert calls == { 'restore_expression': 2, 'feature_batch': 0 }


def test_restore_expressions_falls_back_to_sequential_for_single_face(monkeypatch) -> None:
	calls = { 'restore_expression': 0, 'feature_batch': 0 }

	def fake_restore_expression(face, target_frame, temp_frame):  # noqa: ARG001
		calls['restore_expression'] += 1
		return temp_frame

	def fake_feature_batch(stack):  # pragma: no cover
		calls['feature_batch'] += 1
		raise AssertionError('forward_extract_feature_batch must not run for a single face')

	monkeypatch.setattr(expression_restorer_core, 'restore_expression', fake_restore_expression)
	monkeypatch.setattr(expression_restorer_core, 'forward_extract_feature_batch', fake_feature_batch)

	faces = [ _make_face(_bbox(0, 0, 100, 100)) ]
	temp_frame = numpy.zeros((128, 128, 3), dtype = numpy.uint8)
	expression_restorer_core.restore_expressions(faces, temp_frame, temp_frame)

	assert calls == { 'restore_expression': 1, 'feature_batch': 0 }


def test_restore_expressions_no_op_for_empty_face_list() -> None:
	temp_frame = numpy.full((16, 16, 3), 7, dtype = numpy.uint8)
	output = expression_restorer_core.restore_expressions([], temp_frame, temp_frame)
	# Same array reference -- nothing was touched.
	assert output is temp_frame
