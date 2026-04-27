"""Tests for frame_enhancer dynamic batching (Đợt 1.G2 mở rộng).

The frame_enhancer's tile loop is the same shape as face_swapper's
pixel-boost tile loop -- N independent tiles of (1, C, H, W) -- so the
shared `facefusion.processors.batching` helpers apply identically.

These tests avoid touching the real ONNX runtime by stubbing
`get_inference_pool().get('frame_enhancer')` with a fake session that
records every call. They verify:

* `forward_batch` calls the session **once** when the model declares a
  dynamic batch axis on `'input'`.
* `forward_batch` falls back to **N** calls when the model declares a
  fixed batch=1 axis -- the original per-tile loop's behaviour.
* The end-to-end `enhance_frame` plumbing wires through `forward_batch`
  with the correct number of tiles, and the slice handed to
  `normalize_tile_frame` keeps its (1, C, H, W) shape.
"""
import numpy

from facefusion.processors.modules.frame_enhancer import core as frame_enhancer_core


# ---------------------------------------------------------------------------
# Stub session shaped like an ONNX Runtime InferenceSession.


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
		return [ _FakeInput('input', [ self.batch_dim, 3, 16, 16 ]) ]

	def run(self, output_names, feed):  # noqa: ARG002
		batched = feed['input']
		self.calls.append(batched.shape)
		# Default: identity (just return the batched input as if the model
		# scaled by 1x). Tests that need a non-trivial transform pass a
		# custom output_handler.
		if self.output_handler is None:
			return [ batched ]
		return [ self.output_handler(batched) ]


def _patch_session(monkeypatch, session):
	monkeypatch.setattr(
		frame_enhancer_core,
		'get_inference_pool',
		lambda: { 'frame_enhancer': session }
	)


# ---------------------------------------------------------------------------
# forward_batch — dispatch to batched / looped paths based on ONNX schema.


def test_forward_batch_uses_single_call_for_dynamic_batch(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 'batch')
	_patch_session(monkeypatch, session)
	tiles = [ numpy.zeros((1, 3, 16, 16), dtype = numpy.float32) for _ in range(4) ]

	output = frame_enhancer_core.forward_batch(tiles)

	# Exactly one session.run call with the full (4, 3, 16, 16) batch.
	assert len(session.calls) == 1
	assert session.calls[0] == (4, 3, 16, 16)
	assert output.shape == (4, 3, 16, 16)


def test_forward_batch_falls_back_to_loop_for_fixed_batch(monkeypatch) -> None:
	session = _FakeSession(batch_dim = 1)
	_patch_session(monkeypatch, session)
	tiles = [ numpy.zeros((1, 3, 16, 16), dtype = numpy.float32) for _ in range(4) ]

	output = frame_enhancer_core.forward_batch(tiles)

	# Four sequential session.run calls, each with a 1-element batch.
	assert len(session.calls) == 4
	assert all(call_shape == (1, 3, 16, 16) for call_shape in session.calls)
	assert output.shape == (4, 3, 16, 16)


def test_forward_batch_falls_back_when_runtime_rejects_dynamic_axis(monkeypatch) -> None:
	# Dynamic axis declared statically, but the runtime explodes when given
	# more than one element. The helper must catch and retry per element.
	state = { 'fired': 0 }

	def picky_handler(batched):
		if batched.shape[0] != 1:
			state['fired'] += 1
			raise RuntimeError('this exporter mis-tagged the batch axis')
		return batched

	session = _FakeSession(batch_dim = 'batch', output_handler = picky_handler)
	_patch_session(monkeypatch, session)
	tiles = [ numpy.zeros((1, 3, 16, 16), dtype = numpy.float32) for _ in range(3) ]

	output = frame_enhancer_core.forward_batch(tiles)

	# 1 batched call (raised) + 3 looped calls = 4 total.
	assert len(session.calls) == 4
	assert state['fired'] == 1
	assert output.shape == (3, 3, 16, 16)


# ---------------------------------------------------------------------------
# Numerical equivalence — batched and looped paths must produce identical
# outputs for the same model+inputs.


def test_batched_and_looped_paths_produce_equivalent_outputs(monkeypatch) -> None:
	def per_channel_mean(batched):
		# Some non-trivial model: replace each spatial location with its
		# channel-wise mean broadcast back across H and W.
		mean = batched.mean(axis = (2, 3), keepdims = True)
		return numpy.broadcast_to(mean, batched.shape).copy()

	tiles = [ numpy.random.RandomState(seed).rand(1, 3, 8, 8).astype(numpy.float32) for seed in range(5) ]

	dynamic_session = _FakeSession(batch_dim = 'batch', output_handler = per_channel_mean)
	_patch_session(monkeypatch, dynamic_session)
	batched_output = frame_enhancer_core.forward_batch(tiles)

	fixed_session = _FakeSession(batch_dim = 1, output_handler = per_channel_mean)
	_patch_session(monkeypatch, fixed_session)
	looped_output = frame_enhancer_core.forward_batch(tiles)

	numpy.testing.assert_allclose(batched_output, looped_output)


# ---------------------------------------------------------------------------
# enhance_frame integration — patch tile creation + forward_batch and
# confirm the full plumbing works.


def test_enhance_frame_routes_through_forward_batch(monkeypatch) -> None:
	calls = []

	def fake_create_tile_frames(vision_frame, size):
		# 6 fake tiles of size (16, 16, 3) -- the actual content doesn't
		# matter, prepare_tile_frame normalises them to the model input
		# shape declared by `size`.
		fake_tiles = [ numpy.full((16, 16, 3), index, dtype = numpy.uint8) for index in range(6) ]
		return fake_tiles, 32, 32

	def fake_forward_batch(prepared_tiles):
		calls.append(len(prepared_tiles))
		# Echo the prepared input back as the "enhanced" batch -- the
		# normalize step will deal with shape and dtype.
		return numpy.concatenate(prepared_tiles, axis = 0)

	def fake_merge_tile_frames(tiles, *args, **kwargs):
		# Simple merge -- just return a frame with the same dtype/shape so
		# blend_merge_frame doesn't choke.
		return numpy.zeros((32, 32, 3), dtype = numpy.uint8)

	def fake_get_model_options():
		# size = (model_size, padding_inner, padding_outer); scale = 1 keeps
		# arithmetic simple.
		return {
			'size': (16, 8, 4),
			'scale': 1
		}

	monkeypatch.setattr(frame_enhancer_core, 'create_tile_frames', fake_create_tile_frames)
	monkeypatch.setattr(frame_enhancer_core, 'forward_batch', fake_forward_batch)
	monkeypatch.setattr(frame_enhancer_core, 'merge_tile_frames', fake_merge_tile_frames)
	monkeypatch.setattr(frame_enhancer_core, 'get_model_options', fake_get_model_options)
	# blend_merge_frame reads frame_enhancer_blend; stub state_manager.
	monkeypatch.setattr(frame_enhancer_core.state_manager, 'get_item', lambda key: 80 if key == 'frame_enhancer_blend' else None)

	temp_frame = numpy.zeros((32, 32, 3), dtype = numpy.uint8)
	frame_enhancer_core.enhance_frame(temp_frame)

	assert calls == [ 6 ], 'forward_batch should be called exactly once with all 6 tiles'
