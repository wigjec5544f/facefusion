"""Dynamic-batching helpers for ONNX-based processors.

When a model declares a dynamic (or `>1`) batch dimension, multiple
independent inputs (e.g. pixel-boost tiles, multiple face crops, multiple
audio chunks) can be stacked into a single ONNX call instead of running the
model N times sequentially. The actual win depends on the runtime: CUDA /
TensorRT typically see the largest speedup; CPU/OpenVINO see smaller wins
because they already utilise threads internally; either way, **fewer Python
↔ runtime round-trips** is always cheaper than the per-call overhead.

These helpers keep the calling site simple:

    if supports_dynamic_batch(session, 'target'):
        outputs = run_session_batched(session, base_inputs, 'target', stacked_4d_array)
    else:
        outputs = run_session_looped(session, base_inputs, 'target', stacked_4d_array)

`run_session_batched` returns a single output array of shape (N, ...) with
the same first-axis length as `stacked_4d_array`. `run_session_looped`
returns the same logical thing built by N sequential calls -- both branches
have an identical interface so calling code never has to special-case.

Output indexing: by default the first model output is returned and its
batch axis is preserved. Pass `output_index` if the model has multiple
outputs and you need a different one.

Thread safety mirrors the surrounding codebase -- the caller is expected to
hold the same `thread_semaphore` / `conditional_thread_semaphore` it would
hold around the equivalent `session.run` call. These helpers do not lock.
"""
from typing import Any, Callable, Dict, List, Optional

import numpy

from facefusion.types import VisionFrame


def supports_dynamic_batch(session : Any, input_name : str) -> bool:
	"""Return True if the ONNX session input named ``input_name`` has a
	non-fixed batch axis (i.e. either symbolic such as `'batch'` / `'N'`,
	or an integer greater than 1)."""
	try:
		inputs = session.get_inputs()
	except AttributeError:
		return False
	for model_input in inputs:
		if getattr(model_input, 'name', None) != input_name:
			continue
		shape = getattr(model_input, 'shape', None)
		if not shape:
			return False
		batch_dim = shape[0]
		# str / None / 0 / -1 → symbolic / dynamic.
		if batch_dim is None:
			return True
		if isinstance(batch_dim, str):
			return True
		if isinstance(batch_dim, int):
			# 0 and -1 are sometimes used to mean dynamic by exporters.
			if batch_dim <= 0:
				return True
			# Fixed batch=1 → cannot batch. Fixed >1 means the model has a
			# specific batch size; we'd have to pad to match -- treat as
			# "no dynamic batching" so the caller falls back to per-item
			# loops, which is always safe.
			return False
	return False


def run_session_batched(
	session : Any,
	base_inputs : Dict[str, numpy.ndarray],
	batched_input_name : str,
	batched_inputs : numpy.ndarray,
	output_index : int = 0
) -> numpy.ndarray:
	"""Run ``session`` once with ``batched_inputs`` plugged in under
	``batched_input_name``. ``base_inputs`` are the other (per-call constant)
	inputs such as embeddings and weights -- they are not duplicated; ONNX
	Runtime broadcasts singleton inputs as long as the model accepts them.

	Returns the chosen output with its full batch axis preserved.
	"""
	feed = dict(base_inputs)
	feed[batched_input_name] = batched_inputs
	outputs = session.run(None, feed)
	return outputs[output_index]


def run_session_looped(
	session : Any,
	base_inputs : Dict[str, numpy.ndarray],
	batched_input_name : str,
	batched_inputs : numpy.ndarray,
	output_index : int = 0
) -> numpy.ndarray:
	"""Fallback path: run ``session`` once per element along axis 0 of
	``batched_inputs``, then stack the results. Behaviour matches the
	original per-tile loop in face_swapper / face_enhancer."""
	per_call_outputs : List[numpy.ndarray] = []
	for index in range(batched_inputs.shape[0]):
		feed = dict(base_inputs)
		feed[batched_input_name] = batched_inputs[index : index + 1]
		outputs = session.run(None, feed)
		# Strip the batch dim that the per-call run produced so we re-stack
		# uniformly below.
		per_call_outputs.append(outputs[output_index][0])
	return numpy.stack(per_call_outputs, axis = 0)


def run_with_dynamic_batch(
	session : Any,
	base_inputs : Dict[str, numpy.ndarray],
	batched_input_name : str,
	batched_inputs : numpy.ndarray,
	output_index : int = 0,
	on_fallback : Optional[Callable[[Exception], None]] = None
) -> numpy.ndarray:
	"""Try batched inference; on failure (e.g. unexpected fixed-batch
	rejection at runtime) fall back to looped inference. ``on_fallback`` is
	invoked with the swallowed exception when present -- callers can use it
	for logging/metrics."""
	if supports_dynamic_batch(session, batched_input_name):
		try:
			return run_session_batched(session, base_inputs, batched_input_name, batched_inputs, output_index)
		except Exception as exception:  # pragma: no cover - defensive
			if on_fallback is not None:
				on_fallback(exception)
	return run_session_looped(session, base_inputs, batched_input_name, batched_inputs, output_index)


def stack_prepared_frames(prepared_frames : List[VisionFrame]) -> numpy.ndarray:
	"""Concatenate per-tile (1, C, H, W) arrays into a single (N, C, H, W)
	array. Caller is responsible for `prepare_crop_frame`-ing each tile
	first."""
	if not prepared_frames:
		raise ValueError('stack_prepared_frames requires at least one frame')
	# Each prepared frame has a leading batch axis of 1 already.
	return numpy.concatenate(prepared_frames, axis = 0)
