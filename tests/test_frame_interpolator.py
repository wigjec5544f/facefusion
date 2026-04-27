import os

import numpy
import pytest

from facefusion import frame_interpolator
from facefusion.filesystem import resolve_relative_path


@pytest.fixture(autouse = True)
def clear_extras_env(monkeypatch : pytest.MonkeyPatch) -> None:
	monkeypatch.delenv('FACEFUSION_HF_NAMESPACE', raising = False)
	monkeypatch.delenv('FACEFUSION_EXTRAS_REPO', raising = False)


def test_resolve_extras_url_default() -> None:
	url = frame_interpolator.resolve_extras_url('frame_interpolator/rife_4_9.onnx')
	assert url == 'https://huggingface.co/ngoqquyen/facefusion-extras/resolve/main/frame_interpolator/rife_4_9.onnx'


def test_resolve_extras_url_namespace_override(monkeypatch : pytest.MonkeyPatch) -> None:
	monkeypatch.setenv('FACEFUSION_HF_NAMESPACE', 'someone-else')
	url = frame_interpolator.resolve_extras_url('frame_interpolator/rife_4_9.onnx')
	assert url == 'https://huggingface.co/someone-else/facefusion-extras/resolve/main/frame_interpolator/rife_4_9.onnx'


def test_resolve_extras_url_repo_override(monkeypatch : pytest.MonkeyPatch) -> None:
	monkeypatch.setenv('FACEFUSION_EXTRAS_REPO', 'mystuff')
	url = frame_interpolator.resolve_extras_url('foo.onnx')
	assert url == 'https://huggingface.co/ngoqquyen/mystuff/resolve/main/foo.onnx'


def test_resolve_extras_url_strips_slashes(monkeypatch : pytest.MonkeyPatch) -> None:
	monkeypatch.setenv('FACEFUSION_HF_NAMESPACE', '  /myorg/  ')
	monkeypatch.setenv('FACEFUSION_EXTRAS_REPO', '/myrepo/')
	url = frame_interpolator.resolve_extras_url('foo.onnx')
	assert url == 'https://huggingface.co/myorg/myrepo/resolve/main/foo.onnx'


def test_resolve_extras_url_blank_falls_back(monkeypatch : pytest.MonkeyPatch) -> None:
	monkeypatch.setenv('FACEFUSION_HF_NAMESPACE', '   ')
	url = frame_interpolator.resolve_extras_url('foo.onnx')
	assert url.startswith('https://huggingface.co/ngoqquyen/facefusion-extras/')


def test_create_static_model_set_uses_extras_url() -> None:
	model_set = frame_interpolator.create_static_model_set('full')
	rife = model_set['rife_4_9']
	assert rife['sources']['frame_interpolator']['url'].endswith('/frame_interpolator/rife_4_9.onnx')
	assert rife['hashes']['frame_interpolator']['url'].endswith('/frame_interpolator/rife_4_9.hash')


def test_get_model_options_default() -> None:
	options = frame_interpolator.get_model_options()
	assert options is frame_interpolator.create_static_model_set('full').get('rife_4_9')


def test_frame_to_tensor_shape_and_range() -> None:
	frame = numpy.zeros((64, 80, 3), dtype = numpy.uint8)
	frame[..., 0] = 255  # Blue channel max in BGR.
	tensor = frame_interpolator._frame_to_tensor(frame)
	assert tensor.shape == (1, 3, 64, 80)
	assert tensor.dtype == numpy.float32
	# After BGR->RGB swap the third channel (B in original) is at index 2.
	assert numpy.allclose(tensor[0, 2, :, :], 1.0)
	assert numpy.allclose(tensor[0, 0, :, :], 0.0)
	assert numpy.allclose(tensor[0, 1, :, :], 0.0)


def test_tensor_to_frame_roundtrip() -> None:
	rng = numpy.random.default_rng(0)
	frame = rng.integers(0, 255, size = (32, 48, 3), dtype = numpy.uint8)
	tensor = frame_interpolator._frame_to_tensor(frame)
	roundtrip = frame_interpolator._tensor_to_frame(tensor)
	assert roundtrip.shape == frame.shape
	assert roundtrip.dtype == numpy.uint8
	assert numpy.array_equal(roundtrip, frame)


@pytest.mark.skipif(not os.path.isfile(resolve_relative_path('../.assets/models/rife_4_9.onnx')),
	reason = 'rife_4_9.onnx not downloaded; run `python tools/hf_publish.py --hash-only` after fetching weights')
def test_interpolate_pair_end_to_end() -> None:
	from facefusion import state_manager
	state_manager.init_item('execution_device_ids', [ 0 ])
	state_manager.init_item('execution_providers', [ 'cpu' ])
	state_manager.init_item('execution_thread_count', 4)
	state_manager.init_item('execution_queue_count', 1)

	const_frame = numpy.full((48, 64, 3), 128, dtype = numpy.uint8)
	mid_frame = frame_interpolator.interpolate_pair(const_frame, const_frame, timestep = 0.5)

	assert mid_frame.shape == const_frame.shape
	assert mid_frame.dtype == numpy.uint8
	# Network produces a near-identity result on identical inputs; allow small numerical drift.
	max_diff = numpy.abs(mid_frame.astype(int) - const_frame.astype(int)).max()
	assert max_diff <= 5


def test_tensor_to_frame_clips_overflow() -> None:
	tensor = numpy.zeros((1, 3, 4, 4), dtype = numpy.float32)
	tensor[0, 0, :, :] = 2.0  # Out-of-range high.
	tensor[0, 1, :, :] = -1.0  # Out-of-range low.
	tensor[0, 2, :, :] = 0.5
	frame = frame_interpolator._tensor_to_frame(tensor)
	# After RGB->BGR swap: idx 0 is original RGB[2]=128, idx 1 is RGB[1]=0, idx 2 is RGB[0]=255.
	# 0.5 * 255 == 127.5 -> astype(uint8) truncates to 127.
	assert frame[0, 0, 0] == 127
	assert frame[0, 0, 1] == 0
	assert frame[0, 0, 2] == 255
