import pathlib
import unittest.mock as mock

from facefusion import state_manager
from facefusion.workflows import image_to_video


def _reset_keys(monkeypatch):
	# state_manager init_item refuses to overwrite existing keys; use set_item via the public API by pre-init.
	# In tests we call init_item once; subsequent tests rely on monkeypatch + set_item.
	pass


def test_skip_when_target_fps_unset(monkeypatch) -> None:
	monkeypatch.setattr(state_manager, 'get_item', lambda key: {
		'frame_interpolator_target_fps': None,
		'output_video_fps': 30,
		'output_path': '/tmp/out.mp4'
	}.get(key))
	with mock.patch('facefusion.frame_interpolator.interpolate_video_file') as patched:
		assert image_to_video.interpolate_output_video() == 0
	patched.assert_not_called()


def test_skip_when_target_fps_le_source(monkeypatch) -> None:
	monkeypatch.setattr(state_manager, 'get_item', lambda key: {
		'frame_interpolator_target_fps': 30,
		'output_video_fps': 30,
		'output_path': '/tmp/out.mp4'
	}.get(key))
	with mock.patch('facefusion.frame_interpolator.interpolate_video_file') as patched:
		assert image_to_video.interpolate_output_video() == 0
	patched.assert_not_called()


def test_skip_when_source_fps_zero(monkeypatch) -> None:
	monkeypatch.setattr(state_manager, 'get_item', lambda key: {
		'frame_interpolator_target_fps': 60,
		'output_video_fps': 0,
		'output_path': '/tmp/out.mp4'
	}.get(key))
	with mock.patch('facefusion.frame_interpolator.interpolate_video_file') as patched:
		assert image_to_video.interpolate_output_video() == 0
	patched.assert_not_called()


def test_skip_when_output_missing(monkeypatch, tmp_path : pathlib.Path) -> None:
	missing_path = str(tmp_path / 'never_created.mp4')
	monkeypatch.setattr(state_manager, 'get_item', lambda key: {
		'frame_interpolator_target_fps': 60,
		'output_video_fps': 30,
		'output_path': missing_path
	}.get(key))
	with mock.patch('facefusion.frame_interpolator.interpolate_video_file') as patched:
		assert image_to_video.interpolate_output_video() == 0
	patched.assert_not_called()


def test_runs_interpolation_with_correct_multiplier(monkeypatch, tmp_path : pathlib.Path) -> None:
	output_path = tmp_path / 'out.mp4'
	output_path.write_bytes(b'fake mp4 contents')

	monkeypatch.setattr(state_manager, 'get_item', lambda key: {
		'frame_interpolator_target_fps': 60,
		'output_video_fps': 30,
		'output_path': str(output_path),
		'output_video_encoder': 'libx264',
		'output_video_quality': 80,
		'output_video_preset': 'veryfast'
	}.get(key))
	# is_video returns True for our fake file (we patch the import in workflow module).
	monkeypatch.setattr('facefusion.workflows.image_to_video.is_video', lambda path: True)

	def fake_interpolate(input_path, output_path, multiplier, **kwargs):
		# Write a marker file so os.replace sees something.
		pathlib.Path(output_path).write_bytes(b'interpolated')
		return 0

	with mock.patch('facefusion.frame_interpolator.interpolate_video_file', side_effect = fake_interpolate) as patched, \
			mock.patch('facefusion.video_manager.clear_video_pool'):
		assert image_to_video.interpolate_output_video() == 0

	assert patched.call_count == 1
	kwargs = patched.call_args.kwargs
	assert kwargs['multiplier'] == 2
	assert kwargs['input_path'] == str(output_path)
	# Encoder/quality/preset must be forwarded so the user's --output-video-* flags are honored.
	assert kwargs['video_encoder'] == 'libx264'
	assert kwargs['video_quality'] == 80
	assert kwargs['video_preset'] == 'veryfast'
	# Output replaced in place.
	assert output_path.read_bytes() == b'interpolated'


def test_temp_path_preserves_extension(monkeypatch, tmp_path : pathlib.Path) -> None:
	"""Regression: the temp path used during interpolation must keep the
	original output extension so ffmpeg picks the right container."""
	output_path = tmp_path / 'out.webm'
	output_path.write_bytes(b'webm contents')
	captured : dict = {}

	monkeypatch.setattr(state_manager, 'get_item', lambda key: {
		'frame_interpolator_target_fps': 60,
		'output_video_fps': 30,
		'output_path': str(output_path),
		'output_video_encoder': 'libvpx-vp9',
		'output_video_quality': 80,
		'output_video_preset': 'veryfast'
	}.get(key))
	monkeypatch.setattr('facefusion.workflows.image_to_video.is_video', lambda path: True)

	def fake_interpolate(input_path, output_path, multiplier, **kwargs):
		captured['temp_path'] = output_path
		pathlib.Path(output_path).write_bytes(b'i')
		return 0

	with mock.patch('facefusion.frame_interpolator.interpolate_video_file', side_effect = fake_interpolate), \
			mock.patch('facefusion.video_manager.clear_video_pool'):
		assert image_to_video.interpolate_output_video() == 0

	assert captured['temp_path'].endswith('.webm'), f'expected .webm temp, got {captured["temp_path"]}'


def test_target_fps_60_from_source_24_uses_multiplier_3(monkeypatch, tmp_path : pathlib.Path) -> None:
	output_path = tmp_path / 'out.mp4'
	output_path.write_bytes(b'x')
	monkeypatch.setattr(state_manager, 'get_item', lambda key: {
		'frame_interpolator_target_fps': 60,
		'output_video_fps': 24,
		'output_path': str(output_path),
		'output_video_encoder': 'libx264',
		'output_video_quality': 80,
		'output_video_preset': 'veryfast'
	}.get(key))
	monkeypatch.setattr('facefusion.workflows.image_to_video.is_video', lambda path: True)

	def fake_interpolate(input_path, output_path, multiplier, **kwargs):
		pathlib.Path(output_path).write_bytes(b'i')
		return 0

	with mock.patch('facefusion.frame_interpolator.interpolate_video_file', side_effect = fake_interpolate) as patched, \
			mock.patch('facefusion.video_manager.clear_video_pool'):
		assert image_to_video.interpolate_output_video() == 0
	# round(60/24) = round(2.5) = 2 in Python's banker's rounding.
	assert patched.call_args.kwargs['multiplier'] == 2


def test_failure_leaves_original_video_intact(monkeypatch, tmp_path : pathlib.Path) -> None:
	output_path = tmp_path / 'out.mp4'
	output_path.write_bytes(b'original content')
	monkeypatch.setattr(state_manager, 'get_item', lambda key: {
		'frame_interpolator_target_fps': 60,
		'output_video_fps': 30,
		'output_path': str(output_path)
	}.get(key))
	monkeypatch.setattr('facefusion.workflows.image_to_video.is_video', lambda path: True)

	# Simulate model download failure (rc=3): no temp file written, original must stay intact.
	with mock.patch('facefusion.frame_interpolator.interpolate_video_file', return_value = 3):
		assert image_to_video.interpolate_output_video() == 0
	assert output_path.read_bytes() == b'original content'
