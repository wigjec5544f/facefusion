import os
import pathlib
import shutil
import subprocess
import sys
import unittest.mock as mock

import numpy
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'tools'))

import interpolate_video  # noqa: E402,I100,I202


def test_interpolate_intermediates_count() -> None:
	prev = numpy.zeros((4, 4, 3), dtype = numpy.uint8)
	nxt = numpy.zeros((4, 4, 3), dtype = numpy.uint8)
	with mock.patch.object(interpolate_video.frame_interpolator, 'interpolate_pair', side_effect = lambda a, b, timestep, model_name = None: numpy.full_like(a, int(timestep * 100))):
		intermediates = interpolate_video.frame_interpolator._interpolate_intermediates(prev, nxt, multiplier = 4)
	assert len(intermediates) == 3
	# Timesteps should be 1/4, 2/4, 3/4 -> values 25, 50, 75 from our mock.
	assert intermediates[0][0, 0, 0] == 25
	assert intermediates[1][0, 0, 0] == 50
	assert intermediates[2][0, 0, 0] == 75


def test_interpolate_intermediates_multiplier_two() -> None:
	prev = numpy.zeros((2, 2, 3), dtype = numpy.uint8)
	nxt = numpy.zeros((2, 2, 3), dtype = numpy.uint8)
	with mock.patch.object(interpolate_video.frame_interpolator, 'interpolate_pair', return_value = numpy.zeros((2, 2, 3), dtype = numpy.uint8)) as patched:
		interpolate_video.frame_interpolator._interpolate_intermediates(prev, nxt, multiplier = 2)
	patched.assert_called_once()
	assert patched.call_args.kwargs['timestep'] == 0.5


def test_run_rejects_invalid_multiplier(tmp_path : pathlib.Path) -> None:
	dummy = tmp_path / 'dummy.mp4'
	dummy.write_bytes(b'')
	rc = interpolate_video.run(dummy, tmp_path / 'out.mp4', multiplier = 1, codec = 'libx264', crf = 18, execution_providers = [ 'cpu' ])
	assert rc == 2


def test_run_rejects_missing_input(tmp_path : pathlib.Path) -> None:
	rc = interpolate_video.run(tmp_path / 'missing.mp4', tmp_path / 'out.mp4', multiplier = 2, codec = 'libx264', crf = 18, execution_providers = [ 'cpu' ])
	assert rc == 2


def test_main_argparse_defaults(tmp_path : pathlib.Path) -> None:
	# Just exercise argparse without actually running ffmpeg.
	with mock.patch.object(interpolate_video, 'run', return_value = 0) as runner:
		rc = interpolate_video.main([ '--input', str(tmp_path / 'a.mp4'), '--output', str(tmp_path / 'b.mp4') ])
	assert rc == 0
	args = runner.call_args.args
	# (input, output, multiplier, codec, crf, execution_providers)
	assert args[2] == 2  # default multiplier
	assert args[3] == 'libx264'
	assert args[4] == 18
	assert args[5] == [ 'cpu' ]


@pytest.mark.skipif(shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None,
	reason = 'ffmpeg/ffprobe not on PATH')
@pytest.mark.skipif(not os.path.isfile(pathlib.Path(__file__).resolve().parent.parent / '.assets' / 'models' / 'rife_4_9.onnx'),
	reason = 'rife_4_9.onnx not downloaded')
def test_run_returns_nonzero_on_encoder_failure(tmp_path : pathlib.Path) -> None:
	source_path = tmp_path / 'src.mp4'
	subprocess.run(
		[
			'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
			'-f', 'lavfi', '-i', 'testsrc=duration=1:size=32x32:rate=10',
			'-c:v', 'libx264', '-pix_fmt', 'yuv420p',
			str(source_path)
		],
		check = True
	)
	# An obviously bogus codec name triggers an ffmpeg error during encoder init,
	# but pipe buffering can hide that from `writer.stdin.write`. The CLI must
	# inspect `returncode` and surface the failure.
	rc = interpolate_video.run(source_path, tmp_path / 'dst.mp4', multiplier = 2, codec = 'this_codec_does_not_exist', crf = 23, execution_providers = [ 'cpu' ])
	assert rc != 0


@pytest.mark.skipif(shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None,
	reason = 'ffmpeg/ffprobe not on PATH')
@pytest.mark.skipif(not os.path.isfile(pathlib.Path(__file__).resolve().parent.parent / '.assets' / 'models' / 'rife_4_9.onnx'),
	reason = 'rife_4_9.onnx not downloaded')
def test_audio_track_preserved(tmp_path : pathlib.Path) -> None:
	"""Regression: interpolated output must keep the audio track from the input."""
	source_path = tmp_path / 'src.mp4'
	dest_path = tmp_path / 'dst.mp4'
	# Generate a 1-second clip with both video (testsrc) and audio (sine wave 440 Hz).
	subprocess.run(
		[
			'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
			'-f', 'lavfi', '-i', 'testsrc=duration=1:size=32x32:rate=10',
			'-f', 'lavfi', '-i', 'sine=frequency=440:duration=1',
			'-c:v', 'libx264', '-pix_fmt', 'yuv420p',
			'-c:a', 'aac', '-shortest',
			str(source_path)
		],
		check = True
	)
	rc = interpolate_video.run(source_path, dest_path, multiplier = 2, codec = 'libx264', crf = 23, execution_providers = [ 'cpu' ])
	assert rc == 0
	# Probe output streams; audio stream must be present.
	probe = subprocess.run(
		[
			'ffprobe', '-v', 'error',
			'-show_entries', 'stream=codec_type', '-of', 'json',
			str(dest_path)
		],
		capture_output = True, text = True, check = True
	)
	import json
	codec_types = [ stream['codec_type'] for stream in json.loads(probe.stdout)['streams'] ]
	assert 'video' in codec_types
	assert 'audio' in codec_types, f'audio track was dropped during interpolation; streams: {codec_types}'


@pytest.mark.skipif(shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None,
	reason = 'ffmpeg/ffprobe not on PATH')
@pytest.mark.skipif(not os.path.isfile(pathlib.Path(__file__).resolve().parent.parent / '.assets' / 'models' / 'rife_4_9.onnx'),
	reason = 'rife_4_9.onnx not downloaded')
def test_run_end_to_end(tmp_path : pathlib.Path) -> None:
	source_path = tmp_path / 'src.mp4'
	dest_path = tmp_path / 'dst.mp4'
	# Generate a tiny 1-second 10 fps test pattern via ffmpeg (10 frames @ 32x32).
	subprocess.run(
		[
			'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
			'-f', 'lavfi', '-i', 'testsrc=duration=1:size=32x32:rate=10',
			'-c:v', 'libx264', '-pix_fmt', 'yuv420p',
			str(source_path)
		],
		check = True
	)
	rc = interpolate_video.run(source_path, dest_path, multiplier = 2, codec = 'libx264', crf = 23, execution_providers = [ 'cpu' ])
	assert rc == 0
	assert dest_path.is_file()
	# Source has 10 frames -> output has 10 + 9 = 19 frames at 20 fps.
	probe = subprocess.run(
		[
			'ffprobe', '-v', 'error', '-select_streams', 'v:0',
			'-show_entries', 'stream=r_frame_rate,nb_frames', '-of', 'json',
			str(dest_path)
		],
		capture_output = True, text = True, check = True
	)
	import json
	stream = json.loads(probe.stdout)['streams'][0]
	assert stream['r_frame_rate'] == '20/1'
	assert int(stream['nb_frames']) == 19
