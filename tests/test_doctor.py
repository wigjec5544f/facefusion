import sys

import pytest

from facefusion import doctor, state_manager


@pytest.fixture(scope = 'module', autouse = True)
def before_all() -> None:
	state_manager.init_item('execution_providers', [ 'cpu' ])
	state_manager.init_item('temp_path', '/tmp')
	state_manager.init_item('jobs_path', '/tmp')


def test_check_python() -> None:
	label, status, detail = doctor.check_python()

	assert label == 'python'
	assert status in [ 'ok', 'warn', 'fail' ]

	if sys.version_info >= (3, 12):
		assert status == 'ok'
	elif sys.version_info >= (3, 10):
		assert status == 'warn'


def test_check_platform() -> None:
	label, status, detail = doctor.check_platform()

	assert label == 'platform'
	assert status == 'ok'
	assert detail


def test_check_curl_and_ffmpeg() -> None:
	for check_function, expected_label in [ (doctor.check_curl, 'curl'), (doctor.check_ffmpeg, 'ffmpeg') ]:
		label, status, detail = check_function()
		assert label == expected_label
		assert status in [ 'ok', 'warn', 'fail' ]


def test_check_onnxruntime() -> None:
	label, status, detail = doctor.check_onnxruntime()

	assert label == 'onnxruntime'
	assert status == 'ok'
	assert detail


def test_check_execution_providers() -> None:
	rows = doctor.check_execution_providers()

	assert rows
	assert rows[0][0] == 'execution_providers'
	assert rows[0][1] in [ 'ok', 'fail' ]


def test_check_writable_path_unset() -> None:
	label, status, detail = doctor.check_writable_path('temp_path', None)

	assert label == 'temp_path'
	assert status == 'warn'
	assert detail == 'not configured'


def test_check_writable_path_missing() -> None:
	label, status, detail = doctor.check_writable_path('temp_path', '/nonexistent_path_for_doctor_test_xyz')

	assert label == 'temp_path'
	assert status == 'warn'
	assert 'does not exist' in detail


def test_check_disk_space_runs() -> None:
	label, status, detail = doctor.check_disk_space()

	assert label == 'disk_space'
	assert status in [ 'ok', 'warn', 'fail' ]


def test_check_system_memory_runs() -> None:
	label, status, detail = doctor.check_system_memory()

	assert label == 'system_memory'
	assert status in [ 'ok', 'warn' ]


def test_run_checks_completes() -> None:
	checks = doctor.run_checks()

	assert len(checks) >= 8
	for label, status, detail in checks:
		assert isinstance(label, str)
		assert status in [ 'ok', 'warn', 'fail' ]
		assert isinstance(detail, str)


def test_render_returns_int() -> None:
	exit_code = doctor.render()

	assert exit_code in [ 0, 1 ]
