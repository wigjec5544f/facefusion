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


# ---------------------------------------------------------------------------
# v2: GPU detection (best-effort, must never raise on systems without
# nvidia-smi / rocm-smi / Apple Silicon).


def test_check_gpu_returns_at_least_one_row() -> None:
	rows = doctor.check_gpu()

	assert rows
	for label, status, detail in rows:
		assert isinstance(label, str) and label.startswith('gpu')
		assert status in [ 'ok', 'warn' ]
		assert isinstance(detail, str)


def test_probe_apple_gpus_skipped_off_darwin() -> None:
	import platform as platform_module

	if platform_module.system() != 'Darwin':
		assert doctor._probe_apple_gpus() == []


# ---------------------------------------------------------------------------
# v2: model inventory + CRC32 verification.


def test_check_models_warn_when_directory_missing(monkeypatch, tmp_path) -> None:
	missing_directory = str(tmp_path / 'does_not_exist')
	monkeypatch.setattr(doctor, 'get_models_directory', lambda: missing_directory)
	rows = doctor.check_models()

	assert rows[0][0] == 'models'
	assert rows[0][1] == 'warn'
	assert 'does not exist' in rows[0][2]


def test_check_models_warn_when_directory_empty(monkeypatch, tmp_path) -> None:
	monkeypatch.setattr(doctor, 'get_models_directory', lambda: str(tmp_path))
	rows = doctor.check_models()

	assert rows == [ ('models', 'warn', 'no .onnx files found at {}'.format(str(tmp_path))) ]


def _write_model(directory, name, payload) -> str:
	import os as os_module

	model_path = os_module.path.join(str(directory), name + '.onnx')
	with open(model_path, 'wb') as handle:
		handle.write(payload)
	return model_path


def _write_hash_for(model_path) -> None:
	from facefusion import hash_helper

	hash_path = model_path[:-5] + '.hash'
	with open(model_path, 'rb') as handle:
		content = handle.read()
	with open(hash_path, 'w', encoding = 'utf-8') as handle:
		# Sidecar is a raw 8-char CRC32 hex string (see hash_helper.create_hash).
		handle.write(hash_helper.create_hash(content))


def test_check_models_inventory_only(monkeypatch, tmp_path) -> None:
	_write_model(tmp_path, 'fake_model_a', b'\x00' * 1024)
	_write_model(tmp_path, 'fake_model_b', b'\xff' * 2048)
	monkeypatch.setattr(doctor, 'get_models_directory', lambda: str(tmp_path))
	rows = doctor.check_models()

	# Only the inventory row -- no verify rows when verify=False.
	assert len(rows) == 1
	assert rows[0][0] == 'models'
	assert rows[0][1] == 'ok'
	assert '2 models' in rows[0][2]


def test_check_models_verify_passes_for_matching_hashes(monkeypatch, tmp_path) -> None:
	model_path = _write_model(tmp_path, 'good_model', b'hello-facefusion')
	_write_hash_for(model_path)
	monkeypatch.setattr(doctor, 'get_models_directory', lambda: str(tmp_path))
	rows = doctor.check_models(verify = True)

	# Last row is the success message.
	assert any(label == 'models_hash_verified' and status == 'ok' for label, status, _ in rows)
	assert not any(status == 'fail' for _, status, _ in rows)


def test_check_models_verify_flags_corruption(monkeypatch, tmp_path) -> None:
	# Write a model with a hash sidecar, then corrupt the model so the
	# CRC32 won't match anymore.
	model_path = _write_model(tmp_path, 'bad_model', b'original-content')
	_write_hash_for(model_path)
	with open(model_path, 'wb') as handle:
		handle.write(b'corrupted!')
	monkeypatch.setattr(doctor, 'get_models_directory', lambda: str(tmp_path))
	rows = doctor.check_models(verify = True)

	assert any(label == 'models_hash_mismatch' and status == 'fail' for label, status, _ in rows)


def test_check_models_verify_flags_missing_hash_sidecar(monkeypatch, tmp_path) -> None:
	_write_model(tmp_path, 'orphan_model', b'no-sidecar-here')
	monkeypatch.setattr(doctor, 'get_models_directory', lambda: str(tmp_path))
	rows = doctor.check_models(verify = True)

	assert any(label == 'models_hash_missing' and status == 'warn' for label, status, _ in rows)


def test_check_models_reports_orphan_hash_files(monkeypatch, tmp_path) -> None:
	# .hash exists without matching .onnx -- common after a partial cleanup.
	_write_model(tmp_path, 'live_model', b'still-here')
	with open(str(tmp_path / 'ghost.hash'), 'w', encoding = 'utf-8') as handle:
		handle.write('deadbeef')
	monkeypatch.setattr(doctor, 'get_models_directory', lambda: str(tmp_path))
	rows = doctor.check_models()

	assert any(label == 'models_orphan_hashes' and status == 'warn' for label, status, _ in rows)


def test_run_checks_with_verify_models_includes_verify_rows(monkeypatch, tmp_path) -> None:
	model_path = _write_model(tmp_path, 'shipped_model', b'good-bytes')
	_write_hash_for(model_path)
	monkeypatch.setattr(doctor, 'get_models_directory', lambda: str(tmp_path))
	checks = doctor.run_checks(verify_models = True)

	# Verify rows added on top of the standard suite -- exact label depends
	# on whether we found a mismatch / missing sidecar; a passing run has
	# `models_hash_verified`.
	assert any(label == 'models_hash_verified' for label, _, _ in checks)


def test_render_with_verify_models_returns_int(monkeypatch, tmp_path) -> None:
	# Empty directory -> warn rows; render must still exit cleanly.
	monkeypatch.setattr(doctor, 'get_models_directory', lambda: str(tmp_path))
	exit_code = doctor.render(verify_models = True)

	assert exit_code in [ 0, 1 ]
