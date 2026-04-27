import os
import platform
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

import onnxruntime

from facefusion import cli_helper, hash_helper, logger, metadata, state_manager, translator
from facefusion.execution import get_available_execution_providers
from facefusion.filesystem import is_directory, is_file, resolve_relative_path


def render(verify_models : bool = False) -> int:
	headers = [ 'check', 'status', 'detail' ]
	rows : List[List[str]] = []
	exit_code = 0

	for label, status, detail in run_checks(verify_models = verify_models):
		rows.append([ label, status, detail ])
		if status == 'fail':
			exit_code = 1

	logger.info(translator.get('doctor_header').format(version = metadata.get('version')), __name__)
	cli_helper.render_table(headers, rows)

	if exit_code == 0:
		logger.info(translator.get('doctor_summary_ok'), __name__)
	else:
		logger.error(translator.get('doctor_summary_fail'), __name__)
	return exit_code


def run_checks(verify_models : bool = False) -> List[Tuple[str, str, str]]:
	checks : List[Tuple[str, str, str]] = []
	checks.append(check_python())
	checks.append(check_platform())
	checks.append(check_curl())
	checks.append(check_ffmpeg())
	checks.append(check_onnxruntime())
	checks.extend(check_execution_providers())
	checks.extend(check_gpu())
	checks.append(check_temp_path())
	checks.append(check_jobs_path())
	checks.append(check_disk_space())
	checks.append(check_system_memory())
	checks.extend(check_models(verify = verify_models))
	return checks


def check_python() -> Tuple[str, str, str]:
	version = '{}.{}.{}'.format(*sys.version_info[:3])

	if sys.version_info < (3, 10):
		return ('python', 'fail', version + ' (need >= 3.10)')
	if sys.version_info < (3, 12):
		return ('python', 'warn', version + ' (3.12 recommended)')
	return ('python', 'ok', version)


def check_platform() -> Tuple[str, str, str]:
	return ('platform', 'ok', '{} {} ({})'.format(platform.system(), platform.release(), platform.machine()))


def check_curl() -> Tuple[str, str, str]:
	curl_path = shutil.which('curl')

	if not curl_path:
		return ('curl', 'fail', 'not found in PATH')
	return ('curl', 'ok', curl_path)


def check_ffmpeg() -> Tuple[str, str, str]:
	ffmpeg_path = shutil.which('ffmpeg')

	if not ffmpeg_path:
		return ('ffmpeg', 'fail', 'not found in PATH')

	version = read_ffmpeg_version(ffmpeg_path)

	if not version:
		return ('ffmpeg', 'warn', ffmpeg_path + ' (version unknown)')
	return ('ffmpeg', 'ok', '{} ({})'.format(version, ffmpeg_path))


def read_ffmpeg_version(ffmpeg_path : str) -> Optional[str]:
	try:
		process = subprocess.run([ ffmpeg_path, '-version' ], capture_output = True, text = True, timeout = 10)
	except (OSError, subprocess.SubprocessError):
		return None

	if process.returncode != 0 or not process.stdout:
		return None
	first_line = process.stdout.splitlines()[0]
	parts = first_line.split(' ')

	if len(parts) >= 3:
		return parts[2]
	return first_line


def check_onnxruntime() -> Tuple[str, str, str]:
	return ('onnxruntime', 'ok', onnxruntime.__version__)


def check_execution_providers() -> List[Tuple[str, str, str]]:
	available = get_available_execution_providers()
	configured = state_manager.get_item('execution_providers') or []

	if not available:
		return [ ('execution_providers', 'fail', 'none available') ]

	rows : List[Tuple[str, str, str]] = [ ('execution_providers', 'ok', ', '.join(available)) ]

	for provider in configured:
		if provider in available:
			rows.append(('  -> ' + provider, 'ok', 'configured and available'))
		else:
			rows.append(('  -> ' + provider, 'fail', 'configured but not available'))
	return rows


def check_temp_path() -> Tuple[str, str, str]:
	return check_writable_path('temp_path', state_manager.get_item('temp_path'))


def check_jobs_path() -> Tuple[str, str, str]:
	return check_writable_path('jobs_path', state_manager.get_item('jobs_path'))


def check_writable_path(label : str, target_path : Optional[str]) -> Tuple[str, str, str]:
	if not target_path:
		return (label, 'warn', 'not configured')

	if not is_directory(target_path):
		return (label, 'warn', target_path + ' (does not exist)')

	if not os.access(target_path, os.W_OK):
		return (label, 'fail', target_path + ' (not writable)')
	return (label, 'ok', target_path)


def check_disk_space() -> Tuple[str, str, str]:
	target_path = state_manager.get_item('temp_path') or os.getcwd()

	try:
		usage = shutil.disk_usage(target_path)
	except OSError:
		return ('disk_space', 'warn', 'unable to read')
	free_gib = usage.free / (1024 ** 3)
	detail = '{:.1f} GiB free at {}'.format(free_gib, target_path)

	if free_gib < 5:
		return ('disk_space', 'fail', detail + ' (need >= 5 GiB)')
	if free_gib < 20:
		return ('disk_space', 'warn', detail + ' (recommended >= 20 GiB)')
	return ('disk_space', 'ok', detail)


def check_system_memory() -> Tuple[str, str, str]:
	total_bytes = read_total_memory()

	if total_bytes is None:
		return ('system_memory', 'warn', 'unable to read')
	total_gib = total_bytes / (1024 ** 3)
	detail = '{:.1f} GiB total'.format(total_gib)

	if total_gib < 8:
		return ('system_memory', 'warn', detail + ' (recommended >= 8 GiB)')
	return ('system_memory', 'ok', detail)


def read_total_memory() -> Optional[int]:
	try:
		page_size = os.sysconf('SC_PAGE_SIZE')
		page_count = os.sysconf('SC_PHYS_PAGES')
		return int(page_size) * int(page_count)
	except (AttributeError, OSError, ValueError):
		pass

	if shutil.which('wmic'):
		try:
			process = subprocess.run([ 'wmic', 'computersystem', 'get', 'TotalPhysicalMemory' ], capture_output = True, text = True, timeout = 10)
		except (OSError, subprocess.SubprocessError):
			return None

		for line in process.stdout.splitlines():
			value = line.strip()
			if value.isdigit():
				return int(value)
	return None


def check_gpu() -> List[Tuple[str, str, str]]:
	"""Probe NVIDIA / AMD / Apple GPUs and report name + VRAM if available.

	The check is best-effort: missing tools (nvidia-smi / rocm-smi /
	system_profiler) result in a `warn` row, never `fail`, because a
	CPU-only environment is a perfectly legitimate setup for facefusion.
	"""
	rows : List[Tuple[str, str, str]] = []
	rows.extend(_probe_nvidia_gpus())
	rows.extend(_probe_amd_gpus())
	rows.extend(_probe_apple_gpus())

	if not rows:
		rows.append(('gpu', 'warn', 'no nvidia-smi / rocm-smi / Apple Silicon detected'))
	return rows


def _probe_nvidia_gpus() -> List[Tuple[str, str, str]]:
	binary = shutil.which('nvidia-smi')

	if not binary:
		return []
	try:
		process = subprocess.run(
			[ binary, '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits' ],
			capture_output = True,
			text = True,
			timeout = 10
		)
	except (OSError, subprocess.SubprocessError):
		return [ ('gpu_nvidia', 'warn', 'nvidia-smi present but failed to query') ]

	if process.returncode != 0 or not process.stdout.strip():
		return [ ('gpu_nvidia', 'warn', 'nvidia-smi returned no devices') ]

	rows : List[Tuple[str, str, str]] = []
	for index, line in enumerate(process.stdout.strip().splitlines()):
		parts = [ part.strip() for part in line.split(',') ]
		if len(parts) < 3:
			continue
		name, vram_mib, driver_version = parts[0], parts[1], parts[2]
		try:
			vram_gib = float(vram_mib) / 1024
		except ValueError:
			vram_gib = 0.0
		rows.append((
			'gpu_nvidia[{}]'.format(index),
			'ok',
			'{} ({:.1f} GiB VRAM, driver {})'.format(name, vram_gib, driver_version)
		))
	return rows


def _probe_amd_gpus() -> List[Tuple[str, str, str]]:
	binary = shutil.which('rocm-smi')

	if not binary:
		return []
	try:
		process = subprocess.run([ binary, '--showproductname', '--showmeminfo', 'vram' ], capture_output = True, text = True, timeout = 10)
	except (OSError, subprocess.SubprocessError):
		return [ ('gpu_amd', 'warn', 'rocm-smi present but failed to query') ]

	if process.returncode != 0 or not process.stdout.strip():
		return [ ('gpu_amd', 'warn', 'rocm-smi returned no devices') ]
	# rocm-smi output is dense; just surface the first relevant line so the
	# user knows ROCm is wired up. Detailed parsing is out of scope.
	first_line = next((line.strip() for line in process.stdout.splitlines() if line.strip()), '')
	return [ ('gpu_amd', 'ok', first_line[:120] or 'detected') ]


def _probe_apple_gpus() -> List[Tuple[str, str, str]]:
	if platform.system() != 'Darwin':
		return []
	binary = shutil.which('system_profiler')

	if not binary:
		return []
	try:
		process = subprocess.run([ binary, 'SPDisplaysDataType' ], capture_output = True, text = True, timeout = 10)
	except (OSError, subprocess.SubprocessError):
		return [ ('gpu_apple', 'warn', 'system_profiler failed') ]

	if process.returncode != 0:
		return []
	for line in process.stdout.splitlines():
		stripped = line.strip()
		if stripped.startswith('Chipset Model:'):
			return [ ('gpu_apple', 'ok', stripped.split(':', 1)[1].strip()) ]
	return []


def get_models_directory() -> str:
	return resolve_relative_path('../.assets/models')


def check_models(verify : bool = False) -> List[Tuple[str, str, str]]:
	"""Inventory the local model cache and -- when ``verify=True`` -- check
	each ONNX file's CRC32 against its sibling `.hash` file.

	`verify=True` is opt-in because hashing every model on disk can take a
	few seconds and isn't necessary for routine `doctor` runs.
	"""
	models_directory = get_models_directory()

	if not is_directory(models_directory):
		return [ ('models', 'warn', '{} does not exist (run install.bat / install.py first)'.format(models_directory)) ]

	onnx_files = [ entry for entry in os.listdir(models_directory) if entry.endswith('.onnx') ]
	hash_files = [ entry for entry in os.listdir(models_directory) if entry.endswith('.hash') ]

	if not onnx_files:
		return [ ('models', 'warn', 'no .onnx files found at {}'.format(models_directory)) ]

	total_bytes = 0
	for onnx_file in onnx_files:
		try:
			total_bytes += os.path.getsize(os.path.join(models_directory, onnx_file))
		except OSError:
			continue

	rows : List[Tuple[str, str, str]] = [
		('models', 'ok', '{} models, {:.1f} GiB at {}'.format(len(onnx_files), total_bytes / (1024 ** 3), models_directory))
	]

	orphan_hashes = [ entry for entry in hash_files if entry[:-5] + '.onnx' not in onnx_files ]
	if orphan_hashes:
		rows.append(('models_orphan_hashes', 'warn', '{} orphan .hash without .onnx (e.g. {})'.format(len(orphan_hashes), ', '.join(orphan_hashes[:3]))))

	if verify:
		rows.extend(_verify_model_hashes(models_directory, onnx_files))
	return rows


def _verify_model_hashes(models_directory : str, onnx_files : List[str]) -> List[Tuple[str, str, str]]:
	mismatches : List[str] = []
	missing : List[str] = []
	for onnx_file in onnx_files:
		onnx_path = os.path.join(models_directory, onnx_file)
		hash_path = onnx_path.rsplit('.onnx', 1)[0] + '.hash'

		if not is_file(hash_path):
			missing.append(onnx_file)
			continue
		if not hash_helper.validate_hash(onnx_path):
			mismatches.append(onnx_file)

	rows : List[Tuple[str, str, str]] = []
	if missing:
		rows.append(('models_hash_missing', 'warn', '{} model(s) without .hash sidecar (e.g. {})'.format(len(missing), ', '.join(missing[:3]))))
	if mismatches:
		rows.append(('models_hash_mismatch', 'fail', '{} model(s) failed CRC32 (e.g. {}) -- re-download'.format(len(mismatches), ', '.join(mismatches[:3]))))
	if not missing and not mismatches:
		rows.append(('models_hash_verified', 'ok', '{} model(s) match their .hash sidecar'.format(len(onnx_files))))
	return rows
