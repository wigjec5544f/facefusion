import os
import platform
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

import onnxruntime

from facefusion import cli_helper, logger, metadata, state_manager, translator
from facefusion.execution import get_available_execution_providers
from facefusion.filesystem import is_directory


def render() -> int:
	headers = [ 'check', 'status', 'detail' ]
	rows : List[List[str]] = []
	exit_code = 0

	for label, status, detail in run_checks():
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


def run_checks() -> List[Tuple[str, str, str]]:
	checks : List[Tuple[str, str, str]] = []
	checks.append(check_python())
	checks.append(check_platform())
	checks.append(check_curl())
	checks.append(check_ffmpeg())
	checks.append(check_onnxruntime())
	checks.extend(check_execution_providers())
	checks.append(check_temp_path())
	checks.append(check_jobs_path())
	checks.append(check_disk_space())
	checks.append(check_system_memory())
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
