import pathlib
import sys
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'tools'))

import hf_publish  # noqa: E402,I100,I202
import pytest  # noqa: E402,I100


def test_create_crc32_hash_matches_facefusion_hash_helper(tmp_path : pathlib.Path) -> None:
	weight_path = tmp_path / 'sample.onnx'
	payload = b'facefusion-test-payload'
	weight_path.write_bytes(payload)

	expected = format(zlib.crc32(payload), '08x')
	assert hf_publish.create_crc32_hash(weight_path) == expected
	# Cross-check with the canonical hasher used in facefusion.hash_helper.
	from facefusion import hash_helper
	assert hf_publish.create_crc32_hash(weight_path) == hash_helper.create_hash(payload)


def test_write_hash_file(tmp_path : pathlib.Path) -> None:
	weight_path = tmp_path / 'sample.onnx'
	weight_path.write_bytes(b'abc')
	hash_path = hf_publish.write_hash_file(weight_path)

	assert hash_path == weight_path.with_suffix('.hash')
	assert hash_path.read_text() == format(zlib.crc32(b'abc'), '08x')


def test_main_hash_only(tmp_path : pathlib.Path, capsys : pytest.CaptureFixture[str]) -> None:
	weight_path = tmp_path / 'sample.onnx'
	weight_path.write_bytes(b'data')

	rc = hf_publish.main([ '--source', str(weight_path), '--hash-only' ])
	assert rc == 0
	assert weight_path.with_suffix('.hash').exists()


def test_main_missing_source(tmp_path : pathlib.Path, capsys : pytest.CaptureFixture[str]) -> None:
	rc = hf_publish.main([ '--source', str(tmp_path / 'nonexistent.onnx'), '--hash-only' ])
	assert rc == 2


def test_main_requires_repo_id_when_uploading(tmp_path : pathlib.Path, monkeypatch : pytest.MonkeyPatch) -> None:
	weight_path = tmp_path / 'sample.onnx'
	weight_path.write_bytes(b'data')
	monkeypatch.setenv('HF_TOKEN', 'fake')

	rc = hf_publish.main([ '--source', str(weight_path) ])
	assert rc == 2


def test_main_requires_token_when_uploading(tmp_path : pathlib.Path, monkeypatch : pytest.MonkeyPatch) -> None:
	weight_path = tmp_path / 'sample.onnx'
	weight_path.write_bytes(b'data')
	monkeypatch.delenv('HF_TOKEN', raising = False)

	rc = hf_publish.main([ '--source', str(weight_path), '--repo-id', 'ngoqquyen/facefusion-extras' ])
	assert rc == 2
