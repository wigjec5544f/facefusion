#!/usr/bin/env python3
"""Upload a model weight to a HuggingFace mirror compatible with FaceFusion.

Computes the CRC32 hash of the file, writes a sibling `.hash` file (the format
FaceFusion's `hash_helper.validate_hash` expects), and uploads both to the
target HuggingFace repo. Designed to populate the user's own model mirror so
processors landing in Đợt 3-6 of `ULTRA_ROADMAP.md` can pull weights via
`FACEFUSION_HF_NAMESPACE=<user>` (see `facefusion/download.py`).

Usage examples:

	# Upload a single weight to ngoqquyen/facefusion-extras at frame_interpolator/rife_4_26.onnx
	python tools/hf_publish.py \\
		--repo-id ngoqquyen/facefusion-extras \\
		--source /path/to/rife_4_26.onnx \\
		--dest frame_interpolator/rife_4_26.onnx

	# Same, but compute hash only (no upload) for inspection
	python tools/hf_publish.py \\
		--source /path/to/rife_4_26.onnx \\
		--hash-only

Requires `huggingface_hub` (install via `pip install -e .[dev]` or `pip install huggingface_hub`).
The HF token is read from the `HF_TOKEN` environment variable.
"""
import argparse
import os
import pathlib
import sys
import zlib
from typing import Optional


def create_crc32_hash(file_path : pathlib.Path) -> str:
	"""Match facefusion.hash_helper.create_hash output format (zlib.crc32 lowercase 8-char hex)."""
	with file_path.open('rb') as handle:
		return format(zlib.crc32(handle.read()), '08x')


def write_hash_file(weight_path : pathlib.Path) -> pathlib.Path:
	hash_path = weight_path.with_suffix('.hash')
	hash_value = create_crc32_hash(weight_path)
	hash_path.write_text(hash_value)
	return hash_path


def upload_pair(repo_id : str, weight_path : pathlib.Path, dest_path : str, token : str) -> None:
	from huggingface_hub import HfApi  # local import so --hash-only works without huggingface_hub

	api = HfApi(token = token)
	hash_path = weight_path.with_suffix('.hash')
	if not hash_path.exists():
		raise SystemExit('hash file missing: ' + str(hash_path))

	dest_hash = dest_path.rsplit('.', 1)[0] + '.hash'
	api.upload_file(
		path_or_fileobj = str(weight_path),
		path_in_repo = dest_path,
		repo_id = repo_id,
		repo_type = 'model',
		commit_message = 'add ' + dest_path
	)
	api.upload_file(
		path_or_fileobj = str(hash_path),
		path_in_repo = dest_hash,
		repo_id = repo_id,
		repo_type = 'model',
		commit_message = 'add ' + dest_hash
	)


def main(argv : Optional[list] = None) -> int:
	parser = argparse.ArgumentParser(description = 'publish a model weight + hash to a HuggingFace mirror')
	parser.add_argument('--source', required = True, help = 'path to local weight file (e.g. rife_4_26.onnx)')
	parser.add_argument('--repo-id', help = 'HuggingFace repo id, e.g. ngoqquyen/facefusion-extras')
	parser.add_argument('--dest', help = 'path inside the HF repo, e.g. frame_interpolator/rife_4_26.onnx (defaults to source basename)')
	parser.add_argument('--hash-only', action = 'store_true', help = 'compute and write the .hash file locally; do not upload')
	parser.add_argument('--token', help = 'HF token (defaults to HF_TOKEN env var)')
	args = parser.parse_args(argv)

	source_path = pathlib.Path(args.source).expanduser().resolve()
	if not source_path.is_file():
		print('source not found:', source_path, file = sys.stderr)
		return 2

	hash_path = write_hash_file(source_path)
	hash_value = hash_path.read_text().strip()
	print(f'wrote hash: {hash_path} = {hash_value}')

	if args.hash_only:
		return 0

	if not args.repo_id:
		print('--repo-id is required when not using --hash-only', file = sys.stderr)
		return 2

	token = args.token or os.environ.get('HF_TOKEN')
	if not token:
		print('HF_TOKEN env var not set and --token not provided', file = sys.stderr)
		return 2

	dest_path = args.dest or source_path.name
	upload_pair(args.repo_id, source_path, dest_path, token)
	print(f'uploaded {source_path.name} -> https://huggingface.co/{args.repo_id}/resolve/main/{dest_path}')
	print(f'uploaded {hash_path.name} -> https://huggingface.co/{args.repo_id}/resolve/main/{dest_path.rsplit(".", 1)[0]}.hash')
	return 0


if __name__ == '__main__':
	sys.exit(main())
