"""Tests for Đợt 3.A2 PR-A LatentSync scaffold.

PR-A registers the ``latentsync_1_5`` model entry, the
``--lip-syncer-research-models`` opt-in flag, and a stub forward
function.  PR-B will land the actual DDIM sampler / ONNX exports /
Whisper audio path; these tests pin down the contract PR-B has to
honour.

The tests intentionally avoid spinning up the full processor (no ONNX,
no audio decoding); they exercise the schema, the gate, and the stub.
"""
import pytest

from facefusion import state_manager
from facefusion.processors.modules.lip_syncer import choices as lip_syncer_choices
from facefusion.processors.modules.lip_syncer import core as lip_syncer_core


@pytest.fixture(scope = 'module', autouse = True)
def _init_state() -> None:
	# `resolve_download_url` reads `download_providers` from the global
	# state on import-time evaluation of `create_static_model_set`, so
	# every test that touches the model registry needs a sane default.
	state_manager.init_item('download_providers', [ 'github' ])


# ---------------------------------------------------------------------------
# Schema registration.


def test_latentsync_is_listed_as_a_model_choice() -> None:
	assert 'latentsync_1_5' in lip_syncer_choices.lip_syncer_models


def test_latentsync_model_entry_is_marked_research_only() -> None:
	models = lip_syncer_core.create_static_model_set('full')
	assert 'latentsync_1_5' in models
	metadata = models['latentsync_1_5'].get('__metadata__') or {}
	assert metadata.get('research_only') is True
	assert metadata.get('license') == 'Apache-2.0'
	assert metadata.get('vendor') == 'ByteDance'
	assert 'huggingface.co' in metadata.get('upstream', '')


def test_latentsync_dispatch_type_is_dedicated() -> None:
	# Distinct dispatch type prevents the wav2lip / edtalk paths from
	# accidentally trying to run their pipelines on a latentsync entry.
	models = lip_syncer_core.create_static_model_set('full')
	assert models['latentsync_1_5']['type'] == 'latentsync_research'


def test_latentsync_hashes_and_sources_are_empty_until_pr_b() -> None:
	# Empty mappings stay inert in `conditional_download_*`, so leaving
	# them blank is safe.  PR-B fills them once weights are mirrored.
	models = lip_syncer_core.create_static_model_set('full')
	assert models['latentsync_1_5']['hashes'] == {}
	assert models['latentsync_1_5']['sources'] == {}


# ---------------------------------------------------------------------------
# Stub forward function.


def test_forward_latentsync_raises_not_implemented_error() -> None:
	with pytest.raises(NotImplementedError) as exc_info:
		lip_syncer_core.forward_latentsync(temp_audio_frame = None, crop_vision_frame = None)  # type: ignore[arg-type]
	# The error must point users at the roadmap milestone so they don't
	# silently get garbage frames.
	assert 'PR-B' in str(exc_info.value)
	assert 'LatentSync' in str(exc_info.value) or 'latentsync' in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# pre_check gate.


@pytest.fixture
def select_latentsync(monkeypatch : pytest.MonkeyPatch) -> None:
	state : dict = {
		'lip_syncer_model': 'latentsync_1_5',
		'lip_syncer_research_models': False
	}
	monkeypatch.setattr(state_manager, 'get_item', lambda key : state.get(key))
	# Expose the dict so individual tests can flip the flag.
	monkeypatch.setattr(state_manager, 'set_item', lambda key, value : state.__setitem__(key, value))
	return None


def test_pre_check_rejects_research_model_without_flag(select_latentsync : None, caplog : pytest.LogCaptureFixture) -> None:
	import logging
	caplog.set_level(logging.ERROR)
	assert lip_syncer_core.pre_check() is False
	# Look for the "research-only" guidance message regardless of which
	# log channel it came through.
	error_messages = ' '.join(record.getMessage() for record in caplog.records)
	assert 'research' in error_messages.lower() or 'research' in caplog.text.lower()


def test_pre_check_rejects_research_model_even_with_flag_until_pr_b(select_latentsync : None, monkeypatch : pytest.MonkeyPatch) -> None:
	# Even with the opt-in flag, PR-A must reject because the sampler
	# isn't implemented.  PR-B will flip this expectation.
	monkeypatch.setattr(state_manager, 'get_item', lambda key : 'latentsync_1_5' if key == 'lip_syncer_model' else (True if key == 'lip_syncer_research_models' else None))
	assert lip_syncer_core.pre_check() is False


def test_pre_check_passes_for_existing_model(monkeypatch : pytest.MonkeyPatch) -> None:
	# Sanity check: the gate must not break the wav2lip / edtalk paths.
	# We stub conditional_download_* to skip the network entirely.
	monkeypatch.setattr(state_manager, 'get_item', lambda key : 'wav2lip_gan_96' if key == 'lip_syncer_model' else None)
	monkeypatch.setattr(lip_syncer_core, 'conditional_download_hashes', lambda *args, **kwargs : True)
	monkeypatch.setattr(lip_syncer_core, 'conditional_download_sources', lambda *args, **kwargs : True)
	assert lip_syncer_core.pre_check() is True


# ---------------------------------------------------------------------------
# CLI / state plumbing.


def test_apply_args_propagates_research_flag(monkeypatch : pytest.MonkeyPatch) -> None:
	captured : dict = {}

	def fake_apply(key : str, value : object) -> None:
		captured[key] = value

	lip_syncer_core.apply_args(
		args = { 'lip_syncer_model': 'latentsync_1_5', 'lip_syncer_weight': 0.5, 'lip_syncer_research_models': True },
		apply_state_item = fake_apply
	)
	assert captured['lip_syncer_model'] == 'latentsync_1_5'
	assert captured['lip_syncer_research_models'] is True


def test_apply_args_defaults_research_flag_to_none_when_absent() -> None:
	captured : dict = {}

	lip_syncer_core.apply_args(
		args = { 'lip_syncer_model': 'wav2lip_gan_96', 'lip_syncer_weight': 0.5 },
		apply_state_item = lambda key, value : captured.__setitem__(key, value)
	)
	# Missing keys come through as ``None`` so the state ends up false-y;
	# this matches the existing behaviour for other optional flags.
	assert captured.get('lip_syncer_research_models') is None


def test_register_args_uses_capital_cased_bool_fallback() -> None:
	# `cast_bool` (facefusion/common_helper.py) is case-sensitive: it
	# only round-trips the strings 'True' / 'False' (capitalised);
	# lowercase 'false' returns None. Lock in the convention by
	# inspecting the source so a future regression -- changing the
	# fallback to 'false' -- is caught here instead of by a review bot.
	import inspect

	from facefusion.common_helper import cast_bool

	assert cast_bool('False') is False
	assert cast_bool('True') is True
	# Sentinel: the lowercase form must NOT be accepted -- if this
	# assertion ever flips, the comment above and the whole reason for
	# this test goes away.
	assert cast_bool('false') is None

	source = inspect.getsource(lip_syncer_core.register_args)
	assert "'lip_syncer_research_models', 'False'" in source, (
		"register_args must pass capitalised 'False' as the .ini fallback "
		"-- lowercase 'false' silently parses to None via cast_bool."
	)
