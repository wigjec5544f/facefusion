import pytest

from facefusion import download


@pytest.fixture(autouse = True)
def clear_provider_env(monkeypatch : pytest.MonkeyPatch) -> None:
	for env_var in download.PROVIDER_NAMESPACE_ENV_VAR.values():
		monkeypatch.delenv(env_var, raising = False)


def test_resolve_provider_namespace_override_unset() -> None:
	assert download.resolve_provider_namespace_override('github') is None
	assert download.resolve_provider_namespace_override('huggingface') is None


def test_resolve_provider_namespace_override_huggingface(monkeypatch : pytest.MonkeyPatch) -> None:
	monkeypatch.setenv('FACEFUSION_HF_NAMESPACE', 'wigjec5544f')
	assert download.resolve_provider_namespace_override('huggingface') == 'wigjec5544f'


def test_resolve_provider_namespace_override_github(monkeypatch : pytest.MonkeyPatch) -> None:
	monkeypatch.setenv('FACEFUSION_GH_NAMESPACE', 'wigjec5544f/facefusion-assets')
	assert download.resolve_provider_namespace_override('github') == 'wigjec5544f/facefusion-assets'


def test_resolve_provider_namespace_override_strips(monkeypatch : pytest.MonkeyPatch) -> None:
	monkeypatch.setenv('FACEFUSION_HF_NAMESPACE', '  /myorg/  ')
	assert download.resolve_provider_namespace_override('huggingface') == 'myorg'


def test_resolve_provider_namespace_override_blank(monkeypatch : pytest.MonkeyPatch) -> None:
	monkeypatch.setenv('FACEFUSION_HF_NAMESPACE', '')
	assert download.resolve_provider_namespace_override('huggingface') is None


def test_apply_namespace_override_huggingface() -> None:
	original = '/facefusion/models-frame-enhancer/resolve/main/clear_reality_x4.hash'
	assert download.apply_namespace_override(original, 'huggingface', 'myorg') == '/myorg/models-frame-enhancer/resolve/main/clear_reality_x4.hash'


def test_apply_namespace_override_github() -> None:
	original = '/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.hash'
	assert download.apply_namespace_override(original, 'github', 'wigjec5544f/facefusion-assets') == '/wigjec5544f/facefusion-assets/releases/download/models-3.0.0/inswapper_128.hash'


def test_apply_namespace_override_strips_namespace() -> None:
	original = '/facefusion/models-x/resolve/main/file.onnx'
	assert download.apply_namespace_override(original, 'huggingface', '/myorg/') == '/myorg/models-x/resolve/main/file.onnx'


def test_apply_namespace_override_no_match() -> None:
	original = '/somethingelse/path/to/file.bin'
	assert download.apply_namespace_override(original, 'huggingface', 'myorg') == original


def test_apply_namespace_override_unknown_provider() -> None:
	original = '/facefusion/models-x/file.bin'
	assert download.apply_namespace_override(original, 'github', 'myorg') == original  # GH default ns is org/repo, so no prefix match


def test_provider_default_namespace_keys() -> None:
	# Sanity check: namespace map keys match env-var map keys.
	assert set(download.PROVIDER_DEFAULT_NAMESPACE.keys()) == set(download.PROVIDER_NAMESPACE_ENV_VAR.keys())


def test_env_var_naming_convention() -> None:
	for env_var in download.PROVIDER_NAMESPACE_ENV_VAR.values():
		assert env_var.startswith('FACEFUSION_')
		assert env_var.endswith('_NAMESPACE')
