import json
import logging

import pytest
import random

from docker.client import DockerClient
from docker.models.containers import Container
from pytest_cases import fixture, parametrize_with_cases
from pathlib import Path
from typing import Tuple, Optional

from mlserver.utils import get_normalized_version
from mlserver.repository import DEFAULT_MODEL_SETTINGS_FILENAME
from mlserver.repository.load import load_model_settings
from mlserver.types import InferenceRequest, Parameters
from mlserver.settings import (
    Settings,
    TRUSTED_RUNTIMES_ARTIFACT_PATH,
    ALLOWED_MODEL_IMPLEMENTATIONS,
)
from mlserver.cli.constants import DefaultBaseImage
from mlserver.cli.build import generate_dockerfile, build_image

from ..utils import RESTClient


def _discover_runtime_paths(implementations, base_path: Path):
    runtime_paths = []
    for implementation in implementations:
        module_name = implementation.split(".", 1)[0]
        module_file = base_path / f"{module_name}.py"
        module_dir = base_path / module_name
        if module_file.exists():
            runtime_paths.append(module_file.name)
        elif module_dir.is_dir():
            runtime_paths.append(module_dir.name)

    return sorted(set(runtime_paths))


@fixture
@parametrize_with_cases("custom_runtime_path")
def custom_image(
    docker_client: DockerClient, custom_runtime_path: str, current_cases
) -> str:
    discovered_implementations = []
    settings_paths = sorted(
        Path(custom_runtime_path).rglob(DEFAULT_MODEL_SETTINGS_FILENAME)
    )
    for settings_path in settings_paths:
        model_settings = load_model_settings(str(settings_path))
        discovered_implementations.append(model_settings.implementation_)
    custom_runtime_implementations = sorted(set(discovered_implementations))
    runtime_paths = _discover_runtime_paths(
        custom_runtime_implementations, Path(custom_runtime_path)
    )

    dockerfile = generate_dockerfile(
        custom_runtimes=custom_runtime_implementations,
        runtime_paths=runtime_paths,
        build_folder=custom_runtime_path,
    )
    current_case = current_cases["custom_image"]["custom_runtime_path"]
    image_name = f"{current_case.id}:0.1.0"
    build_image(
        custom_runtime_path,
        dockerfile,
        image_name,
    )

    yield image_name

    # in CI sometimes this fails, TODO: indentify why
    try:
        docker_client.images.remove(image=image_name, force=True)
    except Exception:
        logging.warning("skipping remove")


@pytest.fixture
def random_user_id() -> int:
    return random.randint(1000, 65536)


@pytest.fixture
def custom_runtime_server(
    docker_client: DockerClient,
    custom_image: str,
    settings: Settings,
    free_ports: Tuple[int, int, int],
    random_user_id: int,
) -> Tuple[str, str, Container]:
    host_http_port, host_grpc_port, host_metrics_port = free_ports

    container = docker_client.containers.run(
        custom_image,
        ports={
            f"{settings.http_port}/tcp": str(host_http_port),
            f"{settings.grpc_port}/tcp": str(host_grpc_port),
            f"{settings.metrics_port}/tcp": str(host_metrics_port),
        },
        detach=True,
        user=random_user_id,
        working_dir="/opt/mlserver",
        environment={
            "MLSERVER_MODELS_DIR": ".",
        },
    )

    yield (
        f"127.0.0.1:{host_http_port}",
        f"127.0.0.1:{host_grpc_port}",
        container,
    )

    container.remove(force=True)


@pytest.fixture
def custom_runtime_model_settings(custom_image_custom_runtime_path: str):
    settings_paths = sorted(
        Path(custom_image_custom_runtime_path).rglob(DEFAULT_MODEL_SETTINGS_FILENAME)
    )
    if not settings_paths:
        raise FileNotFoundError(
            f"Could not find {DEFAULT_MODEL_SETTINGS_FILENAME} under "
            f"{custom_image_custom_runtime_path}"
        )

    return load_model_settings(str(settings_paths[0]))


@pytest.mark.parametrize(
    "base_image",
    [
        None,
        "customreg/customimage:{version}-slim",
        "customreg/custonimage:customtag",
    ],
)
def test_generate_dockerfile(base_image: Optional[str]):
    dockerfile = ""
    if base_image is None:
        dockerfile = generate_dockerfile()
        base_image = DefaultBaseImage
    else:
        dockerfile = generate_dockerfile(base_image=base_image)

    expected = base_image.format(version=get_normalized_version())
    assert expected in dockerfile

    # Verify the trusted runtime section is present
    assert TRUSTED_RUNTIMES_ARTIFACT_PATH in dockerfile
    expected_allowlist = sorted(ALLOWED_MODEL_IMPLEMENTATIONS)
    expected_json = json.dumps(expected_allowlist)
    assert expected_json in dockerfile


def test_generate_dockerfile_rejects_custom_runtime_allowlist_without_runtime_paths():
    with pytest.raises(ValueError, match="Missing runtime source paths"):
        generate_dockerfile(
            custom_runtimes=["custom.MyRuntime", "custom.MyRuntime", "other.Runtime"]
        )


def test_generate_dockerfile_with_custom_runtime_allowlist_and_runtime_paths():
    dockerfile = generate_dockerfile(
        custom_runtimes=["custom.MyRuntime", "custom.MyRuntime", "other.Runtime"],
        runtime_paths=["custom.py", "other.py"],
    )

    assert TRUSTED_RUNTIMES_ARTIFACT_PATH in dockerfile
    assert '"custom.MyRuntime"' in dockerfile
    assert '"other.Runtime"' in dockerfile
    assert dockerfile.count('"custom.MyRuntime"') == 1


def test_generate_dockerfile_canonicalizes_legacy_builtin_allow_runtime():
    dockerfile = generate_dockerfile(
        custom_runtimes=["mlserver_sklearn.sklearn.SKLearnModel"]
    )

    assert '"mlserver_sklearn.SKLearnModel"' in dockerfile
    assert '"mlserver_sklearn.sklearn.SKLearnModel"' not in dockerfile


def test_generate_dockerfile_rejects_invalid_runtime_path():
    with pytest.raises(ValueError, match="Invalid runtime import path"):
        generate_dockerfile(custom_runtimes=["invalid-runtime"])


def test_generate_dockerfile_rejects_runtime_paths_without_custom_allowlist():
    with pytest.raises(ValueError, match="require matching custom runtime allowlist"):
        generate_dockerfile(runtime_paths=["custom.py"])


def test_generate_dockerfile_rejects_undeclared_runtime_paths():
    with pytest.raises(ValueError, match="undeclared runtime module"):
        generate_dockerfile(
            custom_runtimes=["custom.Runtime"],
            runtime_paths=["custom.py", "other.py"],
        )


def test_generate_dockerfile_rejects_nested_runtime_with_module_file():
    with pytest.raises(ValueError, match="require package-directory runtime paths"):
        generate_dockerfile(
            custom_runtimes=["acme.runtime.CustomRuntime"],
            runtime_paths=["acme.py"],
        )


def test_generate_dockerfile_rejects_directory_runtime_path_without_build_folder():
    with pytest.raises(
        ValueError, match="Directory runtime paths require build_folder context"
    ):
        generate_dockerfile(
            custom_runtimes=["custom.Runtime"],
            runtime_paths=["custom"],
        )


def test_generate_dockerfile_rejects_non_package_directory_runtime_path(tmp_path: Path):
    (tmp_path / "custom").mkdir()
    with pytest.raises(ValueError, match=r"containing '__init__\.py'"):
        generate_dockerfile(
            custom_runtimes=["custom.Runtime"],
            runtime_paths=["custom"],
            build_folder=str(tmp_path),
        )


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../custom.py",
        "-custom.py",
        "custom runtime.py",
        "custom*.py",
        "custom?.py",
        "custom[ab].py",
    ],
)
def test_generate_dockerfile_rejects_unsafe_runtime_paths(unsafe_path: str):
    with pytest.raises(ValueError):
        generate_dockerfile(
            custom_runtimes=["custom.Runtime"],
            runtime_paths=[unsafe_path],
        )


def test_generate_dockerfile_dev_mode():
    """Test --dev flag generates Dockerfile without allowlist."""
    dockerfile = generate_dockerfile(dev=True)

    # Should NOT contain PRODUCTION mode artifacts
    assert TRUSTED_RUNTIMES_ARTIFACT_PATH not in dockerfile
    # Should NOT contain custom runtime placeholders
    assert "{custom_runtime_copy_instructions}" not in dockerfile
    assert "{custom_runtime_pythonpath_env}" not in dockerfile
    assert "{trusted_runtime_allowlist_json}" not in dockerfile

    # Should have base image
    expected_base = DefaultBaseImage.format(version=get_normalized_version())
    assert expected_base in dockerfile

    # Should have standard CMD
    assert "mlserver start" in dockerfile


def test_build(docker_client: DockerClient, custom_image: str):
    image = docker_client.images.get(custom_image)
    assert image.tags == [custom_image]


async def test_infer_custom_runtime(
    custom_runtime_server: Tuple[str, str, Container],
    custom_runtime_model_settings,
    inference_request: InferenceRequest,
):
    http_server, _, container = custom_runtime_server
    rest_client = RESTClient(http_server)
    try:
        model_name = custom_runtime_model_settings.name

        await rest_client.wait_until_ready()

        await rest_client.wait_until_model_indexed(model_name)

        inference_request.inputs[0].parameters = Parameters(content_type="np")
        inference_response = await rest_client.infer(model_name, inference_request)
        assert len(inference_response.outputs) == 1
    except Exception as exc:
        logs = container.logs(stdout=True, stderr=True).decode(
            "utf-8", errors="replace"
        )
        raise AssertionError(
            f"Custom runtime container failed during readiness/inference.\n"
            f"Container logs:\n{logs}"
        ) from exc
    finally:
        await rest_client.close()
