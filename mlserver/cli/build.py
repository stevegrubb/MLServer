import subprocess
import os
import json

from collections.abc import Sequence
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Optional

from ..logging import logger
from ..settings import (
    ALLOWED_MODEL_IMPLEMENTATIONS,
    TRUSTED_RUNTIMES_ARTIFACT_PATH,
)
from ..utils import get_normalized_version
from ._runtime_utils import (
    calculate_runtime_requirements,
    collect_runtime_declaration_issues,
    format_invalid_runtime_implementations_error,
    format_missing_runtime_declarations_error,
    normalise_runtime_import_paths,
    normalise_runtime_source_paths,
    RUNTIME_IMPORT_PATH_EXPECTED_FORMAT_CLASSNAME,
    RuntimePathValidationError,
    validate_runtime_path_preconditions,
)

from .constants import (
    DockerfileName,
    DockerfileTemplateDevelopment,
    DockerfileTemplateProduction,
    DockerignoreName,
    Dockerignore,
    DefaultBaseImage,
)


def _validate_and_normalise_allow_runtime_import_paths(
    folder: str, allow_runtime_import_paths: tuple[str, ...]
) -> list[str]:
    """Validate allowlist import paths against model-settings declarations."""
    canonical_allow_runtime_import_paths = normalise_runtime_import_paths(
        allow_runtime_import_paths,
        invalid_label="Invalid --allow-runtime value(s)",
        expected_format=RUNTIME_IMPORT_PATH_EXPECTED_FORMAT_CLASSNAME,
    )

    declared_custom_runtimes = set(canonical_allow_runtime_import_paths)
    effective_allowlist = ALLOWED_MODEL_IMPLEMENTATIONS.union(declared_custom_runtimes)
    invalid_runtime_implementations, missing_runtime_declarations = (
        collect_runtime_declaration_issues(folder, effective_allowlist)
    )

    if invalid_runtime_implementations:
        raise ValueError(
            format_invalid_runtime_implementations_error(
                invalid_runtime_implementations
            )
        )

    if missing_runtime_declarations:
        raise ValueError(
            format_missing_runtime_declarations_error(missing_runtime_declarations)
        )

    return canonical_allow_runtime_import_paths


def _validate_and_normalise_runtime_source_paths(
    runtime_source_paths: Optional[Sequence[str]],
    allow_runtime_import_paths: list[str],
    *,
    build_folder: Optional[str] = None,
    runtime_paths_without_allowlist_message: str = (
        "Runtime source paths require matching custom runtime allowlist entries."
    ),
    cli_mode: bool = False,
) -> list[str]:
    """Normalize and validate runtime source paths against custom runtime allowlist.

    Args:
        runtime_source_paths: Runtime source paths provided for validation.
        allow_runtime_import_paths: Canonical allowlisted runtime import paths.
        build_folder: Optional build folder for runtime path resolution.
        runtime_paths_without_allowlist_message: Error message when paths lack
            allowlist.
        cli_mode: If True, use strict CLI validation with relative paths and
            friendly errors.
            When True: validates strictly, returns relative paths, rejects build root,
            provides source-specific error messages (file vs directory), and converts
            RuntimePathValidationError to ValueError with CLI-friendly suggestions.
            When False (default): lenient validation for programmatic API usage.

    Returns:
        Normalized runtime paths (relative when cli_mode=True, as-provided otherwise).

    Raises:
        ValueError: If validation fails (always in cli_mode=True, or BUILD_API mode).
        RuntimePathValidationError: If validation fails in non-CLI mode.
    """
    required_modules, required_nested_packages = calculate_runtime_requirements(
        allow_runtime_import_paths,
        ALLOWED_MODEL_IMPLEMENTATIONS,
    )
    try:
        if not validate_runtime_path_preconditions(
            runtime_source_paths,
            required_modules,
            runtime_paths_without_allowlist_message=(
                runtime_paths_without_allowlist_message
            ),
        ):
            return []
        if runtime_source_paths is None:
            return []
        return normalise_runtime_source_paths(
            runtime_source_paths,
            required_modules,
            required_nested_packages,
            build_folder=build_folder,
            use_relative_paths_in_build_folder=cli_mode,
            reject_build_folder_root=cli_mode,
            source_specific_module_name_errors=cli_mode,
        )
    except RuntimePathValidationError as exc:
        if cli_mode:
            raise ValueError(exc.cli_message()) from exc
        raise


def generate_dockerfile(
    base_image: str = DefaultBaseImage,
    custom_runtimes: Optional[list[str]] = None,
    runtime_paths: Optional[list[str]] = None,
    build_folder: Optional[str] = None,
    dev: bool = False,
) -> str:
    """Generate a Dockerfile with trusted runtime allowlist and copy steps."""
    base_image = base_image.format(version=get_normalized_version())

    if dev:
        # Development mode: no trusted runtimes allowlist file, no custom runtime paths
        return DockerfileTemplateDevelopment.format(base_image=base_image)

    # Production mode: create trusted runtimes allowlist file and handle custom paths
    allow_runtime_import_paths = (
        normalise_runtime_import_paths(custom_runtimes) if custom_runtimes else []
    )
    runtime_source_paths = _validate_and_normalise_runtime_source_paths(
        runtime_paths,
        allow_runtime_import_paths,
        build_folder=build_folder,
        cli_mode=False,
    )
    # Always include default implementations plus any custom runtimes
    all_allowed = sorted(
        ALLOWED_MODEL_IMPLEMENTATIONS.union(allow_runtime_import_paths)
    )
    trusted_runtime_allowlist_json = json.dumps(all_allowed)

    copy_lines = [
        "COPY --chown=1000 "
        f"./{runtime_path} "
        f"/opt/mlserver/custom_runtime/{Path(runtime_path).name}"
        for runtime_path in runtime_source_paths
    ]
    custom_runtime_copy_instructions = "\n".join(copy_lines)
    custom_runtime_pythonpath_env = ""
    if copy_lines:
        custom_runtime_pythonpath_env = (
            'ENV PYTHONPATH="/opt/mlserver/custom_runtime:${PYTHONPATH}"'
        )

    return DockerfileTemplateProduction.format(
        base_image=base_image,
        trusted_runtime_artifact_path=TRUSTED_RUNTIMES_ARTIFACT_PATH,
        trusted_runtime_allowlist_json=trusted_runtime_allowlist_json,
        custom_runtime_copy_instructions=custom_runtime_copy_instructions,
        custom_runtime_pythonpath_env=custom_runtime_pythonpath_env,
    )


@dataclass(frozen=True)
class DockerBuildContext:
    """Validated build context for Docker image generation."""

    folder: str
    allow_runtime_import_paths: list[str]
    runtime_source_paths: list[str]
    dev: bool
    dockerfile: str

    @classmethod
    def from_cli_args(
        cls,
        folder: str,
        allow_runtime_import_paths: tuple[str, ...],
        runtime_source_paths: tuple[str, ...],
        dev: bool = False,
    ) -> "DockerBuildContext":
        """Validate and build context from CLI arguments."""
        if dev:
            # Development mode: skip validation, generate simple Dockerfile
            dockerfile = generate_dockerfile(dev=True)
            return cls(
                folder=folder,
                allow_runtime_import_paths=[],
                runtime_source_paths=[],
                dev=True,
                dockerfile=dockerfile,
            )

        # Production mode: validate and include custom runtimes
        canonical_allow_runtime_import_paths = (
            _validate_and_normalise_allow_runtime_import_paths(
                folder, allow_runtime_import_paths
            )
        )
        normalised_runtime_source_paths = _validate_and_normalise_runtime_source_paths(
            runtime_source_paths,
            canonical_allow_runtime_import_paths,
            build_folder=folder,
            runtime_paths_without_allowlist_message=(
                "Runtime source paths were provided without custom allowlisted "
                "runtimes."
            ),
            cli_mode=True,
        )
        dockerfile = generate_dockerfile(
            custom_runtimes=canonical_allow_runtime_import_paths,
            runtime_paths=normalised_runtime_source_paths,
            build_folder=folder,
            dev=False,
        )
        return cls(
            folder=folder,
            allow_runtime_import_paths=canonical_allow_runtime_import_paths,
            runtime_source_paths=normalised_runtime_source_paths,
            dev=False,
            dockerfile=dockerfile,
        )


def write_dockerfile(
    folder: str, dockerfile: str, include_dockerignore: bool = True
) -> str:
    """Write Dockerfile (and optional .dockerignore) and return its path."""
    dockerfile_path = os.path.join(folder, DockerfileName)
    with open(dockerfile_path, "w", encoding="utf-8") as dockerfile_handler:
        logger.info(f"Writing Dockerfile in {dockerfile_path}")
        dockerfile_handler.write(dockerfile)

    if include_dockerignore:
        # Point to our own .dockerignore
        # https://docs.docker.com/engine/reference/commandline/build/#use-a-dockerignore-file
        dockerignore_path = dockerfile_path + DockerignoreName
        with open(dockerignore_path, "w", encoding="utf-8") as dockerignore_handler:
            logger.info(f"Writing .dockerignore in {dockerignore_path}")
            dockerignore_handler.write(Dockerignore)

    return dockerfile_path


def _build_docker_command(
    folder: str,
    dockerfile_path: str,
    image_tag: str,
    no_cache: bool = False,
) -> list[str]:
    """Build docker build command argv."""
    command = ["docker", "build", "--rm"]
    if no_cache:
        command.append("--no-cache")
    command.extend(["-f", dockerfile_path, "-t", image_tag, folder])
    return command


def build_image(
    folder: str, dockerfile: str, image_tag: str, no_cache: bool = False
) -> str:
    """Build Docker image from generated Dockerfile."""
    logger.info(f"Building Docker image with tag {image_tag}")
    with TemporaryDirectory() as tmp_dir:
        dockerfile_path = write_dockerfile(tmp_dir, dockerfile)
        build_cmd = _build_docker_command(folder, dockerfile_path, image_tag, no_cache)
        build_env = os.environ.copy()
        build_env["DOCKER_BUILDKIT"] = "1"
        subprocess.run(build_cmd, check=True, shell=False, env=build_env)

    return image_tag
