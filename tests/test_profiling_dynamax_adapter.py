# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Dependency-boundary tests for the optional Dynamax adapter.

The ordinary unit suite must not import or execute Dynamax or its TensorFlow
Probability dependency.  Numerical and integration validation belongs in the
one-time external validation workflow; these tests cover only our lazy import
and exact-version boundary using blocked imports or synthetic modules.
"""

import builtins
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import benchmarks.profiling.dynamax_adapter as dynamax_adapter

_ROOT = Path(__file__).resolve().parents[1]


def test_adapter_module_does_not_request_external_dependencies() -> None:
    """Importing our adapter never requests Dynamax or TFP."""
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    script = """
import importlib.abc
import sys


class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        root = fullname.partition('.')[0]
        if root in {'dynamax', 'tensorflow_probability'}:
            raise AssertionError(f'unexpected external import: {fullname}')
        return None


sys.meta_path.insert(0, BlockExternal())
import benchmarks.profiling.dynamax_adapter  # noqa: E402
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        cwd=_ROOT,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_dynamax_reports_optional_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing package metadata fails before any external import."""

    def missing_version(distribution_name: str) -> str:
        assert distribution_name == "dynamax"
        raise PackageNotFoundError(distribution_name)

    monkeypatch.setattr(dynamax_adapter, "version", missing_version)
    with pytest.raises(
        ImportError,
        match=(
            "optional 'notebooks' dependency group; its resolved Dynamax "
            "version must be 1\\.0\\.2"
        ),
    ):
        dynamax_adapter._linear_gaussian_ssm_class()


def test_wrong_dynamax_version_fails_before_external_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the preregistered release may cross the import boundary."""
    real_import = builtins.__import__

    def block_external_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.partition(".")[0] in {"dynamax", "tensorflow_probability"}:
            raise AssertionError(f"unexpected external import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(dynamax_adapter, "version", lambda _: "1.1.0")
    monkeypatch.setattr(builtins, "__import__", block_external_import)
    with pytest.raises(
        RuntimeError,
        match="preregistered for version 1\\.0\\.2; found 1\\.1\\.0",
    ):
        dynamax_adapter._linear_gaussian_ssm_class()


def test_exact_dynamax_version_loads_only_fake_public_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact-version path resolves a synthetic public API module."""

    class FakeLinearGaussianSSM:
        pass

    fake_package = ModuleType("dynamax")
    fake_module = ModuleType("dynamax.linear_gaussian_ssm")
    fake_module.__dict__["LinearGaussianSSM"] = FakeLinearGaussianSSM
    fake_package.__dict__["linear_gaussian_ssm"] = fake_module

    monkeypatch.setitem(sys.modules, "dynamax", fake_package)
    monkeypatch.setitem(
        sys.modules,
        "dynamax.linear_gaussian_ssm",
        fake_module,
    )
    monkeypatch.setattr(
        dynamax_adapter,
        "version",
        lambda _: dynamax_adapter.DYNAMAX_VERSION,
    )
    tfp_modules_before = {
        name
        for name in sys.modules
        if name == "tensorflow_probability"
        or name.startswith("tensorflow_probability.")
    }

    loaded = dynamax_adapter._linear_gaussian_ssm_class()

    assert loaded is FakeLinearGaussianSSM
    tfp_modules_after = {
        name
        for name in sys.modules
        if name == "tensorflow_probability"
        or name.startswith("tensorflow_probability.")
    }
    assert tfp_modules_after == tfp_modules_before


def test_exact_version_import_failure_is_wrapped_without_real_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken exact installation yields the adapter's stable error."""
    real_import = builtins.__import__

    def fail_fake_dynamax_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "dynamax.linear_gaussian_ssm":
            raise ImportError("synthetic unavailable dependency")
        if name.partition(".")[0] == "tensorflow_probability":
            raise AssertionError(f"unexpected external import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(
        dynamax_adapter,
        "version",
        lambda _: dynamax_adapter.DYNAMAX_VERSION,
    )
    monkeypatch.setattr(builtins, "__import__", fail_fake_dynamax_import)

    with pytest.raises(
        ImportError,
        match="Dynamax 1\\.0\\.2 is installed but could not be imported",
    ) as error:
        dynamax_adapter._linear_gaussian_ssm_class()
    assert str(error.value.__cause__) == "synthetic unavailable dependency"
