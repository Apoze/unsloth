# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import subprocess

import pytest

from core.inference.llama_cpp import LlamaCppBackend


_CAPABILITY = "supports_reasoning_flag"


def _backend(
    *,
    style: str = "enable_thinking",
    supports_preserve: bool = False,
    preserve_default: bool = False,
) -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._reasoning_style = style
    backend._architecture = None
    backend._supports_preserve_thinking = supports_preserve
    backend._preserve_thinking_default = preserve_default
    return backend


def _template_kwargs(command: list[str]):
    if "--chat-template-kwargs" not in command:
        return None
    return json.loads(command[command.index("--chat-template-kwargs") + 1])


@pytest.mark.parametrize("thinking_default", [True, False])
def test_modern_launch_uses_reasoning_flag(thinking_default):
    command = ["llama-server"]
    _backend()._append_launch_reasoning_args(
        command,
        thinking_default,
        {_CAPABILITY: True},
        env = {},
    )

    assert command == ["llama-server", "--reasoning", "on" if thinking_default else "off"]
    assert "--chat-template-kwargs" not in command


def test_modern_launch_keeps_preserve_thinking_as_independent_kwarg():
    command = ["llama-server"]
    _backend(supports_preserve = True, preserve_default = True)._append_launch_reasoning_args(
        command,
        True,
        {_CAPABILITY: True},
        env = {},
    )

    assert command[1:3] == ["--reasoning", "on"]
    assert _template_kwargs(command) == {"preserve_thinking": True}


@pytest.mark.parametrize(
    "thinking_default, env_value",
    [
        (True, "off"),
        (False, "on"),
        (True, "auto"),
    ],
)
def test_modern_launch_preserves_operator_reasoning_env_override(thinking_default, env_value):
    command = ["llama-server"]
    _backend(supports_preserve = True, preserve_default = True)._append_launch_reasoning_args(
        command,
        thinking_default,
        {_CAPABILITY: True},
        env = {"LLAMA_ARG_REASONING": env_value},
    )

    assert "--reasoning" not in command
    assert _template_kwargs(command) == {"preserve_thinking": True}


def test_old_binary_keeps_chat_template_kwargs_fallback():
    command = ["llama-server"]
    _backend(supports_preserve = True, preserve_default = True)._append_launch_reasoning_args(
        command,
        False,
        {_CAPABILITY: False},
        env = {"LLAMA_ARG_REASONING": "on"},
    )

    assert "--reasoning" not in command
    assert _template_kwargs(command) == {"enable_thinking": False, "preserve_thinking": True}


def test_reasoning_effort_style_stays_on_template_kwargs():
    command = ["llama-server"]
    _backend(style = "reasoning_effort")._append_launch_reasoning_args(
        command,
        True,
        {_CAPABILITY: True},
        env = {},
    )

    assert "--reasoning" not in command
    assert _template_kwargs(command) == {"reasoning_effort": "high"}


def test_probe_detects_reasoning_flag_and_missing_binary_falls_back(tmp_path, monkeypatch):
    missing = tmp_path / "missing-llama-server"
    assert LlamaCppBackend.probe_server_capabilities(str(missing))[_CAPABILITY] is False

    binary = tmp_path / "llama-server.exe"
    binary.write_text("stub", encoding = "utf-8")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_llama_server_env_for_binary",
        classmethod(lambda cls, path: {}),
    )
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            "--reasoning VALUE\n",
            "",
        ),
    )
    LlamaCppBackend._capability_cache.clear()

    assert LlamaCppBackend.probe_server_capabilities(str(binary))[_CAPABILITY] is True
