from __future__ import annotations

import importlib
import multiprocessing
import os
import sys
import uuid
from pathlib import Path

import pytest

from activitysim.core import workflow


@pytest.fixture
def extension(tmp_path):
    name = "extension_" + uuid.uuid4().hex
    package = tmp_path / "model space" / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from .values import VALUE\n")
    (package / "values.py").write_text("VALUE = 42\n")
    yield package
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]


def state_at(directory):
    for name in ("configs", "data"):
        (directory / name).mkdir(exist_ok=True)
    return workflow.State.make_default(directory)


@pytest.mark.parametrize(
    "form", ["bare", "relative", "absolute", "dot", "trailing", "pathlike", "posix"]
)
def test_api_paths(extension, tmp_path, monkeypatch, form):
    state = state_at(extension.parent)
    # The state's working directory deliberately differs from the process CWD.
    monkeypatch.chdir(tmp_path)
    options = {
        "bare": extension.name,
        "relative": os.path.join("..", extension.parent.name, extension.name),
        "absolute": str(extension),
        "dot": os.path.join(".", extension.name),
        "trailing": str(extension) + os.sep,
        "pathlike": Path(extension.name),
        "posix": extension.as_posix(),
    }
    old_path = sys.path[:]
    state.import_extensions(options[form])
    assert sys.path == old_path
    assert sys.modules[extension.name].VALUE == 42
    assert state.get("_extension_locations") == {extension.name: str(extension)}
    # Match downstream consumers such as SANDAG's settings-checker discovery.
    for name in state.get("imported_extensions"):
        assert importlib.import_module(name + ".values").VALUE == 42
    assert state.get("imported_extensions") == [extension.name]


def test_api_without_filesystem(extension, monkeypatch):
    monkeypatch.chdir(extension.parent)
    state = workflow.State()
    state.import_extensions(extension.name)
    assert state.get("imported_extensions") == [extension.name]


def test_append_replace_and_noop(extension):
    state = state_at(extension.parent)
    state.import_extensions(extension.name)
    before = state.get("imported_extensions")
    before_locations = state.get("_extension_locations")
    state.import_extensions([extension.name + ".values"])
    assert before == [extension.name]  # Do not mutate a caller's retained list.
    assert len(state.get("imported_extensions")) == 2
    assert before_locations == {extension.name: str(extension)}
    assert set(state.get("_extension_locations")) == {
        extension.name,
        extension.name + ".values",
    }
    state.import_extensions(None, append=False)
    assert len(state.get("imported_extensions")) == 2
    state.import_extensions(extension.name, append=False)
    assert state.get("imported_extensions") == [extension.name]
    assert state.get("_extension_locations") == {extension.name: str(extension)}
    state.import_extensions([], append=False)
    assert state.get("imported_extensions") == []
    assert state.get("_extension_locations") == {}


def test_dotted_name_on_python_path(extension, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(extension.parent))
    state = state_at(tmp_path)
    state.import_extensions(extension.name + ".values")
    assert sys.modules[extension.name + ".values"].VALUE == 42


@pytest.mark.parametrize(
    "source,exception",
    [
        ("import missing_extension_dependency_1118", ModuleNotFoundError),
        ('raise RuntimeError("extension failed")', RuntimeError),
    ],
)
def test_failed_import_restores_path_and_registration(extension, source, exception):
    (extension / "__init__.py").write_text(source)
    state = state_at(extension.parent)
    old_path = sys.path[:]
    with pytest.raises(exception):
        state.import_extensions(extension.name)
    assert sys.path == old_path
    assert state.get("imported_extensions", []) == []


def test_missing_extension(extension):
    state = state_at(extension.parent)
    old_path = sys.path[:]
    with pytest.raises(ModuleNotFoundError, match="missing_extension_1118"):
        state.import_extensions("missing_extension_1118")
    assert sys.path == old_path


def _worker_import(injectables, cwd, name, connection):
    """Use the real worker setup in a fresh process with a different CWD."""
    from activitysim.core.mp_tasks import setup_injectables_and_logging

    os.chdir(cwd)
    # fork inherits modules; explicitly require an import from the saved location.
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]
    try:
        old_path = sys.path[:]
        state = setup_injectables_and_logging(injectables)
        module = importlib.import_module(state.get("imported_extensions")[0])
        connection.send((module.VALUE, module.__file__, sys.path == old_path))
    finally:
        connection.close()


@pytest.mark.parametrize("method", multiprocessing.get_all_start_methods())
@pytest.mark.parametrize("saved_locations", [True, False])
def test_worker_after_cwd_changes(extension, tmp_path, method, saved_locations):
    state = state_at(extension.parent)
    state.import_extensions(extension.name)
    elsewhere = tmp_path / "other cwd"
    elsewhere.mkdir()
    # A same-named module in the worker CWD must not shadow the saved location.
    (elsewhere / (extension.name + ".py")).write_text("VALUE = -1\n")
    injectables = dict(
        configs_dir=[extension.parent / "configs"],
        data_dir=[extension.parent / "data"],
        output_dir=tmp_path / "worker-output",
        imported_extensions=state.get("imported_extensions"),
        _extension_locations=state.get("_extension_locations"),
    )
    if not saved_locations:
        # Legacy callers can still supply just the public module-name registry.
        injectables.pop("_extension_locations")
        injectables["working_dir"] = extension.parent
    context = multiprocessing.get_context(method)
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_import, args=(injectables, elsewhere, extension.name, sender)
    )
    process.start()
    sender.close()
    try:
        assert receiver.poll(60), "Worker did not return an imported extension"
        value, filename, restored = receiver.recv()
        process.join(30)
        assert process.exitcode == 0
        assert value == 42
        assert Path(filename) == extension / "__init__.py"
        assert restored
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        receiver.close()
