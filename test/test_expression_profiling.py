from __future__ import annotations

import argparse
import os
from contextlib import contextmanager

from activitysim.cli.run import add_run_args, run
from activitysim.core import timing, workflow


@contextmanager
def change_directory(path):
    """
    Context manager to temporarily change the current working directory.
    """
    old_cwd = os.getcwd()  # Save the current working directory
    try:
        os.chdir(path)  # Change to the new directory
        yield  # Execute the code within the 'with' block
    finally:
        os.chdir(old_cwd)  # Revert to the original directory


def test_expression_profiling_mtc():
    state = workflow.create_example("prototype_mtc", temp=True)

    state.settings.expression_profile = True
    state.run.all()

    # generate a summary of slower expression evaluation times
    # across all models and write to a file
    analyze = timing.AnalyzeEvalTiming(state)
    analyze.component_report(style=state.settings.expression_profile_style)
    analyze.subcomponent_report(style=state.settings.expression_profile_style)

    workdir = state.filesystem.working_dir
    outdir = workdir.joinpath(state.filesystem.output_dir)
    timestamp = state.get("run_timestamp", "unknown")
    assert timestamp != "unknown", "Run timestamp should not be 'unknown'"
    assert outdir.joinpath(
        f"log/expr-performance/{timestamp}/expression-timing-subcomponents.html"
    ).exists()
    assert outdir.joinpath(
        f"log/expr-performance/{timestamp}/expression-timing-components.html"
    ).exists()
    assert outdir.joinpath(
        f"log/expr-performance/{timestamp}/tour_mode_choice.work.simple_simulate.eval_nl.eval_utils.log"
    ).exists()


def test_expression_profiling_semcog():
    # testing a two zone system model
    state = workflow.create_example("production_semcog", temp=True)

    state.settings.expression_profile = True

    print("state.filesystem.working_dir=", state.filesystem.working_dir)
    # import the extensions module, which is located in the working directory
    with change_directory(state.filesystem.working_dir):
        import sys

        sys.path.insert(0, ".")
        import extensions  # noqa: F401

        sys.path.pop(0)

    state.run.all()

    # generate a summary of slower expression evaluation times
    # across all models and write to a file
    analyze = timing.AnalyzeEvalTiming(state)
    analyze.component_report(style=state.settings.expression_profile_style)
    analyze.subcomponent_report(style=state.settings.expression_profile_style)

    workdir = state.filesystem.working_dir
    outdir = workdir.joinpath(state.filesystem.output_dir)

    timestamp = state.get("run_timestamp", "unknown")
    assert timestamp != "unknown", "Run timestamp should not be 'unknown'"

    assert outdir.joinpath(
        f"expr-performance/{timestamp}/expression-timing-subcomponents.html"
    ).exists()
    assert outdir.joinpath(
        f"expr-performance/{timestamp}/expression-timing-components.html"
    ).exists()
    assert outdir.joinpath(
        f"expr-performance/{timestamp}/trip_destination.trip_num_1.atwork.compute_logsums.dp.preprocessor.trip_mode_choice_annotate_trips_preprocessor.log"
    ).exists()


def test_expression_profiling_mtc_mp():
    state = workflow.create_example("prototype_mtc", temp=True)
    # state = workflow.create_example("prototype_mtc", "/tmp/exprprof5")
    with state.filesystem.working_dir.joinpath("configs_mp", "settings.yaml").open(
        mode="a"
    ) as f:
        f.write("\n\nexpression_profile: true\n")

    args = [
        # "-c",
        # str(state.filesystem.working_dir.joinpath("configs-override")),
        "-c",
        str(state.filesystem.working_dir.joinpath("configs_mp")),
        "-c",
        str(state.filesystem.working_dir.joinpath("configs")),
        "-d",
        str(state.filesystem.working_dir.joinpath("data")),
        "-o",
        str(state.filesystem.working_dir.joinpath("output")),
    ]

    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args(args)
    run(args)

    ep_dir = state.filesystem.working_dir.joinpath("output", "log", "expr-performance")

    # list all the subdirectories in the output directory
    subdirectories = [item for item in ep_dir.iterdir() if item.is_dir()]

    # should only be one subdirectory with a timestamp
    assert (
        len(subdirectories) == 1
    ), "There should be exactly one subdirectory with a timestamp"

    timestamp = subdirectories[0].name
    base_dir = ep_dir.joinpath(timestamp)

    assert base_dir.joinpath("expression-timing-subcomponents.html").exists()
    assert base_dir.joinpath("expression-timing-components.html").exists()
    assert base_dir.joinpath(
        "mp_households_0-trip_destination.trip_num_1.atwork.compute_logsums.dp.preprocessor.trip_mode_choice_annotate_trips_preprocessor.log"
    ).exists()
