from __future__ import annotations

import os
from contextlib import contextmanager

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
    assert outdir.joinpath(
        "log/expr-performance/expression-timing-subcomponents.html"
    ).exists()
    assert outdir.joinpath(
        "log/expr-performance/expression-timing-components.html"
    ).exists()
    assert outdir.joinpath(
        "log/expr-performance/tour_mode_choice.work.simple_simulate.eval_nl.eval_utils.log"
    ).exists()


def test_expression_profiling_semcog():
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
    assert outdir.joinpath(
        "expr-performance/expression-timing-subcomponents.html"
    ).exists()
    assert outdir.joinpath(
        "expr-performance/expression-timing-components.html"
    ).exists()
    assert outdir.joinpath(
        "expr-performance/trip_destination.trip_num_1.atwork.compute_logsums.dp.preprocessor.trip_mode_choice_annotate_trips_preprocessor.log"
    ).exists()
