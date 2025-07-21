from __future__ import annotations

import activitysim.abm  # register components
from activitysim.core import timing, workflow


def test_expression_profiling():
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
