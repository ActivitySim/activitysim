from __future__ import annotations

import logging

from activitysim.core import expressions, tracing, workflow

# ActivitySim
# See full license in LICENSE.txt.

"""
Code for annotating tables
"""

logger = logging.getLogger(__name__)


def annotate_tours(state: workflow.State, model_settings, trace_label):
    """
    Add columns to the tours table in the pipeline according to spec.

    Parameters
    ----------
    model_settings : dict
    trace_label : str
    """
    tours = state.get_dataframe("tours")
    expressions.assign_columns(
        state,
        df=tours,
        model_settings=model_settings.get("annotate_tours"),
        trace_label=tracing.extend_trace_label(trace_label, "annotate_tours"),
    )
    state.add_table("tours", tours)


def annotate_trips(state: workflow.State, model_settings, trace_label):
    """
    Add columns to the trips table in the pipeline according to spec.

    Parameters
    ----------
    model_settings : dict
    trace_label : str
    """
    trips = state.get_dataframe("trips")
    expressions.assign_columns(
        state,
        df=trips,
        model_settings=model_settings.get("annotate_trips"),
        trace_label=tracing.extend_trace_label(trace_label, "annotate_trips"),
    )
    state.add_table("trips", trips)
