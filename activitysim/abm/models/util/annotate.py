import logging

from activitysim.core import expressions, tracing, workflow

# ActivitySim
# See full license in LICENSE.txt.

"""
Code for annotating tables
"""

logger = logging.getLogger(__name__)


def annotate_tours(whale: workflow.Whale, model_settings, trace_label):
    """
    Add columns to the tours table in the pipeline according to spec.

    Parameters
    ----------
    model_settings : dict
    trace_label : str
    """
    tours = whale.get_table("tours")
    expressions.assign_columns(
        whale,
        df=tours,
        model_settings=model_settings.get("annotate_tours"),
        trace_label=tracing.extend_trace_label(trace_label, "annotate_tours"),
    )
    whale.add_table("tours", tours)


def annotate_trips(whale: workflow.Whale, model_settings, trace_label):
    """
    Add columns to the trips table in the pipeline according to spec.

    Parameters
    ----------
    model_settings : dict
    trace_label : str
    """
    trips = whale.get_table("trips")
    expressions.assign_columns(
        whale,
        df=trips,
        model_settings=model_settings.get("annotate_trips"),
        trace_label=tracing.extend_trace_label(trace_label, "annotate_trips"),
    )
    whale.add_table("trips", trips)
