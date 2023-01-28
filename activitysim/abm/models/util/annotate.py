# ActivitySim
# See full license in LICENSE.txt.
import pandas as pd
import logging

from activitysim.core import expressions
from activitysim.core import tracing
from activitysim.core import inject
from activitysim.core import pipeline

"""
Code for annotating tables
"""

logger = logging.getLogger(__name__)


def annotate_tours(model_settings, trace_label):
    """
    Add columns to the tours table in the pipeline according to spec.

    Parameters
    ----------
    model_settings : dict
    trace_label : str
    """
    tours = inject.get_table("tours").to_frame()
    expressions.assign_columns(
        df=tours,
        model_settings=model_settings.get("annotate_tours"),
        trace_label=tracing.extend_trace_label(trace_label, "annotate_tours"),
    )
    pipeline.replace_table("tours", tours)


def annotate_trips(model_settings, trace_label):
    """
    Add columns to the trips table in the pipeline according to spec.

    Parameters
    ----------
    model_settings : dict
    trace_label : str
    """
    trips = inject.get_table("trips").to_frame()
    expressions.assign_columns(
        df=trips,
        model_settings=model_settings.get("annotate_trips"),
        trace_label=tracing.extend_trace_label(trace_label, "annotate_trips"),
    )
    pipeline.replace_table("trips", trips)
