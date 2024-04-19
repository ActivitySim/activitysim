# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.core import expressions, tracing, workflow
from activitysim.core.configuration import PydanticBase

"""
Code for annotating tables
"""

logger = logging.getLogger(__name__)


def annotate_households(
    state: workflow.State,
    model_settings: dict | PydanticBase,
    trace_label: str,
    locals_dict: dict | None = None,
):
    """
    Add columns to the households table in the pipeline according to spec.

    Parameters
    ----------
    model_settings : dict
    trace_label : str
    """
    if isinstance(model_settings, PydanticBase):
        model_settings = model_settings.dict()
    if locals_dict is None:
        locals_dict = {}
    households = state.get_dataframe("households")
    expressions.assign_columns(
        state,
        df=households,
        model_settings=model_settings.get("annotate_households"),
        locals_dict=locals_dict,
        trace_label=tracing.extend_trace_label(trace_label, "annotate_households"),
    )
    state.add_table("households", households)


def annotate_persons(
    state: workflow.State,
    model_settings: dict | PydanticBase,
    trace_label: str,
    locals_dict: dict | None = None,
):
    """
    Add columns to the persons table in the pipeline according to spec.

    Parameters
    ----------
    model_settings : dict
    trace_label : str
    """
    if isinstance(model_settings, PydanticBase):
        model_settings = model_settings.dict()
    if locals_dict is None:
        locals_dict = {}
    persons = state.get_dataframe("persons")
    expressions.assign_columns(
        state,
        df=persons,
        model_settings=model_settings.get("annotate_persons"),
        locals_dict=locals_dict,
        trace_label=tracing.extend_trace_label(trace_label, "annotate_persons"),
    )
    state.add_table("persons", persons)


def annotate_tours(
    state: workflow.State,
    model_settings: dict | PydanticBase,
    trace_label: str,
    locals_dict: dict | None = None,
):
    """
    Add columns to the tours table in the pipeline according to spec.

    Parameters
    ----------
    state : workflow.State
    model_settings : dict or PydanticBase
    trace_label : str
    locals_dict : dict, optional
    """
    if isinstance(model_settings, PydanticBase):
        model_settings = model_settings.dict()
    if locals_dict is None:
        locals_dict = {}
    tours = state.get_dataframe("tours")
    expressions.assign_columns(
        state,
        df=tours,
        model_settings=model_settings.get("annotate_tours"),
        locals_dict=locals_dict,
        trace_label=tracing.extend_trace_label(trace_label, "annotate_tours"),
    )
    state.add_table("tours", tours)


def annotate_trips(
    state: workflow.State,
    model_settings: dict | PydanticBase,
    trace_label: str,
    locals_dict=None,
):
    """
    Add columns to the trips table in the pipeline according to spec.

    Parameters
    ----------
    state : workflow.State
    model_settings : dict or PydanticBase
    trace_label : str
    locals_dict : dict, optional
    """
    if isinstance(model_settings, PydanticBase):
        model_settings = model_settings.dict()
    if locals_dict is None:
        locals_dict = {}
    trips = state.get_dataframe("trips")
    expressions.assign_columns(
        state,
        df=trips,
        model_settings=model_settings.get("annotate_trips"),
        locals_dict=locals_dict,
        trace_label=tracing.extend_trace_label(trace_label, "annotate_trips"),
    )
    state.add_table("trips", trips)
