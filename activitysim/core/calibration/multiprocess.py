# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path
from typing import Any

import pandas as pd

from activitysim.core import workflow
from activitysim.core.configuration.top import MultiprocessStep

logger = logging.getLogger("calibration")

MP_INJECTABLES = [
    "data_dir",
    "configs_dir",
    "data_model_dir",
    "output_dir",
    "cache_dir",
    "settings_file_name",
    "imported_extensions",
    "run_timestamp",
    "run_id",
    "pipeline_file_name",
]


def _run_mp_single_component(
    state: workflow.State,
    component_name: str,
    run_label: str,
    restore_checkpoint: str,
    shared_data_buffers: dict,
) -> None:
    """Run a single component in multiprocess mode with explicit checkpoint control.

    This directly orchestrates the apportion → simulate → coalesce flow
    without going through run_multiprocess/get_run_list, giving us precise
    control over which checkpoint to apportion from. This is essential for
    calibration component re-runs where we must always restart from prior_step's
    state regardless of what other checkpoints exist in the pipeline.

    Parameters
    ----------
    state : workflow.State
    component_name : str
        The model component to run.
    run_label : str
        Unique name for this calibration execution's subprocess pipelines and
        checkpoint. It must differ from ``component_name`` so restoring the
        apportioned checkpoint does not cause ActivitySim to skip the model.
    restore_checkpoint : str
        The checkpoint name to restore from before apportioning.
        This should be the checkpoint representing prior_step's state.
    shared_data_buffers : dict
        Pre-allocated shared memory buffers for skims/shadow pricing.
    """
    from activitysim.core import mp_tasks

    from .execution import _restore_parent_state_from_pipeline

    # Determine slice info from original settings
    original_steps = state.settings.multiprocess_steps
    all_models = state.settings.models

    slice_info = None
    num_processes = state.settings.num_processes or 2
    chunk_size = state.settings.chunk_size or 0

    # Find which original step this component belongs to
    step_boundaries = []
    for i, step in enumerate(original_steps):
        step_boundaries.append(all_models.index(step.begin))
    step_boundaries.append(len(all_models))

    component_idx = all_models.index(component_name)
    for i, step in enumerate(original_steps):
        if step_boundaries[i] <= component_idx < step_boundaries[i + 1]:
            if step.slice:
                slice_info = step.slice.model_dump()
            if step.num_processes:
                num_processes = step.num_processes
            if step.chunk_size:
                chunk_size = step.chunk_size
            break

    # Build step_info dict matching what mp_tasks functions expect
    step_info = {
        "name": run_label,
        "models": [component_name],
        "num_processes": num_processes,
        "chunk_size": chunk_size,
        "step_num": 0,
        "slice": slice_info,
        "last_checkpoint_in_previous_multiprocess_step": restore_checkpoint,
    }

    injectables = _build_calibration_injectables(state)

    if num_processes == 1:
        sub_proc_names = [run_label]
    else:
        sub_proc_names = [f"{run_label}_{i}" for i in range(num_processes)]

    fail_fast = state.settings.fail_fast

    # Apportion pipeline (split tables across sub-processes)
    if num_processes > 1 and slice_info is not None:
        mp_tasks.run_sub_task(
            state,
            multiprocessing.Process(
                target=mp_tasks.mp_apportion_pipeline,
                name=f"{run_label}_apportion",
                args=(injectables, sub_proc_names, step_info),
            ),
        )

    # For multi-process runs, subprocesses must restore from the apportioned
    # pipeline (which has one checkpoint). Use LAST_CHECKPOINT so they don't
    # overwrite the apportioned data with a fresh pipeline.
    # For single-process runs (no apportion), use restore_checkpoint to resume
    # from the correct point in the main pipeline.
    if num_processes > 1:
        sim_resume_after = "_"  # LAST_CHECKPOINT in apportioned sub-pipeline
    else:
        sim_resume_after = restore_checkpoint

    # Run simulations in sub-processes
    completed = mp_tasks.run_sub_simulations(
        state,
        injectables,
        shared_data_buffers,
        step_info,
        sub_proc_names,
        sim_resume_after,
        [],  # previously_completed
        fail_fast,
    )

    if len(completed) != num_processes:
        from activitysim.core.exceptions import SubprocessError

        raise SubprocessError(
            f"{num_processes - len(completed)} processes failed in "
            f"calibration step {component_name}"
        )

    # Coalesce sub-process pipelines back into main pipeline
    if num_processes > 1 and slice_info is not None:
        mp_tasks.run_sub_task(
            state,
            multiprocessing.Process(
                target=mp_tasks.mp_coalesce_pipelines,
                name=f"{run_label}_coalesce",
                args=(injectables, sub_proc_names, slice_info),
            ),
        )

    # Restore coalesced results into parent state
    _restore_parent_state_from_pipeline(state)


def _restore_from_subprocess_pipelines(
    state: workflow.State, resume_after: str
) -> bool:
    """Restore state from subprocess pipelines at a specific model checkpoint.

    Subprocess pipelines retain model-level checkpoints that don't exist in
    the main pipeline.  This performs a "coalesce at checkpoint" — reading
    mirrored tables from one subprocess and concatenating sliced tables from
    all subprocesses at the specified checkpoint name.

    Parameters
    ----------
    state : workflow.State
    resume_after : str
        Model-level checkpoint name to restore from.

    Returns
    -------
    bool
        True if the restore succeeded; False if subprocess pipelines don't
        exist or don't contain the requested checkpoint.
    """
    from activitysim.core.workflow.checkpoint import (
        CHECKPOINT_NAME,
        CHECKPOINT_TABLE_NAME,
        HdfStore,
        NON_TABLE_COLUMNS,
        ParquetStore,
    )

    all_models = state.settings.models
    mp_steps = state.settings.multiprocess_steps
    if not mp_steps or resume_after not in all_models:
        return False

    # Find the multiprocess step containing resume_after
    resume_idx = all_models.index(resume_after)
    step_boundaries = [all_models.index(s.begin) for s in mp_steps]
    step_boundaries.append(len(all_models))

    enclosing_step = None
    num_processes = state.settings.num_processes or 2
    slice_info = None
    for i, step in enumerate(mp_steps):
        if step_boundaries[i] <= resume_idx < step_boundaries[i + 1]:
            enclosing_step = step
            if step.num_processes:
                num_processes = step.num_processes
            if step.slice:
                slice_info = (
                    step.slice.model_dump()
                    if hasattr(step.slice, "model_dump")
                    else step.slice
                )
            break

    if enclosing_step is None or num_processes <= 1:
        return False

    # Build subprocess pipeline file paths
    step_name = enclosing_step.name
    pipeline_file_name = state.filesystem.pipeline_file_name
    sub_proc_names = [f"{step_name}_{i}" for i in range(num_processes)]

    def _subprocess_path(proc_name):
        base = state.get_output_file_path(pipeline_file_name, prefix=proc_name)
        if state.settings.checkpoint_format == "hdf":
            return base
        pq = Path(str(base)).with_suffix(ParquetStore.extension)
        return pq if pq.exists() else base

    first_path = _subprocess_path(sub_proc_names[0])
    if not first_path.exists():
        return False

    # Open first subprocess pipeline and verify checkpoint exists
    if state.settings.checkpoint_format == "hdf":
        first_store = HdfStore(first_path, mode="r")
    else:
        first_store = ParquetStore(first_path, mode="r")

    try:
        cp_names = first_store.list_checkpoint_names()
        if resume_after not in cp_names:
            return False

        # Read checkpoint row to get table→checkpoint mapping
        cp_df = first_store.get_dataframe(CHECKPOINT_TABLE_NAME)
        cp_row = cp_df[cp_df[CHECKPOINT_NAME] == resume_after].iloc[-1]

        table_map = {}
        for col in cp_row.index:
            if col not in NON_TABLE_COLUMNS and cp_row[col]:
                table_map[col] = cp_row[col]

        # Read all tables from first subprocess at this checkpoint
        tables = {}
        for table_name, cp_for_table in table_map.items():
            try:
                tables[table_name] = first_store.get_dataframe(table_name, cp_for_table)
            except (FileNotFoundError, KeyError):
                logger.warning(
                    f"calibration: subprocess pipeline missing table "
                    f"{table_name} at {cp_for_table}"
                )
    finally:
        first_store.close()

    if not tables:
        return False

    # Determine sliced tables that need concatenation across processes
    sliced_table_names = set(slice_info.get("tables", [])) if slice_info else set()

    # Read sliced tables from remaining subprocesses and concatenate
    if num_processes > 1 and sliced_table_names:
        omnibus = {t: [tables[t]] for t in sliced_table_names if t in tables}

        for proc_name in sub_proc_names[1:]:
            proc_path = _subprocess_path(proc_name)
            if not proc_path.exists():
                logger.warning(
                    f"calibration: subprocess pipeline not found: {proc_path}"
                )
                return False

            if state.settings.checkpoint_format == "hdf":
                proc_store = HdfStore(proc_path, mode="r")
            else:
                proc_store = ParquetStore(proc_path, mode="r")

            try:
                proc_cp_df = proc_store.get_dataframe(CHECKPOINT_TABLE_NAME)
                proc_row = proc_cp_df[proc_cp_df[CHECKPOINT_NAME] == resume_after].iloc[
                    -1
                ]

                for table_name in list(omnibus.keys()):
                    cp_for_table = proc_row.get(table_name, "")
                    if cp_for_table:
                        omnibus[table_name].append(
                            proc_store.get_dataframe(table_name, cp_for_table)
                        )
            finally:
                proc_store.close()

        # Replace sliced tables with concatenated versions
        for table_name, dfs in omnibus.items():
            tables[table_name] = pd.concat(dfs, sort=False)

    # Load into parent state
    prior_rng_channels = list(state.get_injectable("rng_channels", []))
    prior_index_to_channel = (
        dict(state.rng().index_to_channel)
        if hasattr(state.rng(), "index_to_channel")
        else {}
    )

    state.init_state()
    if state.checkpoint.store_is_open():
        state.checkpoint.close_store()
    state.checkpoint.open_store(overwrite=False)

    for table_name, df in tables.items():
        state.add_table(table_name, df)

    _reregister_rng_channels(state, prior_rng_channels, prior_index_to_channel)

    # Mark all tables dirty for subsequent checkpoint.add
    for table_name in list(state.existing_table_names):
        state.existing_table_status[table_name] = True

    logger.info(
        "calibration: restored %d tables from subprocess pipelines at "
        "checkpoint '%s'",
        len(tables),
        resume_after,
    )
    return True


def _run_multiprocess_with_overrides(
    state: workflow.State,
    models: list[str],
    resume_after: str | None,
    shared_data_buffers: dict | None = None,
    can_reuse_subprocs: bool = False,
) -> None:
    """Run multiprocess with temporary settings overrides for calibration passes.

    Parameters
    ----------
    can_reuse_subprocs : bool, default False
        When True, subprocess pipelines from a prior run are assumed to exist
        and contain the ``resume_after`` checkpoint.  Breadcrumbs are written
        so that ``get_run_list`` populates ``step_info["resume_after"]``,
        apportion is skipped (reusing existing subprocess pipelines), and
        subprocesses resume from their model-level checkpoint — skipping
        already-completed models.
    """
    from collections import OrderedDict

    from activitysim.core import mp_tasks

    original_models = state.settings.models
    original_mp_steps = state.settings.multiprocess_steps
    original_resume_after = state.settings.resume_after

    # Build valid multiprocess_steps for the requested model subset.
    calibration_mp_steps = _build_calibration_mp_steps(
        models=models,
        original_steps=original_mp_steps,
        all_models=original_models,
    )

    state.settings.models = models
    state.settings.multiprocess_steps = calibration_mp_steps

    if can_reuse_subprocs and resume_after:
        # Include resume_after in the models list so get_breadcrumbs can
        # locate the step containing it.  Subprocesses will skip this model
        # (it's already checkpointed in their pipeline) and run the rest.
        models = [resume_after] + models

        # Rebuild steps with resume_after included.
        calibration_mp_steps = _build_calibration_mp_steps(
            models=models,
            original_steps=original_mp_steps,
            all_models=original_models,
        )
        state.settings.models = models
        state.settings.multiprocess_steps = calibration_mp_steps
        state.settings.resume_after = resume_after

        # Write minimal breadcrumbs indicating the step containing
        # resume_after has completed apportion (so it's skipped) but
        # simulate/coalesce need re-running.
        breadcrumbs = OrderedDict()
        for step in calibration_mp_steps:
            step_dict = {"name": step.name, "apportion": True}
            breadcrumbs[step.name] = step_dict
            # Find the step containing resume_after
            all_models = state.settings.models
            if resume_after in all_models:
                step_begin = all_models.index(step.begin)
                step_models_in_step = [
                    m for m in all_models[step_begin:] if m in models
                ]
                if resume_after in step_models_in_step:
                    # This step contains resume_after — stop here.
                    # get_breadcrumbs will mark simulate/coalesce for re-run.
                    break

        mp_tasks.write_breadcrumbs(state, breadcrumbs)
    else:
        # No reuse: calibration manages pipeline state externally via
        # _restore_parent_state_from_pipeline and checkpoint.add, so the MP
        # system's breadcrumb-based resume logic must not be triggered.
        state.settings.resume_after = None

    try:
        injectables = _build_calibration_injectables(state)
        mp_tasks.run_multiprocess(
            state,
            injectables,
            shared_data_buffers=shared_data_buffers,
            skip_final_checkpoint=True,
            force_resume=resume_after is not None and not can_reuse_subprocs,
        )
    finally:
        state.settings.models = original_models
        state.settings.resume_after = original_resume_after
        state.settings.multiprocess_steps = original_mp_steps


def _reregister_rng_channels(
    state: workflow.State,
    prior_channels: list[str],
    prior_index_to_channel: dict[str, str] = None,
) -> None:
    """Re-register RNG channels that were lost during init_state()."""
    current_channels = set(state.get_injectable("rng_channels", []))
    for channel_name in prior_channels:
        if channel_name not in state.rng().channels and state.is_table(channel_name):
            try:
                state.rng().add_channel(channel_name, state.get_dataframe(channel_name))
            except Exception:
                pass
        if channel_name in state.rng().channels:
            current_channels.add(channel_name)
    # For channels whose tables don't exist at the restored checkpoint,
    # register an empty channel.  Do NOT pre-load from a later checkpoint
    # in the store — that data may include modifications from downstream
    # models and would pollute the pre-model state.  The empty channel
    # allows the model's normal table factory to create the table fresh
    # and extend the channel without hitting the disjoint-index assertion.
    if prior_index_to_channel:
        for index_name, channel_name in prior_index_to_channel.items():
            if index_name not in state.rng().index_to_channel:
                if channel_name not in state.rng().channels:
                    if state.is_table(channel_name):
                        # The channel may have been registered dynamically and
                        # therefore be absent from the rng_channels injectable.
                        # Populate it from the exact table restored from the
                        # target checkpoint rather than creating an empty
                        # channel for an existing domain.
                        state.rng().add_channel(
                            channel_name, state.get_dataframe(channel_name)
                        )
                    else:
                        empty_df = pd.DataFrame(
                            index=pd.Index([], dtype="int64", name=index_name)
                        )
                        state.rng().add_channel(channel_name, empty_df)
                else:
                    state.rng().index_to_channel[index_name] = channel_name
            if channel_name in state.rng().channels:
                current_channels.add(channel_name)
    state.add_injectable("rng_channels", list(current_channels))


def _initialize_mp_shared_resources(state: workflow.State) -> dict:
    """Allocate shared data buffers (skims, shadow pricing) once for reuse.

    This mirrors the allocation logic in mp_tasks.run_multiprocess but
    is called once at calibration start rather than on every sub-run.
    """
    from activitysim.core import mp_tasks, tracing

    shared_data_buffers = {}
    sharrow_enabled = state.settings.sharrow

    t0 = tracing.print_elapsed_time()
    if not sharrow_enabled:
        shared_data_buffers.update(mp_tasks.allocate_shared_skim_buffers(state))
        t0 = tracing.print_elapsed_time("calibration: allocate shared skim buffer", t0)

    shared_data_buffers.update(mp_tasks.allocate_shared_shadow_pricing_buffers(state))
    t0 = tracing.print_elapsed_time(
        "calibration: allocate shared shadow_pricing buffer", t0
    )

    shared_data_buffers.update(
        mp_tasks.allocate_shared_shadow_pricing_buffers_choice(state)
    )
    t0 = tracing.print_elapsed_time(
        "calibration: allocate shared shadow_pricing choice buffer", t0
    )

    # Load skim data into the shared buffers.
    if sharrow_enabled:
        shared_data_buffers["skim_dataset"] = "sh.Dataset:skim_dataset"
        from activitysim.core import flow, skim_dataset  # noqa: F401

        state.get_injectable("skim_dataset")
    else:
        if len(shared_data_buffers) > 0:
            injectables = _build_calibration_injectables(state)
            mp_tasks.run_sub_task(
                state,
                multiprocessing.Process(
                    target=mp_tasks.mp_setup_skims,
                    name="mp_setup_skims_calibration",
                    args=(injectables,),
                    kwargs=shared_data_buffers,
                ),
            )

    # Make skims available in the parent process for expression evaluation.
    state.add_injectable("data_buffers", shared_data_buffers)
    try:
        state.get_injectable("network_los")
    except Exception:
        logger.warning(
            "calibration: could not resolve network_los in parent process; "
            "skim-dependent expressions may fail"
        )

    return shared_data_buffers


def _build_calibration_mp_steps(
    models: list[str],
    original_steps: list[MultiprocessStep],
    all_models: list[str],
) -> list[MultiprocessStep]:
    """Build valid MultiprocessStep objects for a calibration model subset.

    The key challenge is that get_run_list() in mp_tasks requires:
    - The first step's begin == models[0]
    - Steps are ordered and non-overlapping
    - Each step's begin is in the models list

    We intersect the original multiprocess_steps with the requested model
    subset and construct new steps that satisfy these constraints.
    """
    if not models:
        return []

    # Determine which original step each model in the full list belongs to.
    # Build a mapping: model_name -> original step index
    model_to_step: dict[str, int] = {}
    step_boundaries = []
    for i, step in enumerate(original_steps):
        begin_idx = all_models.index(step.begin)
        step_boundaries.append(begin_idx)
    step_boundaries.append(len(all_models))

    for i, step in enumerate(original_steps):
        for model_idx in range(step_boundaries[i], step_boundaries[i + 1]):
            model_to_step[all_models[model_idx]] = i

    # Group the requested models by their original step
    from collections import OrderedDict

    step_model_groups: OrderedDict[int, list[str]] = OrderedDict()
    for model in models:
        step_idx = model_to_step.get(model)
        if step_idx is None:
            continue
        step_model_groups.setdefault(step_idx, []).append(model)

    # Build new MultiprocessStep for each group.
    # Some original steps (e.g. mp_initialize) omit num_processes, slice,
    # and chunk_size — these default to None on MultiprocessStep and
    # get_run_list() applies global defaults when they are absent.
    # Step names include the first model to ensure uniqueness across multiple
    # intermediate runs that draw from the same original step.
    new_steps = []
    for step_idx, step_models in step_model_groups.items():
        orig_step = original_steps[step_idx]
        kwargs: dict[str, Any] = {
            "name": orig_step.name,
            "begin": step_models[0],
        }
        if orig_step.num_processes is not None:
            kwargs["num_processes"] = orig_step.num_processes
        if orig_step.slice is not None:
            kwargs["slice"] = orig_step.slice
        if orig_step.chunk_size is not None:
            kwargs["chunk_size"] = orig_step.chunk_size
        new_steps.append(MultiprocessStep(**kwargs))

    return new_steps


def _build_calibration_injectables(state: workflow.State) -> dict:
    """Build the injectables dict for multiprocess sub-processes."""
    injectables = {}
    for key in MP_INJECTABLES:
        try:
            injectables[key] = state.get_injectable(key)
        except KeyError:
            pass
    injectables["settings"] = state.settings
    return injectables
