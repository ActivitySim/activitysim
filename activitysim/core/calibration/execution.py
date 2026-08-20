# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from activitysim.core import workflow
from activitysim.core.workflow.checkpoint import (
    CHECKPOINT_NAME,
    CHECKPOINT_TABLE_NAME,
    LAST_CHECKPOINT,
    NON_TABLE_COLUMNS,
)

from .multiprocess import (
    _reregister_rng_channels,
    _restore_from_subprocess_pipelines,
    _run_multiprocess_with_overrides,
)
logger = logging.getLogger("calibration")


def _run_in_configured_mode(
    state: workflow.State,
    models: list[str],
    resume_after: str | None,
    shared_data_buffers: dict | None = None,
) -> None:
    """Run models using the same single/multiprocess mode as the parent run."""
    if not models:
        return

    extra_models = _prep_model_data(state, resume_after=resume_after)
    if extra_models:
        # Models from the step beginning through resume_after must run first
        # to recreate the correct intermediate state (since no model-level
        # checkpoint existed for resume_after).
        models = extra_models + models

    if state.settings.multiprocess:
        # Write the restored state as a checkpoint so LAST_CHECKPOINT on disk
        # reflects the correct (clean) state for the apportion subprocess.
        # When extra_models were prepended, the state is from a PRIOR step
        # (not the actual resume_after point) — use a non-conflicting name
        # so it won't be mistakenly loaded as the resume_after state on a
        # subsequent restart.
        if extra_models:
            state.checkpoint.add("_calibration_staging")
        else:
            state.checkpoint.add(resume_after or models[0])
        state.checkpoint.close_store()

        # When subprocess pipelines from a prior run already have the
        # resume_after checkpoint (Path 2 in _prep_model_data), subprocesses
        # can skip models before resume_after by reusing those pipelines
        # instead of freshly apportioning.  Signal this by passing
        # can_reuse_subprocs=True.
        can_reuse = not extra_models and resume_after is not None

        _run_multiprocess_with_overrides(
            state,
            models=models,
            resume_after=resume_after,
            shared_data_buffers=shared_data_buffers,
            can_reuse_subprocs=can_reuse,
        )
        # After multiprocess completes, the coalesced pipeline exists on disk.
        # Restore it into the parent process state so tables are accessible
        # for calibration expression evaluation.
        _restore_parent_state_from_pipeline(state)
        # Add a checkpoint named after the last model so that model-name
        # references (e.g. _prior_step_name, resume_after on global_iter > 1)
        # resolve correctly. Without this, only the step-level coalesce name
        # exists in the pipeline.
        state.checkpoint.add(models[-1])
        return

    # State is already at the correct point from _prep_model_data above.
    # Do NOT call _prep_model_data again — the second call would build its
    # table_checkpoint_map from the now-truncated in-memory checkpoint history,
    # losing references to tables created after resume_after (e.g. vehicles).
    state.checkpoint.add(resume_after or models[0])
    for model in models:
        state.run.by_name(model)
    # Ensure final model's state is persisted even if should_save_checkpoint
    # returned False for it — _calibrate_component needs to restore to it.
    if models:
        state.checkpoint.add(models[-1])


def _prep_model_data(state, resume_after=None):
    """Restore the pipeline to the correct state before running models.

    Resolution priority:
    1. Direct model-level checkpoint in the main pipeline (fastest, exact).
    2. Subprocess pipelines from a prior multiprocess run — performs a
       "coalesce at specific checkpoint" to recover the exact intermediate
       state without re-running anything.
    3. Previous step checkpoint + re-run models from step begin through
       resume_after (slowest but always works).

    Returns
    -------
    list[str]
        Models that must be prepended to the caller's models list to reach
        the correct state at ``resume_after``.  Empty when the exact
        checkpoint was found and restored directly (paths 1 or 2).
    """
    if resume_after:
        try:
            if state.checkpoint.store_is_open():
                checkpoint_names = [
                    cp.get("checkpoint_name", "") for cp in state.checkpoint.checkpoints
                ]
            else:
                from activitysim.core.workflow.checkpoint import HdfStore, ParquetStore

                pipeline_path = Path(state.checkpoint.default_pipeline_file_path())
                if state.settings.checkpoint_format == "hdf":
                    store = HdfStore(pipeline_path, mode="r")
                else:
                    store = ParquetStore(pipeline_path, mode="r")
                try:
                    checkpoint_names = store.list_checkpoint_names()
                finally:
                    store.close()

            # Path 1: direct model-level checkpoint in main pipeline
            if resume_after in checkpoint_names:
                _restore_parent_state_from_pipeline(state, checkpoint_name=resume_after)
                return []

            # Path 2: subprocess pipelines (model-level checkpoints preserved)
            if _restore_from_subprocess_pipelines(state, resume_after):
                return []

            # Path 3: restore from previous step and re-run
            all_models = state.settings.models
            mp_steps = state.settings.multiprocess_steps
            if mp_steps and resume_after in all_models:
                resume_idx = all_models.index(resume_after)
                step_boundaries = [all_models.index(s.begin) for s in mp_steps]
                step_boundaries.append(len(all_models))
                for i, step in enumerate(mp_steps):
                    if step_boundaries[i] <= resume_idx < step_boundaries[i + 1]:
                        if i > 0 and mp_steps[i - 1].name in checkpoint_names:
                            _restore_parent_state_from_pipeline(
                                state, checkpoint_name=mp_steps[i - 1].name
                            )
                            step_begin_idx = step_boundaries[i]
                            extra_models = all_models[step_begin_idx : resume_idx + 1]
                            return extra_models
                        elif i == 0:
                            _restore_parent_state_from_pipeline(
                                state, checkpoint_name="_"
                            )
                            extra_models = all_models[: resume_idx + 1]
                            return extra_models
                        break
        except Exception:
            logger.warning(
                "calibration: could not restore from checkpoint %r, "
                "falling back to LAST_CHECKPOINT",
                resume_after,
            )

    # Fallback: load LAST_CHECKPOINT (appropriate after a coalesce that
    # only ran the desired models)
    _restore_parent_state_from_pipeline(state)
    return []


def _restore_parent_state_from_pipeline(
    state: workflow.State, checkpoint_name: str = "_"
) -> None:
    """Restore pipeline tables into the parent process state.

    After a multiprocess run or calibration rewind, the parent's in-memory
    state may contain tables created after the requested checkpoint.  Remove
    those tables before loading the checkpoint so the restored state exactly
    represents that point in the model sequence.

    Parameters
    ----------
    checkpoint_name : str, default "_"
        The checkpoint to restore from.  Use a model-level checkpoint name
        (e.g. the prior step name) to get the exact state at that point,
        avoiding pollution from downstream models that may have added rows
        to shared tables like ``tours``.  The default ``"_"`` loads the
        last checkpoint, which is appropriate immediately after a coalesce
        that only ran the desired models.

    All tables are explicitly re-checkpointed so that subsequent apportion
    subprocesses can load them from a direct file path without relying on
    checkpoint backtracking through potentially ambiguous checkpoint history.
    """
    # Read the target checkpoint manifest before restore truncates checkpoint
    # history. Tables represented by columns in this manifest are pipeline-
    # managed; a false/empty value means the table did not yet exist.
    checkpoints = state.checkpoint.store.get_dataframe(CHECKPOINT_TABLE_NAME)
    if checkpoint_name == LAST_CHECKPOINT:
        target_checkpoint = checkpoints.iloc[-1]
    else:
        matching_checkpoints = checkpoints[
            checkpoints[CHECKPOINT_NAME] == checkpoint_name
        ]
        if matching_checkpoints.empty:
            # Let checkpoint.restore raise its normal, more specific exception.
            target_checkpoint = None
        else:
            target_checkpoint = matching_checkpoints.iloc[-1]

    stale_tables: set[str] = set()
    if target_checkpoint is not None:
        pipeline_tables = set(checkpoints.columns) - set(NON_TABLE_COLUMNS)
        target_tables = {
            table_name
            for table_name in pipeline_tables
            if pd.notna(target_checkpoint[table_name])
            and bool(target_checkpoint[table_name])
        }
        stale_tables = pipeline_tables - target_tables

    # Capture RNG state before restore — models may have dynamically added
    # channels that aren't in the default rng_channels injectable. Do not carry
    # channels for stale tables across the rewind; their table factories will
    # register fresh channels when normal downstream execution recreates them.
    prior_rng_channels = [
        channel_name
        for channel_name in state.get_injectable("rng_channels", [])
        if channel_name not in stale_tables
    ]
    prior_index_to_channel = {
        index_name: channel_name
        for index_name, channel_name in getattr(
            state.rng(), "index_to_channel", {}
        ).items()
        if channel_name not in stale_tables
    }

    for table_name in stale_tables:
        # Use State.drop rather than drop_table: a prior restore may have reset
        # salient-table metadata while leaving the cached DataFrame in context.
        if table_name in state:
            state.drop(table_name)
            logger.debug(
                "calibration: exact restore removed post-checkpoint table '%s'",
                table_name,
            )
        if table_name in state.rng().channels:
            state.rng().drop_channel(table_name)

    # checkpoint.load uses this injectable to decide which restored tables get
    # RNG channels. Remove stale channel names before it performs that work.
    state.add_injectable("rng_channels", prior_rng_channels)

    if state.checkpoint.store_is_open():
        state.checkpoint.close_store()
    state.checkpoint.restore(resume_after=checkpoint_name)

    _reregister_rng_channels(state, prior_rng_channels, prior_index_to_channel)

    # After restore, all tables are clean (status=False). Mark them dirty so
    # the next checkpoint.add() writes them to disk at a known checkpoint name.
    # This ensures apportion subprocesses find table files at a single,
    # unambiguous checkpoint rather than needing to backtrack through history.
    for table_name in list(state.existing_table_names):
        state.existing_table_status[table_name] = True
