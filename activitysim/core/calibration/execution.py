# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
from pathlib import Path

from activitysim.core import workflow

from .multiprocess import (
    _reregister_rng_channels,
    _restore_from_subprocess_pipelines,
    _run_multiprocess_with_overrides,
)
from .settings import read_calibration_settings

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


def _invalidate_derived_tables(state: workflow.State) -> None:
    """Drop factory-produced tables that may be stale after a calibration restore.

    When a calibrated model (e.g. auto_ownership) changes a table that a
    @workflow.table factory depends on (e.g. vehicles depends on households),
    the checkpoint may contain a stale version of that factory table.  Dropping
    it forces the factory to regenerate from current data on next access.

    Auto-detection rule: invalidate any table that is (a) registered as a
    @workflow.table factory, (b) has DataFrame parameters (= table dependencies),
    and (c) is in RANDOM_CHANNELS.  This currently matches only 'vehicles' but
    will automatically cover future factory tables with the same pattern.
    """
    settings = read_calibration_settings(state)
    if not settings:
        return

    tables_to_invalidate = settings.run.invalidate_tables
    if tables_to_invalidate is None:
        # Vehicles needs regeneration only when calibration changes
        # households.auto_ownership. Downstream calibration components must
        # retain vehicle_type_choice's vehicle attributes.
        tables_to_invalidate = (
            ["vehicles"]
            if "auto_ownership_simulate" in settings.run.calibrate_models
            else []
        )

    logger.debug(
        "calibration: tables detected for invalidation: %s", tables_to_invalidate
    )
    tables_before = set(state.existing_table_names)

    for table_name in tables_to_invalidate:
        if state.is_table(table_name):
            state.drop_table(table_name)
            state.rng().drop_channel(table_name)
            state.get_dataframe(table_name, as_copy=False)
            logger.debug("calibration: invalidated derived table '%s'", table_name)

    tables_after = set(state.existing_table_names)
    lost = tables_before - tables_after - set(tables_to_invalidate)
    if lost:
        logger.error(
            "calibration: tables unexpectedly removed during invalidation: %s", lost
        )


def _restore_parent_state_from_pipeline(
    state: workflow.State, checkpoint_name: str = "_"
) -> None:
    """Restore pipeline tables into the parent process state.

    After a multiprocess run, the parent's in-memory state is stale.
    This loads a specific checkpoint from the pipeline store so that
    calibration expressions can evaluate against model outputs.

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
    # Capture RNG state before restore — models may have dynamically
    # added channels (e.g. "vehicles") that aren't in the default
    # rng_channels injectable and would be lost by init_state().
    prior_rng_channels = list(state.get_injectable("rng_channels", []))
    prior_index_to_channel = (
        dict(state.rng().index_to_channel)
        if hasattr(state.rng(), "index_to_channel")
        else {}
    )

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
