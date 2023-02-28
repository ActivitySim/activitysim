from __future__ import annotations

import logging
import multiprocessing
import time
from datetime import timedelta
from typing import Callable, Iterable

from activitysim.core.exceptions import DuplicateWorkflowNameError
from activitysim.core.workflow.accessor import FromState, StateAccessor
from activitysim.core.workflow.checkpoint import (
    CHECKPOINT_NAME,
    CHECKPOINT_TABLE_NAME,
    FINAL_CHECKPOINT_NAME,
    LAST_CHECKPOINT,
)
from activitysim.core.workflow.steps import run_named_step

# single character prefix for run_list model name to indicate that no checkpoint should be saved
NO_CHECKPOINT_PREFIX = "_"


logger = logging.getLogger(__name__)


def _format_elapsed_time(t):
    t = round(t, 3)
    if t < 60:
        return f"{round(t, 3)} seconds"
    td = str(timedelta(seconds=t)).rstrip("0")
    if td.startswith("0:"):
        td = td[2:]
    return td


def split_arg(s, sep, default=""):
    """
    split str s in two at first sep, returning empty string as second result if no sep
    """
    r = s.split(sep, 2)
    r = list(map(str.strip, r))

    arg = r[0]

    if len(r) == 1:
        val = default
    else:
        val = r[1]
        val = {"true": True, "false": False}.get(val.lower(), val)

    return arg, val


class Runner(StateAccessor):
    """
    This accessor provides the tools to actually run ActivitySim workflow steps.
    """

    def __get__(self, instance, objtype=None) -> "Runner":
        # derived __get__ changes annotation, aids in type checking
        return super().__get__(instance, objtype)

    def __call__(self, models, resume_after=None, memory_sidecar_process=None):
        """
        run the specified list of models, optionally loading checkpoint and resuming after specified
        checkpoint.

        Since we use model_name as checkpoint name, the same model may not be run more than once.

        If resume_after checkpoint is specified and a model with that name appears in the models list,
        then we only run the models after that point in the list. This allows the user always to pass
        the same list of models, but specify a resume_after point if desired.

        Parameters
        ----------
        models : list[str] or Callable
            A list of the model names to run, which should all have been
            registered with the @workflow.step decorator.  Alternative, give
            a single function that is or could have been so-decorated.
        resume_after : str or None
            model_name of checkpoint to load checkpoint and AFTER WHICH to resume model run
        memory_sidecar_process : MemorySidecar, optional
            Subprocess that monitors memory usage

        returns:
            nothing, but with pipeline open
        """
        if isinstance(models, Callable) and models.__name__ is not None:
            if models is self._obj._RUNNABLE_STEPS.get(models.__name__, None):
                self([models.__name__], resume_after=None, memory_sidecar_process=None)
            elif models is self._obj._LOADABLE_OBJECTS.get(models.__name__, None):
                self._obj.set(models.__name__, self._obj.get(models.__name__))
            elif models is self._obj._LOADABLE_TABLES.get(models.__name__, None):
                self._obj.set(models.__name__, self._obj.get(models.__name__))
            else:
                raise DuplicateWorkflowNameError(models.__name__)
            return

        if isinstance(models, str):
            return self.by_name(models)

        from activitysim.core.tracing import print_elapsed_time

        t0 = print_elapsed_time()

        if resume_after == LAST_CHECKPOINT:
            _checkpoints = self._obj.checkpoint.store.list_checkpoint_names()
            if len(_checkpoints):
                _resume_after = _checkpoints[-1]
            else:
                # nothing available in the checkpoint.store, cannot resume_after
                resume_after = _resume_after = None
        else:
            _resume_after = resume_after

        if _resume_after:

            if (
                _resume_after != self._obj.checkpoint.last_checkpoint_name
                or self._obj.uncheckpointed_table_names()
            ):
                logger.debug(
                    f"last_checkpoint_name = {self._obj.checkpoint.last_checkpoint_name}"
                )
                logger.debug(
                    f"uncheckpointed_table_names = {self._obj.uncheckpointed_table_names()}"
                )
                logger.debug(f"restoring from store with resume_after = {resume_after}")
                self._obj.checkpoint.restore(resume_after)
                t0 = print_elapsed_time("checkpoint.restore", t0)
            else:
                logger.debug(f"good to go with resume_after = {resume_after}")

        if resume_after == LAST_CHECKPOINT:
            resume_after = self._obj.checkpoint.last_checkpoint[CHECKPOINT_NAME]

        if resume_after:
            logger.info("resume_after %s" % resume_after)
            if resume_after in models:
                models = models[models.index(resume_after) + 1 :]

        self._obj.trace_memory_info("pipeline.run before preload_injectables")

        # preload any bulky injectables (e.g. skims) not in pipeline
        if self._obj.get("preload_injectables", None):
            if memory_sidecar_process:
                memory_sidecar_process.set_event("preload_injectables")
            t0 = print_elapsed_time("preload_injectables", t0)

        self._obj.trace_memory_info("pipeline.run after preload_injectables")

        t0 = print_elapsed_time()
        for model in models:
            if memory_sidecar_process:
                memory_sidecar_process.set_event(model)
            t1 = print_elapsed_time()
            self.by_name(model)
            self._obj.trace_memory_info(f"pipeline.run after {model}")

            self.log_runtime(model_name=model, start_time=t1)

        if memory_sidecar_process:
            memory_sidecar_process.set_event("finalizing")

        # add checkpoint with final tables even if not intermediate checkpointing
        if not self._obj.should_save_checkpoint():
            self._obj.checkpoint.add(FINAL_CHECKPOINT_NAME)

        self._obj.trace_memory_info("pipeline.run after run_models")

        t0 = print_elapsed_time("run_model (%s models)" % len(models), t0)

        # don't close the pipeline, as the user may want to read intermediate results from the store

    def __dir__(self) -> Iterable[str]:
        return self._obj._RUNNABLE_STEPS.keys() | {"all"}

    def __getattr__(self, item):
        if item in self._obj._RUNNABLE_STEPS:
            f = lambda **kwargs: self.by_name(item, **kwargs)
            f.__doc__ = self._obj._RUNNABLE_STEPS[item].__doc__
            return f
        raise AttributeError(item)

    timing_notes: set[str] = FromState(default_init=True)

    def log_runtime(self, model_name, start_time=None, timing=None, force=False):
        assert (start_time or timing) and not (start_time and timing)

        timing = timing if timing else time.time() - start_time
        seconds = round(timing, 1)
        minutes = round(timing / 60, 1)

        process_name = multiprocessing.current_process().name

        if self._obj.settings.multiprocess and not force:
            # when benchmarking, log timing for each processes in its own log
            if self._obj.settings.benchmarking:
                header = "component_name,duration"
                with self._obj.filesystem.open_log_file(
                    f"timing_log.{process_name}.csv", "a", header
                ) as log_file:
                    print(f"{model_name},{timing}", file=log_file)
            # only continue to log runtime in global timing log for locutor
            if not self._obj.get_injectable("locutor", False):
                return

        header = "process_name,model_name,seconds,minutes,notes"
        note = " ".join(self.timing_notes)
        with self._obj.filesystem.open_log_file(
            "timing_log.csv", "a", header
        ) as log_file:
            print(
                f"{process_name},{model_name},{seconds},{minutes},{note}", file=log_file
            )

        self.timing_notes.clear()

    def _pre_run_step(self, model_name: str):
        if model_name in [
            checkpoint[CHECKPOINT_NAME]
            for checkpoint in self._obj.checkpoint.checkpoints
        ]:
            raise RuntimeError("Cannot run model '%s' more than once" % model_name)

        self._obj.rng().begin_step(model_name)

        # check for args
        if "." in model_name:
            step_name, arg_string = model_name.split(".", 1)
            args = dict(
                (k, v)
                for k, v in (
                    split_arg(item, "=", default=True) for item in arg_string.split(";")
                )
            )
        else:
            step_name = model_name
            args = {}

        # check for no_checkpoint prefix
        if step_name[0] == NO_CHECKPOINT_PREFIX:
            step_name = step_name[1:]
            checkpoint = False
        else:
            checkpoint = self._obj.should_save_checkpoint(model_name)

        self._obj.add_injectable("step_args", args)

        self._obj.trace_memory_info(f"pipeline.run_model {model_name} start")

        from activitysim.core.tracing import print_elapsed_time

        t0 = print_elapsed_time()
        logger.info(f"#run_model running step {step_name}")

        # these values are cached in the runner object itself, not in the context.
        self.step_name = step_name
        self.checkpoint = checkpoint
        self.t0 = t0

    def by_name(self, model_name, **kwargs):
        """
        Run the specified model and add checkpoint for model_name

        Since we use model_name as checkpoint name, the same model may not be run more than once.

        Parameters
        ----------
        model_name : str
            model_name is assumed to be the name of a registered orca step
        """
        self._pre_run_step(model_name)

        instrument = self._obj.settings.instrument
        if instrument is not None:
            try:
                from pyinstrument import Profiler
            except ImportError:
                instrument = False
        if isinstance(instrument, (list, set, tuple)):
            if self.step_name not in instrument:
                instrument = False
            else:
                instrument = True

        if instrument:
            from pyinstrument import Profiler

            with Profiler() as profiler:
                self._obj._context = run_named_step(
                    self.step_name, self._obj._context, **kwargs
                )
            out_file = self._obj.filesystem.get_profiling_file_path(
                f"{self.step_name}.html"
            )
            with open(out_file, "wt") as f:
                f.write(profiler.output_html())
        else:
            self._obj._context = run_named_step(
                self.step_name, self._obj._context, **kwargs
            )

        from activitysim.core.tracing import print_elapsed_time

        self.t0 = self.log_elapsed_time(f"run.{model_name}", self.t0)
        self._obj.trace_memory_info(f"pipeline.run_model {model_name} finished")

        self._obj.add_injectable("step_args", None)

        self._obj.rng().end_step(model_name)
        if self.checkpoint:
            self._obj.checkpoint.add(model_name)
        else:
            logger.info(
                "##### skipping %s checkpoint for %s" % (self.step_name, model_name)
            )

    def all(self, resume_after=LAST_CHECKPOINT, memory_sidecar_process=None):
        self(
            models=self._obj.settings.models,
            resume_after=resume_after,
            memory_sidecar_process=memory_sidecar_process,
        )

    def log_elapsed_time(self, msg, t0=None, level=25):
        t1 = time.time()
        assert t0 is not None
        t = t1 - (t0 or t1)
        msg = f" time to execute {msg} : {_format_elapsed_time(t)}"
        logger.log(level, msg)
        return t1
