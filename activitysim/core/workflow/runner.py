import logging
import multiprocessing
import time
from typing import Callable, Iterable

from activitysim.core.exceptions import DuplicateWorkflowNameError
from activitysim.core.workflow.accessor import FromWhale, WhaleAccessor
from activitysim.core.workflow.checkpoint import (
    CHECKPOINT_NAME,
    FINAL_CHECKPOINT_NAME,
    LAST_CHECKPOINT,
)

logger = logging.getLogger(__name__)


class Runner(WhaleAccessor):
    """
    This accessor provides the tools to actually run ActivitySim workflow steps.
    """

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
            if models is self.obj._RUNNABLE_STEPS.get(models.__name__, None):
                self([models.__name__], resume_after=None, memory_sidecar_process=None)
            elif models is self.obj._LOADABLE_OBJECTS.get(models.__name__, None):
                self.obj.set(models.__name__, self.obj.get(models.__name__))
            elif models is self.obj._LOADABLE_TABLES.get(models.__name__, None):
                self.obj.set(models.__name__, self.obj.get(models.__name__))
            else:
                raise DuplicateWorkflowNameError(models.__name__)

        from activitysim.core.tracing import print_elapsed_time

        t0 = print_elapsed_time()

        if resume_after:
            self.obj.checkpoint.restore(resume_after)
        t0 = print_elapsed_time("open_pipeline", t0)

        if resume_after == LAST_CHECKPOINT:
            resume_after = self.obj.checkpoint.last_checkpoint[CHECKPOINT_NAME]

        if resume_after:
            logger.info("resume_after %s" % resume_after)
            if resume_after in models:
                models = models[models.index(resume_after) + 1 :]

        self.obj.trace_memory_info("pipeline.run before preload_injectables")

        # preload any bulky injectables (e.g. skims) not in pipeline
        # if inject.get_injectable("preload_injectables", None):
        #     if memory_sidecar_process:
        #         memory_sidecar_process.set_event("preload_injectables")
        #     t0 = print_elapsed_time("preload_injectables", t0)

        self.obj.trace_memory_info("pipeline.run after preload_injectables")

        t0 = print_elapsed_time()
        for model in models:
            if memory_sidecar_process:
                memory_sidecar_process.set_event(model)
            t1 = print_elapsed_time()
            self.obj.run_model(model)
            self.obj.trace_memory_info(f"pipeline.run after {model}")

            self.log_runtime(model_name=model, start_time=t1)

        if memory_sidecar_process:
            memory_sidecar_process.set_event("finalizing")

        # add checkpoint with final tables even if not intermediate checkpointing
        if not self.obj.should_save_checkpoint():
            self.obj.checkpoint.add(FINAL_CHECKPOINT_NAME)

        self.obj.trace_memory_info("pipeline.run after run_models")

        t0 = print_elapsed_time("run_model (%s models)" % len(models), t0)

        # don't close the pipeline, as the user may want to read intermediate results from the store

    def __dir__(self) -> Iterable[str]:
        return self.obj._RUNNABLE_STEPS.keys()

    def __getattr__(self, item):
        if item in self.obj._RUNNABLE_STEPS:
            f = lambda **kwargs: self.obj._RUNNABLE_STEPS[item](
                self.obj.context, **kwargs
            )
            f.__doc__ = self.obj._RUNNABLE_STEPS[item].__doc__
            return f
        raise AttributeError(item)

    timing_notes: set[str] = FromWhale(default_init=True)

    def log_runtime(self, model_name, start_time=None, timing=None, force=False):

        assert (start_time or timing) and not (start_time and timing)

        timing = timing if timing else time.time() - start_time
        seconds = round(timing, 1)
        minutes = round(timing / 60, 1)

        process_name = multiprocessing.current_process().name

        if self.obj.settings.multiprocess and not force:
            # when benchmarking, log timing for each processes in its own log
            if self.obj.settings.benchmarking:
                header = "component_name,duration"
                with self.obj.filesystem.open_log_file(
                    f"timing_log.{process_name}.csv", "a", header
                ) as log_file:
                    print(f"{model_name},{timing}", file=log_file)
            # only continue to log runtime in global timing log for locutor
            if not self.obj.get_injectable("locutor", False):
                return

        header = "process_name,model_name,seconds,minutes,notes"
        note = " ".join(self.timing_notes)
        with self.obj.filesystem.open_log_file(
            "timing_log.csv", "a", header
        ) as log_file:
            print(
                f"{process_name},{model_name},{seconds},{minutes},{note}", file=log_file
            )

        self.timing_notes.clear()
