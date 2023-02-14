import logging
from typing import Iterable

from activitysim.core.workflow.checkpoint import (
    CHECKPOINT_NAME,
    FINAL_CHECKPOINT_NAME,
    LAST_CHECKPOINT,
)

logger = logging.getLogger(__name__)


class Runner:
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, objtype=None):
        from .state import Whale

        assert isinstance(instance, Whale)
        self._obj = instance
        return self

    def __set__(self, instance, value):
        raise ValueError(f"cannot set {self._name}")

    def __delete__(self, instance):
        raise ValueError(f"cannot delete {self._name}")

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
        models : [str]
            list of model_names
        resume_after : str or None
            model_name of checkpoint to load checkpoint and AFTER WHICH to resume model run
        memory_sidecar_process : MemorySidecar, optional
            Subprocess that monitors memory usage

        returns:
            nothing, but with pipeline open
        """
        from activitysim.core.tracing import print_elapsed_time

        t0 = print_elapsed_time()

        if resume_after:
            self._obj.open_pipeline(resume_after)
        t0 = print_elapsed_time("open_pipeline", t0)

        if resume_after == LAST_CHECKPOINT:
            resume_after = self._obj.checkpoint.last_checkpoint[CHECKPOINT_NAME]

        if resume_after:
            logger.info("resume_after %s" % resume_after)
            if resume_after in models:
                models = models[models.index(resume_after) + 1 :]

        self._obj.trace_memory_info("pipeline.run before preload_injectables")

        # preload any bulky injectables (e.g. skims) not in pipeline
        # if inject.get_injectable("preload_injectables", None):
        #     if memory_sidecar_process:
        #         memory_sidecar_process.set_event("preload_injectables")
        #     t0 = print_elapsed_time("preload_injectables", t0)

        self._obj.trace_memory_info("pipeline.run after preload_injectables")

        t0 = print_elapsed_time()
        for model in models:
            if memory_sidecar_process:
                memory_sidecar_process.set_event(model)
            t1 = print_elapsed_time()
            self._obj.run_model(model)
            self._obj.trace_memory_info(f"pipeline.run after {model}")

            from activitysim.core.tracing import log_runtime

            log_runtime(self._obj, model_name=model, start_time=t1)

        if memory_sidecar_process:
            memory_sidecar_process.set_event("finalizing")

        # add checkpoint with final tables even if not intermediate checkpointing
        if not self._obj.should_save_checkpoint():
            self._obj.checkpoint.add(FINAL_CHECKPOINT_NAME)

        self._obj.trace_memory_info("pipeline.run after run_models")

        t0 = print_elapsed_time("run_model (%s models)" % len(models), t0)

        # don't close the pipeline, as the user may want to read intermediate results from the store

    def __dir__(self) -> Iterable[str]:
        return self._obj._RUNNABLE_STEPS.keys()

    def __getattr__(self, item):
        if item in self._obj._RUNNABLE_STEPS:
            f = lambda **kwargs: self._obj._RUNNABLE_STEPS[item](
                self._obj.context, **kwargs
            )
            f.__doc__ = self._obj._RUNNABLE_STEPS[item].__doc__
            return f
