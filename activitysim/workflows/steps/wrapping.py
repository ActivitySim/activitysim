import importlib
from inspect import getfullargspec
from pypyr.context import Context
from . import get_formatted_or_default
from .progression import reset_progress_step
from .error_handler import error_logging
import logging

logger = logging.getLogger(__name__)


class MockReport:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type or exc_val or exc_tb:
            logger.exception(repr(exc_val))
    def __lshift__(self, other):
        logger.info(str(other))


def report_step(func):

    _args, _varargs, _varkw, _defaults, _kwonlyargs, _kwonlydefaults, _annotations = getfullargspec(func)

    def run_step(context:Context=None) -> None:

        caption = get_formatted_or_default(context, 'caption', None)
        progress_tag = get_formatted_or_default(context, 'progress_tag', caption)
        if progress_tag is not None:
            reset_progress_step(description=progress_tag)

        if _annotations.get('return', '<missing>') not in {None, dict, Context}:
            context.assert_key_has_value(key='report', caller=func.__module__)
        report = context.get('report', MockReport())
        with report:
            caption_type = get_formatted_or_default(context, 'caption_type', 'fig')
            caption_maker = get_formatted_or_default(context, caption_type, None)
            # parse and run function itself
            if _defaults is None:
                ndefault = 0
                _required_args = _args
            else:
                ndefault = len(_defaults)
                _required_args = _args[:-ndefault]
            args = []
            for arg in _required_args:
                context.assert_key_has_value(key=arg, caller=func.__module__)
                try:
                    args.append(context.get_formatted_or_raw(arg))
                except Exception as err:
                    raise ValueError(f"extracting {arg} from context") from err
            if ndefault:
                for arg, default in zip(_args[-ndefault:], _defaults):
                    args.append(get_formatted_or_default(context, arg, default))
            kwargs = {}
            for karg in _kwonlyargs:
                if karg in _kwonlydefaults:
                    kwargs[karg] = get_formatted_or_default(context, karg, _kwonlydefaults[karg])
                else:
                    context.assert_key_has_value(key=karg, caller=func.__module__)
                    try:
                        kwargs[karg] = context.get_formatted_or_raw(karg)
                    except Exception as err:
                        raise ValueError(f"extracting {karg} from context") from err
            outcome = error_logging(func)(*args, **kwargs)
            logger.critical(f"{type(outcome)=}")
            if isinstance(outcome, (dict, Context)):
                for k, v in outcome.items():
                    context[k] = v
            elif outcome is not None:
                caption_level = get_formatted_or_default(context, 'caption_level', None)
                if caption is not None:
                    report << caption_maker(caption, level=caption_level)
                report << outcome

    module = importlib.import_module(func.__module__)
    if hasattr(module, 'run_step'):
        raise ValueError(f"{func.__module__}.run_step exists, there can be only one per module")
    setattr(module, 'run_step', run_step)
