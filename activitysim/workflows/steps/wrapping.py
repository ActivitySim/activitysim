import importlib
from inspect import getfullargspec
from pypyr.context import Context
from . import get_formatted_or_default
from .progression import reset_progress_step


def report_step(func):

    _args, _varargs, _varkw, _defaults, _kwonlyargs, _kwonlydefaults, _annotations = getfullargspec(func)

    def run_step(context:Context=None) -> None:

        progress_tag = get_formatted_or_default(context, 'progress_tag', None)
        if progress_tag is not None:
            reset_progress_step(description=progress_tag)

        context.assert_key_has_value(key='report', caller=func.__module__)
        report = context.get('report')
        caption_type = get_formatted_or_default(context, 'caption_type', 'fig')
        caption_maker = get_formatted_or_default(context, caption_type, None)

        # parse and run function itself
        ndefault = len(_defaults)
        args = []
        for arg in _args[:-ndefault]:
            context.assert_key_has_value(key=arg, caller=func.__module__)
            args.append(context.get_formatted(arg))
        for arg, default in zip(_args[-ndefault:], _defaults):
            args.append(get_formatted_or_default(context, arg, default))
        kwargs = {}
        for karg in _kwonlyargs:
            if karg in _kwonlydefaults:
                kwargs[karg] = get_formatted_or_default(context, karg, _kwonlydefaults[karg])
            else:
                context.assert_key_has_value(key=karg, caller=func.__module__)
                kwargs[karg] = context.get_formatted(karg)
        outcome = func(*args, **kwargs)
        if isinstance(outcome, dict):
            for k, v in outcome.items():
                context[k] = v
        elif outcome is not None:
            caption = get_formatted_or_default(context, 'caption', None)
            caption_level = get_formatted_or_default(context, 'caption_level', None)
            with report:
                if caption is not None:
                    report << caption_maker(caption, level=caption_level)
                report << outcome

    module = importlib.import_module(func.__module__)
    setattr(module, 'run_step', run_step)
