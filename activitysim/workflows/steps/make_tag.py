from .progression import reset_progress_step
import pprint
import time


def run_step(context):
    """Execute dynamic python code.

    Takes two forms of input:
        py: exec contents as dynamically interpreted python statements, with
            contents of context available as vars.
        pycode: exec contents as dynamically interpreted python statements,
            with the context object itself available as a var.

    Args:
        context (pypyr.context.Context): Mandatory.
            Context is a dictionary or dictionary-like.
            Context must contain key 'py' or 'pycode'
    """
    reset_progress_step(description="Initialize Tag")

    tag = context.get('tag', None)
    compile = context.get('compile', True)
    sharrow = context.get('sharrow', True)
    legacy = context.get('legacy', True)
    resume_after = context.get('resume_after', True)
    fast = context.get('fast', True)

    if tag is None:
        tag = time.strftime("%Y-%m-%d-%H%M%S")
        context['tag'] = tag
    context['compile'] = (compile and sharrow)
    context['contrast'] = (sharrow and legacy)

    flags = []
    if resume_after:
        flags.append(f" -r {resume_after}")
    if fast:
        flags.append("--fast")
    context['flags'] = " ".join(flags)

    pprint.pprint(context)
