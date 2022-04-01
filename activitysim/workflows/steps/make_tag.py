from .progression import reset_progress_step
import pprint
import time


def run_step(context):
    reset_progress_step(description="Initialize Tag")

    context.assert_key_has_value(key='example_name', caller=__name__)
    example_name = context.get_formatted('example_name')
    tag = context.get('tag', None)
    compile = context.get('compile', True)
    sharrow = context.get('sharrow', True)
    legacy = context.get('legacy', True)
    resume_after = context.get('resume_after', True)
    fast = context.get('fast', True)
    mp = context.get('mp', False)

    if tag is None:
        tag = time.strftime("%Y-%m-%d-%H%M%S")
        context['tag'] = tag
    # context['compile'] = (compile and sharrow)
    context['contrast'] = (sharrow and legacy)

    flags = []

    from activitysim.cli.create import EXAMPLES
    run_flags = EXAMPLES.get(example_name, {}).get('run_flags', {})
    if isinstance(run_flags, str):
        flags.append(run_flags)
    else:
        flags.append(run_flags.get('multi' if mp else 'single'))

    if resume_after:
        flags.append(f" -r {resume_after}")
    if fast:
        flags.append("--fast")
    context['flags'] = " ".join(flags)

    pprint.pprint(context)
