from pypyr.errors import KeyNotInContextError
from .cmd import run_step as _run_cmd


def _get_formatted(context, key, default):
    try:
        out = context.get_formatted(key)
    except KeyNotInContextError:
        out = None
    if out is None:
        out = default
    return out


def run_step(context):
    tag = context.get_formatted('tag')
    pre_config_dirs = _get_formatted(context, 'pre_config_dirs', [])
    if isinstance(pre_config_dirs, str):
        pre_config_dirs = [pre_config_dirs]
    config_dirs = _get_formatted(context, 'config_dirs', ['configs'])
    if isinstance(config_dirs, str):
        config_dirs = [config_dirs]
    data_dir = _get_formatted(context, 'data_dir', 'data')
    output_dir = _get_formatted(context, 'output_dir', f'output-{tag}')
    resume_after = context.get('resume_after', None)
    fast = context.get('fast', True)
    flags = []
    if resume_after:
        flags.append(f" -r {resume_after}")
    if fast:
        flags.append("--fast")
    flags = " ".join(flags)
    cmd = {}
    cmd['cwd'] = context.get_formatted('cwd')
    cmd['label'] = context.get_formatted('label')
    cfgs = " ".join(f"-c {c}" for c in pre_config_dirs+config_dirs)
    cmd['run'] = f"python -m activitysim run {cfgs} -d {data_dir} -o {output_dir} {flags}"
    context['cmd'] = cmd
    _run_cmd(context)
