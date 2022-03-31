import os
import time
from datetime import timedelta

class DummyProgress:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def add_task(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

dummy_progress = DummyProgress()

try:

    from rich.progress import Progress, ProgressColumn, Text
    from rich.panel import Panel

except ImportError:

    # allow printing in color on windows terminal
    if os.name == 'nt':
        import ctypes
        hOut = ctypes.windll.kernel32.GetStdHandle(-11)
        out_modes = ctypes.c_uint32()
        ENABLE_VT_PROCESSING = ctypes.c_uint32(0x0004)
        ctypes.windll.kernel32.GetConsoleMode(hOut, ctypes.byref(out_modes))
        out_modes = ctypes.c_uint32(out_modes.value | 0x0004)
        ctypes.windll.kernel32.SetConsoleMode(hOut, out_modes)

    Progress = DummyProgress

else:

    class MyProgress(Progress):
        def get_renderables(self):
            yield Panel(self.make_tasks_table(self.tasks))


    class TimeElapsedColumn(ProgressColumn):
        """Renders time elapsed."""

        def render(self, task: "Task") -> Text:
            """Show time remaining."""
            elapsed = task.finished_time if task.finished else task.elapsed
            if elapsed is None:
                return Text("-:--:--", style="progress.elapsed")
            delta = timedelta(seconds=(elapsed))
            return Text(str(delta)[:-5], style="progress.elapsed")


    progress = MyProgress(
        TimeElapsedColumn(),
        "[progress.description]{task.description}",
    )

    progress_step = progress.add_task("step")
    progress_overall = progress.add_task("overall")


def get_progress():
    if os.environ.get('NO_RICH', False):
        return dummy_progress
    else:
        return progress


def update_progress_overall(*args, **kwargs):
    progress.update(progress_overall, *args, **kwargs)


def reset_progress_step(*args, description='', prefix='', **kwargs):
    print("╭" + "─" * (len(description)+2) + "╮")
    print(f"│ {description} │ {time.strftime('%I:%M:%S %p')}")
    print("╰" + "─" * (len(description)+2) + "╯")
    if not os.environ.get('NO_RICH', False):
        progress.reset(progress_step, *args, description=prefix+description, **kwargs)
