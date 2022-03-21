import os

try:

    from rich.progress import Progress, TimeElapsedColumn

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

    class Progress:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def add_task(self, *args, **kwargs):
            pass

    class TimeElapsedColumn:
        pass


from rich.panel import Panel
from rich.progress import Progress

class MyProgress(Progress):
    def get_renderables(self):
        yield Panel(self.make_tasks_table(self.tasks))

progress = MyProgress(
    TimeElapsedColumn(),
    "[progress.description]{task.description}",
)

progress_step = progress.add_task("step")
progress_overall = progress.add_task("overall")