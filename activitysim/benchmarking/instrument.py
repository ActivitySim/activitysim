import importlib
import os
import webbrowser

from pyinstrument import Profiler


def run_instrument(bench_name, component_name, out_file=None):
    bench_module = importlib.import_module(
        f"activitysim.benchmarking.benchmarks.{bench_name}"
    )

    component = getattr(bench_module, component_name)()
    component.setup()
    with Profiler() as profiler:
        component.time_component()
    component.teardown()

    if out_file is None:
        out_file = f"instrument/{bench_name}/{component_name}.html"
    dirname = os.path.dirname(out_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    if out_file:
        with open(out_file, "wt") as f:
            f.write(profiler.output_html())
            webbrowser.open(f"file://{os.path.realpath(out_file)}")
