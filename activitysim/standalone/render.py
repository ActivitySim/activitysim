import logging
import os
import nbformat as nbf
from nbconvert import HTMLExporter
import nbclient
import textwrap
from contextlib import contextmanager
from pathlib import Path
from xmle import Reporter

# from jupyter_contrib_nbextensions.nbconvert_support import TocExporter # problematic


@contextmanager
def chdir(path: Path):
    """
    Sets the cwd within the context

    Args:
        path (Path): The path to the cwd

    Yields:
        None
    """

    cwd = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)



def render_notebook(nb_filename, cellcontent):
    nb_filename = os.path.splitext(nb_filename)[0]
    nb = nbf.v4.new_notebook()

    cells = []
    for c in cellcontent:
        c = textwrap.dedent(c).strip()
        if c[:4] == "[md]":
            cells.append(nbf.v4.new_markdown_cell(c[4:]))
        else:
            cells.append(nbf.v4.new_code_cell(c))
    nb['cells'] = cells
    nbf.write(nb, nb_filename+".ipynb")

    nb = nbclient.execute(nb, cwd=os.path.dirname(nb_filename))
    nbf.write(nb, nb_filename+"-e.ipynb")

    html_exporter = HTMLExporter(
        embed_images=True,
        exclude_input_prompt=True,
        exclude_output_prompt=True,
        exclude_input=True,
        # template_name = 'classic'
    )
    (body, resources) = html_exporter.from_notebook_node(nb)

    with open(nb_filename+".html", 'w') as f:
        f.write(body)


def render_comparison(html_filename, title, dist_skim="DIST", county_id="county_id", timing_log=None):

    from activitysim.standalone.compare import load_final_tables, compare_trip_mode_choice, compare_trip_distance, compare_work_district, compare_runtime
    from activitysim.standalone.skims import load_skims

    with chdir(os.path.dirname(html_filename)):

        data = load_final_tables(
            {
                "sharrow": "output-sharrow",
                "legacy": "output-legacy",
            },
            {"trips": "final_trips.csv", "persons": "final_persons.csv", "land_use": "final_land_use.csv"},
            {"trips": "trip_id", "persons": "person_id", "land_use": "zone_id"},
        )

        report = Reporter(title=title)

        if timing_log:
            report << "## Model Runtime"
            report << compare_runtime(timing_log)

        report << "## Trip Mode Choice"
        report << compare_trip_mode_choice(
            data,
            title=None,
        )

        if dist_skim:
            skims = load_skims("../configs/network_los.yaml", "../data")
            report << "## Trip Distance"

            report << "### Trip Length Distribution <10 miles"
            report << compare_trip_distance(
                data,
                skims,
                dist_skim,
                max_dist=10,
                title=None,
            )

            report << "### Trip Length Distribution Overall"
            report << compare_trip_distance(
                data,
                skims,
                dist_skim,
                title=None,
            )

        work_district = compare_work_district(
            data,
            district_id=county_id,
            label='county',
            hometaz_col='home_zone_id',
            worktaz_col='workplace_zone_id',
            data_dictionary="../configs/data_dictionary.yaml",
        )
        if work_district is not None:
            report << "## Workers by Home and Work County"
            report << work_district


        # save final report
        report.save(os.path.basename(html_filename), overwrite=True)