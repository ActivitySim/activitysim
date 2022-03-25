import nbformat as nbf
from nbconvert import HTMLExporter
import nbclient

# from jupyter_contrib_nbextensions.nbconvert_support import TocExporter # problematic

def render_notebook(nb_filename, cellcontent):

    nb = nbf.v4.new_notebook()

    cells = []
    for c in cellcontent:
        if c[:4] == "[md]":
            cells.append(nbf.v4.new_markdown_cell(c[4:]))
        else:
            cells.append(nbf.v4.new_code_cell(c))
    nb['cells'] = cells

    nb = nbclient.execute(nb)

    html_exporter = HTMLExporter(
        embed_images=True,
        exclude_input_prompt=True,
        exclude_output_prompt=True,
        exclude_input=True,
        # template_name = 'classic'
    )
    (body, resources) = html_exporter.from_notebook_node(nb)

    with open(nb_filename, 'w') as f:
        f.write(body)


def render_comparison(html_filename, title, dist_skim="DIST"):

    cells = [
        f"[md]# {title}",

        """\
        from activitysim.standalone.compare import load_final_tables, compare_trip_mode_choice, compare_trip_distance
        from activitysim.standalone.skims import load_skims
        """,

        """\
        data = load_final_tables(
            {"sharrow": "output-sharrow", "legacy": "output-legacy"},
            {"trips": "final_trips.csv"},
            {"trips": "trip_id"},
        )
        """,

        f"[md]## Trip Mode Choice",

        """\
        compare_trip_mode_choice(data)
        """,
    ]

    if dist_skim:
        dist_cells = [
            f"[md]## Trip Distance",

            f"""\
            skims = load_skims("../configs/network_los.yaml", "../data")
            compare_trip_distance(
                data,
                skims,
                "{dist_skim}",
                max_dist=10,
                title="Trip Length Distribution <10 miles",
                )
            """,

            f"""\
            compare_trip_distance(
                data,
                skims,
                "{dist_skim}",
                title="Trip Length Distribution Overall",
                )
            """,
        ]
    else:
        dist_cells = []

    render_notebook(
        html_filename,
        cells + dist_cells
    )