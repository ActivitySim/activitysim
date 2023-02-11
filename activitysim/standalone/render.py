import logging
import os
import textwrap
from contextlib import contextmanager
from pathlib import Path

import nbclient
import nbformat as nbf
from nbconvert import HTMLExporter
from xmle import NumberedCaption, Reporter

from .. import __version__

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
    nb["cells"] = cells
    nbf.write(nb, nb_filename + ".ipynb")

    nb = nbclient.execute(nb, cwd=os.path.dirname(nb_filename))
    nbf.write(nb, nb_filename + "-e.ipynb")

    html_exporter = HTMLExporter(
        embed_images=True,
        exclude_input_prompt=True,
        exclude_output_prompt=True,
        exclude_input=True,
        # template_name = 'classic'
    )
    (body, resources) = html_exporter.from_notebook_node(nb)

    with open(nb_filename + ".html", "w") as f:
        f.write(body)
