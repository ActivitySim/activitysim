import logging
import os
import nbformat as nbf
from nbconvert import HTMLExporter
import nbclient
import textwrap
from contextlib import contextmanager
from pathlib import Path
from xmle import Reporter, NumberedCaption
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


# def render_comparison(html_filename, title, dist_skim="DIST", county_id="county_id", timing_log=None):
#
#     from activitysim.standalone.compare import load_final_tables, compare_trip_mode_choice, compare_trip_distance, compare_work_district, compare_runtime
#     from activitysim.standalone.skims import load_skims
#
#     with chdir(os.path.dirname(html_filename)):
#
#         data = load_final_tables(
#             {
#                 "sharrow": "output-sharrow",
#                 "legacy": "output-legacy",
#             },
#             {"trips": "final_trips.csv", "persons": "final_persons.csv", "land_use": "final_land_use.csv"},
#             {"trips": "trip_id", "persons": "person_id", "land_use": "zone_id"},
#         )
#
#         bootstrap_font_family = (
#             'font-family: -apple-system, "system-ui", "Segoe UI",'
#             ' "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji",'
#             ' "Segoe UI Emoji", "Segoe UI Symbol";'
#         )
#         toc_width = 200
#         signature_font = 'font-size:70%; font-weight:200;'
#         signature_name_font = 'font-weight:400; font-style:normal;'
#
#         css = """
#         html { """ + bootstrap_font_family + """ }
#         div.xmle_title { font-weight:700; font-size: 2em; color: rgb(35, 144, 98);}
#         body { margin-left: """ + str(toc_width) + """px; }
#         div.xmle_html_report { padding-left: 5px; }
#         .table_of_contents_frame { width: """ + str(
#         toc_width - 13) + """px; position: fixed; margin-left: -""" + str(toc_width) + """px; top:0; padding-top:10px; z-index:2000;}
#         .table_of_contents { width: """ + str(toc_width - 13) + """px; position: fixed; margin-left: -""" + str(
#         toc_width) + """px; font-size:85%;}
#         .table_of_contents_head { font-weight:700; padding-left:25px;  }
#         .table_of_contents ul { padding-left:25px;  }
#         .table_of_contents ul ul { font-size:75%; padding-left:15px; }
#         .xmle_signature {""" + signature_font + """ width: """ + str(toc_width - 30) + """px; position: fixed; left: 0px; bottom: 0px; padding-left:20px; padding-bottom:2px; background-color:rgba(255,255,255,0.9);}
#         .xmle_name_signature {""" + signature_name_font + """}
#         a.parameter_reference {font-style: italic; text-decoration: none}
#         .strut2 {min-width:2in}
#         .histogram_cell { padding-top:1; padding-bottom:1; vertical-align:center; }
#         table.floatinghead thead {background-color:#FFF;}
#         table.dataframe thead {background-color:#FFF;}
#         @media print {
#            body { color: #000; background: #fff; width: 100%; margin: 0; padding: 0;}
#            /*.table_of_contents { display: none; }*/
#            @page {
#               margin: 1in;
#            }
#            h1, h2, h3 { page-break-after: avoid; }
#            img { max-width: 100% !important; }
#            ul, img, table { page-break-inside: avoid; }
#            .xmle_signature {""" + signature_font + """ padding:0; background-color:#fff; position: fixed; bottom: 0;}
#            .xmle_name_signature {""" + signature_name_font + """}
#            .xmle_signature img {display:none;}
#            .xmle_signature .noprint {display:none;}
#         }
#         """
#         report = Reporter(title=title)
#         Fig = NumberedCaption("Figure", level=2, anchor=True)
#
#         with report:
#             if timing_log:
#                 report << Fig("Model Runtime")
#                 report << compare_runtime(timing_log)
#
#         with report:
#             report << Fig("Trip Mode Choice")
#             report << compare_trip_mode_choice(
#                 data,
#                 title=None,
#             )
#
#         with report:
#             if dist_skim:
#                 skims = load_skims("../configs/network_los.yaml", "../data")
#                 report << Fig("Trip Length Distribution <10 miles")
#                 report << compare_trip_distance(
#                     data,
#                     skims,
#                     dist_skim,
#                     max_dist=10,
#                     title=None,
#                 )
#
#                 report << Fig("Trip Length Distribution Overall")
#                 report << compare_trip_distance(
#                     data,
#                     skims,
#                     dist_skim,
#                     title=None,
#                 )
#
#         with report:
#             work_district = compare_work_district(
#                 data,
#                 district_id=county_id,
#                 label='county',
#                 hometaz_col='home_zone_id',
#                 worktaz_col='workplace_zone_id',
#                 data_dictionary="../configs/data_dictionary.yaml",
#             )
#             if work_district is not None:
#                 report << Fig("Workers by Home and Work County")
#                 report << work_district
#
#         # save final report
#         report.save(
#             os.path.basename(html_filename),
#             overwrite=True,
#             css=css,
#             toc_font=bootstrap_font_family,
#             toc_color='forest',
#             branding=f"ActivitySim {__version__}",
#         )