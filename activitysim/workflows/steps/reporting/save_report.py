from __future__ import annotations

import logging
import os
import shutil

from .... import __version__
from ..wrapping import workstep

# Get decorated version when in development
try:
    top_package = __import__(__name__.split(".")[0])
    import setuptools_scm

    version = setuptools_scm.get_version(os.path.dirname(top_package.__path__[0]))
except:  # noqa: E722
    version = __version__


@workstep
def save_report(
    report,
    html_filename,
    toc_color="forest",
    copy_filename=None,
):
    bootstrap_font_family = (
        'font-family: -apple-system, "system-ui", "Segoe UI",'
        ' "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji",'
        ' "Segoe UI Emoji", "Segoe UI Symbol";'
    )
    toc_width = 200
    signature_font = "font-size:70%; font-weight:200;"
    signature_name_font = "font-weight:400; font-style:normal;"

    css = (
        """
    html { """
        + bootstrap_font_family
        + """ }
    div.xmle_title { font-weight:700; font-size: 2em; color: rgb(35, 144, 98);}
    body { margin-left: """
        + str(toc_width)
        + """px; }
    div.xmle_html_report { padding-left: 5px; }
    .table_of_contents_frame { width: """
        + str(toc_width - 13)
        + """px; position: fixed; margin-left: -"""
        + str(toc_width)
        + """px; top:0; padding-top:10px; z-index:2000;}
    .table_of_contents { width: """
        + str(toc_width - 13)
        + """px; position: fixed; margin-left: -"""
        + str(toc_width)
        + """px; font-size:85%;}
    .table_of_contents_head { font-weight:700; padding-left:25px;  }
    .table_of_contents ul { padding-left:25px;  }
    .table_of_contents ul ul { font-size:75%; padding-left:15px; }
    .xmle_signature {"""
        + signature_font
        + """ width: """
        + str(toc_width - 30)
        + """px; position: fixed; left: 0px; bottom: 0px; padding-left:20px;"""
        + """padding-bottom:2px; background-color:rgba(255,255,255,0.9);}
    .xmle_name_signature {"""
        + signature_name_font
        + """}
    a.parameter_reference {font-style: italic; text-decoration: none}
    .strut2 {min-width:2in}
    .histogram_cell { padding-top:1; padding-bottom:1; vertical-align:center; }
    table { border-spacing: 0; border: none; }
    .dataframe tbody tr:nth-child(odd) {
        background: #f5f5f5;
    }
    .dataframe tr, .dataframe th, .dataframe td {
        text-align: right;
        vertical-align: middle;
        padding: 0.2em 0.2em;
        line-height: normal;
        white-space: normal;
        max-width: none;
        border: none;
    }
    .dataframe table {
        margin-left: auto;
        margin-right: auto;
        border: none;
        border-collapse: collapse;
        border-spacing: 0;
        color: black;
        font-size: 12px;
        table-layout: fixed;
    }
    table.floatinghead thead {background-color:#FFF;}
    table.dataframe thead {background-color:#FFF;}
    @media print {
       body { color: #000; background: #fff; width: 100%; margin: 0; padding: 0;}
       /*.table_of_contents { display: none; }*/
       @page {
          margin: 1in;
       }
       h1, h2, h3 { page-break-after: avoid; }
       img { max-width: 100% !important; }
       ul, img, table { page-break-inside: avoid; }
       .xmle_signature {"""
        + signature_font
        + """ padding:0; background-color:#fff; position: fixed; bottom: 0;}
       .xmle_name_signature {"""
        + signature_name_font
        + """}
       .xmle_signature img {display:none;}
       .xmle_signature .noprint {display:none;}
    }
    """
    )

    logging.critical(f"SAVING REPORT TO {(html_filename)}")
    report.save(
        html_filename,
        overwrite=True,
        css=css,
        toc_font=bootstrap_font_family,
        toc_color=toc_color,
        branding=f"ActivitySim {version.replace('+', ' +')}",
    )
    if copy_filename is not None:
        os.makedirs(os.path.dirname(copy_filename), exist_ok=True)
        shutil.copy2(html_filename, copy_filename)
