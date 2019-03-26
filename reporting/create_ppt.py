"""
See http://pbpython.com/creating-powerpoint.html for details on this script
Requires https://python-pptx.readthedocs.org/en/latest/index.html
Example program showing how to read in Excel, process with pandas and
output to a PowerPoint file.
"""

from __future__ import print_function
from pptx import Presentation
from pptx.util import Inches
import argparse
import numpy as np
from datetime import date
import jsonschema, json
import matplotlib.pyplot as plt
from summarize_metrics import summarize_data, baa_thresholds


def parse_args():
    """ Setup the input and output arguments for the script
    Return the parsed input and output files
    """
    parser = argparse.ArgumentParser(description='Create ppt report')
    parser.add_argument('infile',
                        type=argparse.FileType('r'),
                        help='Powerpoint file used as the template')
    parser.add_argument('outfile',
                        type=argparse.FileType('w'),
                        help='Output powerpoint report file')
    parser.add_argument('data',
                        type=argparse.FileType('r'),
                        help='json_data')
    return parser.parse_args()


def create_ppt(input, output, json_data, metric_images=None):
    """ Take the input powerpoint file and use it as the template for the output
    file.
    """
    data = parse_metrics_code(json_data)
    prs = Presentation(input)
    # Use the output from analyze_ppt to understand which layouts and placeholders
    # to use
    # Create a title slide first
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "CORE3D Metrics Report"
    subtitle.text = "Generated on {:%m-%d-%Y}".format(date.today())
    prs.save(output)


def parse_metrics_code(json_file_path):
    baa_threshold = baa_thresholds()
    summarize_data(baa_threshold)
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    return data


if __name__ == "__main__":
    args = parse_args()
    create_ppt(args.infile.name, args.outfile.name, args.data.name)
