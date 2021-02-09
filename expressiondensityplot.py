#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from granatum_sdk import Granatum
from palettable.cmocean.sequential import Amp_3
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
from itertools import cycle
import seaborn as sns
from sklearn.linear_model import PoissonRegressor

import os
import traceback
import sys

from gbox_py_helpers import bug_report

def parse(st):
    return list(map(lambda s: s.strip(), list(filter(lambda s: s != "", st.split(',')))))

def main():
    gn = Granatum()

    sample_coords = gn.get_import("viz_data")
    df = gn.pandas_from_assay(gn.get_import("assay"))
    gene_ids = parse(gn.get_arg("gene_ids"))

    coords = sample_coords.get("coords")
    dim_names = sample_coords.get("dimNames")

    for gene in gene_ids:
        plt.figure()

        glm = PoissonRegressor()
        glm.fit(range(len(df.columns)), df.loc[gene, :].to_list())

        gn.add_result("Statistic {}".format(glm.get_params()), "markdown")


        plt.subplot(1, 1, 1)
        plt.title('Gene expression level distribution')
        sns.distplot(df.loc[gene,:].to_list(), bins=int(100), color = 'darkblue', kde_kws={'linewidth': 2})
        plt.ylabel('Frequency')
        plt.xlabel('Gene expression')

        plt.tight_layout()

        caption = ( "The distribution of expression levels for gene {}.".format(gene))
        gn.add_current_figure_to_results(caption, zoom=1, dpi=75)

    gn.commit()

if __name__ == "__main__":
    # Try except block to send an email about error #
    try:
        main()
    except:
        error_message = traceback.format_exc()
        sys.stderr.write(error_message) # Write the error to stderr anyway so the user can see what went wrong
        bug_report("Color Scatter-Plot", "amantrav@umich.edu", error_message)
