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
from scipy.stats import poisson
from scipy.optimize import least_squares
import seaborn as sns
import statistics as s
import statsmodels.stats.api as sms


from sklearn.mixture import GaussianMixture as GM

import os
import traceback
import sys

from gbox_py_helpers import bug_report

def confint(X, alpha=0.05):
    resultdict = {}
    meanbounds = sms.DescrStatsW(X).tconfint_mean(alpha=alpha)
    resultdict["low"] = meanbounds[0]
    resultdict["high"] = meanbounds[1]
    resultdict["n"] = len(X)
    return resultdict

def dist(int1, int2):
    if int1["low"] >= int2["high"]:
        return int1["low"] - int2["high"]
    if int2["low"] >= int1["high"]:
        return int2["low"] - int1["high"]
    return 0.0

# return hash of labels associated to its data
def trygmonvector(gm, X):
    vectors = gm.predict(np.array(X).reshape(-1, 1))
    inv_map = {}
    for i, v in enumerate(vectors):
        inv_map[v] = inv_map.get(v, []) + [X[i]]
    return inv_map


def model1(t, coeffs):
    lamb1 = coeffs[0]
    return poisson.pmf(np.round(t), lamb1)

def residuals1(coeffs, y, t):
    return y - model1(t, coeffs)

def model2(t, coeffs):
    a = coeffs[0]
    lamb1 = coeffs[1]
    lamb2 = coeffs[2]
    return (a*poisson.pmf(np.round(t), lamb1)+poisson.pmf(np.round(t), lamb2))/(1.0+a)

def residuals2(coeffs, y, t):
    return y - model2(t, coeffs)


def fit_data_two_poissons(X, initial):
    entries, bins = np.histogram(X, bins="auto", density=True)
    bins = (bins[1:] + bins[:-1]) / 2
    parameters = least_squares(residuals2, initial, args=(entries, bins))
    return parameters


def fit_data_one_poisson(X, initial):
    entries, bins = np.histogram(X, bins="auto", density=True)
    bins = (bins[1:] + bins[:-1]) / 2
    parameters = least_squares(residuals1, initial, args=(entries, bins))
    return parameters


def one_or_two_mixtures(X, alpha=0.05, min_dist=0.2, min_zscore=2):
    column = np.array(X).reshape(-1, 1)
    gm = GM(n_components=2).fit(column)
    inv_map = trygmonvector(gm, X)
    mean = s.mean(X)
    std = s.stdev(X)

    if len(inv_map) <= 1 or len(inv_map[0]) < 3 or len(inv_map[1]) < 3:
        gm = GM(n_components=1).fit(column)
        mi = confint(X)
        return {"data": X, "mean": mean, "std": std, "gm": gm, "low_means": [mi["low"]], "high_means": [mi["high"]], "n": [len(X)]}

    mi1 = confint(inv_map[0], alpha=alpha)
    mi2 = confint(inv_map[1], alpha=alpha)
    # zscore1 = abs(s.mean(inv_map[0])-s.mean(inv_map[1]))/(s.stdev(inv_map[1])+1e-16)
    # zscore2 = abs(s.mean(inv_map[1])-s.mean(inv_map[0]))/(s.stdev(inv_map[0])+1e-16)
    if dist(mi1, mi2) <= min_dist or abs(gm.means_[1][0]-gm.means_[0][0])/(max(gm.covariances_)[0][0]) < min_zscore:
        gm = GM(n_components=1).fit(column)
        mi = confint(X)
        result = {"data": X, "mean": mean, "std": std, "gm": gm, "low_means": [mi["low"]], "high_means": [mi["high"]], "n": [len(X)]}
    elif mi1["low"] < mi2["low"]:
        result = {"data": X, "mean": mean, "std": std, "gm": gm, "label_order": [0, 1], "low_means": [mi1["low"], mi2["low"]], "high_means": [mi1["high"], mi2["high"]], "n": [mi1["n"], mi2["n"]]}
    else:
        result = {"data": X, "mean": mean, "std": std, "gm": gm, "label_order": [1, 0], "low_means": [mi2["low"], mi1["low"]], "high_means": [mi2["high"], mi1["high"]], "n": [mi2["n"], mi1["n"]]}
    return result

# First transform X into log(X)+c such that it does not go below 0
# X is a list
def fit_poissons(X, alpha=0.05, min_dist=0.2, min_zscore=2):
    shift = np.min(X) - 1                  # Needed later to shift back
    Xarr = np.log(X - shift)
    res = one_or_two_mixtures(Xarr.tolist(), alpha=0.05, min_dist=min_dist, min_zscore=min_zscore)
    numcomponents = len(res["low_means"])
    if numcomponents == 2:
        mean1 = 0.5*(res["low_means"][0] + res["high_means"][0])
        mean2 = 0.5*(res["low_means"][1] + res["high_means"][1])
        mean1 = np.exp(mean1) + shift
        mean2 = np.exp(mean2) + shift
        sz1 = res["n"][0]
        sz2 = res["n"][1]
        alpha = sz1/(sz1+sz2)
        # Now optimize with estimates
        coeffs = fit_data_two_poissons(X, [alpha, mean1, mean2])
        if coeffs.x[0] > 0.0 and coeffs.x[0] < 1.0:
            return {"n":2, "coeffs":coeffs}
    mean1 = np.exp(0.5*(res["low_means"][0] + res["high_means"][0])) + shift
    coeffs = fit_data_one_poisson(X, [mean1])
    return {"n":1, "coeffs":coeffs}

def plot_fits(row, alpha=0.05, min_dist=0.2, min_zscore = 2):
    params = fit_poissons(row, alpha=alpha, min_dist=min_dist, min_zscore=min_zscore)
    X, bins, blah = np.histogram(row, bins=int(np.round(np.max(row)-np.min(row))), density=True)
    bins = (bins[1:] + bins[:-1]) / 2.0
    sns.kdeplot(row, shade=True)
    if params["n"] == 1:
        plt.plot(bins, model1(bins, params["coeffs"].x), "rx")
    else:
        plt.plot(bins, model2(bins, params["coeffs"].x), "gx")
        a = params["coeffs"].x[0]
        sc = (a+1.0)/a
        mn1 = params["coeffs"].x[1]
        plt.plot(bins, model1(bins, [mn1])/sc, "bx")
        print("Standard deviation = {}".format(np.sqrt(mn1)))


def parse(st):
    return list(map(lambda s: s.strip(), list(filter(lambda s: s != "", st.split(',')))))


def main():
    gn = Granatum()

    sample_coords = gn.get_import("viz_data")
    df = gn.pandas_from_assay(gn.get_import("assay"))
    gene_ids = parse(gn.get_arg("gene_ids"))
    groups = gn.get_import("groups")

    coords = sample_coords.get("coords")
    dim_names = sample_coords.get("dimNames")

    inv_map = {}
    for k, v in groups.items():
        inv_map[v] = inv_map.get(v, []) + [k]

    for gene in gene_ids:
        for k, v in inv_map.items():
            plt.figure()

            plt.subplot(1, 1, 1)
            plt.title('Gene expression level distribution')
            plot_fits(df.loc[gene,v].to_list(), alpha=0.05, min_dist=0.1, min_zscore=2)
            # sns.distplot(df.loc[gene,:].to_list(), bins=int(100), color = 'darkblue', kde_kws={'linewidth': 2})
            plt.ylabel('Frequency')
            plt.xlabel('Gene expression')

            plt.tight_layout()

            caption = ( "The distribution of expression levels for gene {} and cluster {}.".format(gene, k))
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
