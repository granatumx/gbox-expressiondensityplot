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
import statistics as s
import scipy.stats as st
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


def trygmonvector(gm, X):   # return hash of labels associated to its data
    vectors = gm.predict(np.array(X).reshape(-1, 1))
    inv_map = {}
    for i, v in enumerate(vectors):
        inv_map[v] = inv_map.get(v, []) + [X[i]]
    return inv_map


def model1(t, coeffs):
    lamb1 = coeffs[0]
    eval1 = poisson.pmf(np.floor(t), lamb1)
    eval2 = poisson.pmf(np.ceil(t), lamb1)
    return eval1 + (t - np.floor(t)) * (eval2 - eval1)


def residuals1(coeffs, y, t):
    return y - model1(t, coeffs)


def model2(t, coeffs):
    a = coeffs[0]
    lamb1 = coeffs[1]
    lamb2 = coeffs[2]
    return (a * model1(t, [lamb1]) + model1(t, [lamb2])) / (1.0 + a)


def residuals2(coeffs, y, t):
    return y - model2(t, coeffs)


def model3_fix_params(lamb1, lamb2):
    def model3(t, coeffs):
        a = coeffs[0]
        return model2(t, [a, lamb1, lamb2])

    return model3


def residuals3_fix_params(lamb1, lamb2):
    model3 = model3_fix_params(lamb1, lamb2)

    def residuals3(coeffs, y, t):
        return y - model3(t, coeffs)

    return residuals3


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


def fit_poissons_fixed_means(X, lamb1, lamb2):
    residuals3 = residuals3_fix_params(lamb1, lamb2)
    entries, bins = np.histogram(X, bins="auto", density=True)
    bins = (bins[1:] + bins[:-1]) / 2
    parameters = least_squares(residuals3, [0.5], args=(entries, bins))
    parameters.x = np.append(parameters.x, [lamb1, lamb2])
    return parameters


def one_or_two_mixtures(X, alpha=0.05, min_dist=0.1, min_zscore=2):
    column = np.array(X).reshape(-1, 1)
    gm = GM(n_components=2).fit(column)
    inv_map = trygmonvector(gm, X)
    mean = s.mean(X)
    std = s.stdev(X)

    if len(inv_map) <= 1 or len(inv_map[0]) < 3 or len(inv_map[1]) < 3:
        gm = GM(n_components=1).fit(column)
        mi = confint(X)
        return {"data": X, "mean": mean, "std": std, "gm": gm, "low_means": [mi["low"]], "high_means": [mi["high"]],
                "n": [len(X)]}

    mi1 = confint(inv_map[0], alpha=alpha)
    mi2 = confint(inv_map[1], alpha=alpha)
    if dist(mi1, mi2) <= min_dist or abs(gm.means_[1][0] - gm.means_[0][0]) / (max(gm.covariances_)[0][0]) < min_zscore:
        gm = GM(n_components=1).fit(column)
        mi = confint(X)
        result = {"data": X, "mean": mean, "std": std, "gm": gm, "low_means": [mi["low"]], "high_means": [mi["high"]],
                  "n": [len(X)]}
    elif mi1["low"] < mi2["low"]:
        result = {"data": X, "mean": mean, "std": std, "gm": gm, "label_order": [0, 1],
                  "low_means": [mi1["low"], mi2["low"]], "high_means": [mi1["high"], mi2["high"]],
                  "n": [mi1["n"], mi2["n"]]}
    else:
        result = {"data": X, "mean": mean, "std": std, "gm": gm, "label_order": [1, 0],
                  "low_means": [mi2["low"], mi1["low"]], "high_means": [mi2["high"], mi1["high"]],
                  "n": [mi2["n"], mi1["n"]]}
    return result

# First transform X into log(X)+c such that it does not go below 0, X is a list
def fit_poissons(X, alpha=0.05, min_dist=0.1, min_zscore=2):
    if np.mean(X) < 5:  # Can't really form a good statistic
        meanbounds = sms.DescrStatsW(X).tconfint_mean(alpha=alpha)
        return {"n": 1, "coeffs": [meanbounds[1]]}
    shift = np.min(X) - 1  # Needed later to shift back
    Xarr = np.log(X - shift)
    res = one_or_two_mixtures(Xarr.tolist(), alpha=0.05, min_dist=min_dist, min_zscore=min_zscore)
    numcomponents = len(res["low_means"])
    if numcomponents == 2:
        mean1 = 0.5 * (res["low_means"][0] + res["high_means"][0])
        mean2 = 0.5 * (res["low_means"][1] + res["high_means"][1])
        mean1 = np.exp(mean1) + shift
        mean2 = np.exp(mean2) + shift
        sz1 = res["n"][0]
        sz2 = res["n"][1]
        alpha = sz1 / sz2
        # Now optimize with estimates
        # coeffs = fit_data_two_poissons(X, [alpha, mean1, mean2])
        coeffs_fm = fit_poissons_fixed_means(X, mean1, mean2)
        print("Optimality fm = {}".format(coeffs_fm.cost))
        coeffs_2 = fit_data_two_poissons(X, [alpha, mean1, mean2])
        print("Optimality 2 = {}".format(coeffs_2.cost))
        coeffs_1 = fit_data_one_poisson(X, [np.mean(X)])
        print("Optimality 1 = {}".format(coeffs_1.cost))
        if coeffs_2.cost < coeffs_fm.cost:
            coeffs = coeffs_2
        else:
            coeffs = coeffs_fm
        if coeffs.x[0] > 0.0 and 2 * coeffs.cost < coeffs_1.cost:
            return {"n": 2, "coeffs": coeffs}

    print("Only have one!")
    Xarr = np.array(X)
    mean1 = np.mean(Xarr)
    mean2 = mean1 + min_zscore * np.sqrt(mean1)
    mean1 = np.mean(Xarr[Xarr < mean2 - np.sqrt(mean2) / 2.0])

    coeffs = fit_poissons_fixed_means(X, mean1, mean2)
    print("Alpha = {}".format(coeffs.x[0]))
    # coeffs = fit_data_one_poisson(X, [mean1])
    return {"n": 2, "coeffs": coeffs}


def percent_upregulation(a):
    return 100.0*(1.0/(a+1.0))


def upregulation_from_gaussian(X, mean):
    X = np.array(X)
    return 100.0*np.size(X[X-mean > 0]) / np.size(X)

def get_poisson_weight_with_statistic(row, params):   # Returns a
    coeffs_fm = fit_poissons_fixed_means(row, params["coeffs"].x[1], params["coeffs"].x[2]) # Use the statistic means
    print("Optimality fm = {}".format(coeffs_fm.cost))
    return coeffs_fm.x[0]


def plot_hist(row, label=""):
    sns.histplot(row, binwidth=1.0, label=label, element="step", fill=False, kde=True, kde_kws={"bw_adjust":2})                 # Plot histogram for cluster


def plot_two_poissons_from_params(row, params, color="r", label=""):   # Return the bins
    bins=np.arange(np.min(row), np.max(row))
    X, bins = np.histogram(row, bins=bins, density=True)   # Sample histogram to pull out bins
    mbins = (bins[1:] + bins[:-1]) / 2.0
    plot_hist(row, label=label)
    plt.axvline(x=params["coeffs"].x[1], color=color)          # Show where the ref's distributions are
    plt.axvline(x=params["coeffs"].x[2], color=color)          # Show where the ref's distributions are
    return mbins


def plot_predict(row, params, color="r", alpha=0.05, min_dist=0.1, min_zscore = 2, label=""):
    if params["n"] == 1:
        pt_up = upregulation_from_gaussian(row, params["coeffs"][0])
        plot_hist(row, label=label+" +{:.0f}%".format(pt_up))
        return pt_up

    a = get_poisson_weight_with_statistic(row, params)
    params["coeffs"].x[0] = a                   # Update params with fitted a, this may side-effect
    pt_up = percent_upregulation(a)
    plot_two_poissons_from_params(row, params, label=label+" +{:.0f}%".format(pt_up))
    return pt_up


def plot_fits(row, color="r", alpha=0.05, min_dist=0.1, min_zscore = 2, label=""):  # Return the fitted params
    params = fit_poissons(row, alpha=alpha, min_dist=min_dist, min_zscore=min_zscore)
    if params["n"] == 1:
        pt_up = upregulation_from_gaussian(row, params["coeffs"][0])
        plot_hist(row, label=label+" +{:.0f}%".format(pt_up))
    else:
        pt_up = percent_upregulation(params["coeffs"].x[0])
        bins = plot_two_poissons_from_params(row, params, label=label+" +{:.0f}%".format(pt_up))
    return params


def parse(st):
    return list(map(lambda s: s.strip(), list(filter(lambda s: s != "", st.split(',')))))


def main():
    gn = Granatum()

    sample_coords = gn.get_import("viz_data")
    df = gn.pandas_from_assay(gn.get_import("assay"))
    gene_ids = parse(gn.get_arg("gene_ids"))
    groups = gn.get_import("groups")
    alpha = 1.0-gn.get_arg("confint")/100.0
    min_zscore = st.norm.ppf(gn.get_arg("confint"))
    min_dist = 0.1

    coords = sample_coords.get("coords")
    dim_names = sample_coords.get("dimNames")

    inv_map = {}
    for k, v in groups.items():
        inv_map[v] = inv_map.get(v, []) + [k]

    for gene in gene_ids:
        plt.figure()
        # First form a statistic for all values, also puts out plot
        params = plot_fits(df.loc[gene, :].to_list(), color="r", alpha=alpha, min_dist=min_dist, min_zscore=min_zscore, label="All")
        for k, v in inv_map.items():

            plt.subplot(1, 1, 1)
            plt.title('Gene expression level distribution for each cluster')
            plot_predict(df.loc[gene, v].to_list(), params, label=k)
            # sns.distplot(df.loc[gene,:].to_list(), bins=int(100), color = 'darkblue', kde_kws={'linewidth': 2})
            plt.ylabel('Frequency')
            plt.xlabel('Gene expression')

        plt.legend()
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
