#module for colors and markers.
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import gaussian_filter
import numpy as np
print(sys.path)

###Définition des couleurs
colors_5 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
colors_10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
colors_22 = [ '0',"#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5", '0.95']
marker_type = ['+','x','o','s','^','D']
symb = ['o', 's', '^', 'v', '<', '>', 'd', 'p', 'h', '*', 'D', 'X', 'P', 'H', '8']

## Visualisation des couleurs
def vis_colors(color):
    for i in range(len(color)):
        plt.barh(i, 1, color=color[i], edgecolor='k')
    plt.show()

## Lissage 1D
def smooth_ln_1d(X, Z, limits_x, reso, factor=4, smooth_param_x=1.0):
    H, xedges = np.histogram(
        np.log(X),
        bins=reso * factor,
        weights=Z,
        range=np.log(limits_x)
    )
    H_smooth = gaussian_filter(H, sigma=smooth_param_x/((xedges[-1]-xedges[0])/len(xedges)))
    H_small = H_smooth.reshape(H_smooth.shape[0] // factor, factor).mean(axis=1)
    return H_small

## Lissage 2D
def smooth_ln_2d(X, Y, Z, limits_x, limits_y, reso, smooth_param_x, smooth_param_y, factor = 4):
    H, xedges, yedges = np.histogram2d(np.log(X), np.log(Y), bins=reso*factor, weights = Z, range=[np.log(limits_x), np.log(limits_y)])
    Z = gaussian_filter(H, sigma=[smooth_param_x/((xedges[-1]-xedges[0])/len(xedges)), smooth_param_y/((yedges[-1]-yedges[0])/len(yedges))])
    xi = 0.5 * (xedges[1:] + xedges[:-1])
    yi = 0.5 * (yedges[1:] + yedges[:-1])
    xi_small = xi.reshape(-1, factor).mean(axis=1)
    yi_small = yi.reshape(-1, factor).mean(axis=1)
    Z_small = Z.reshape(Z.shape[0]//factor, factor, Z.shape[1]//factor, factor).mean(axis=(1,3))
    return(Z_small, xi_small, yi_small)

## Quantile matrix
def quantile_classification(array, quant):
    flat_T = array.flatten()

    # Tri décroissant
    sorted_indices = np.argsort(flat_T)[::-1]
    sorted_T = flat_T[sorted_indices]

    cumulative_sum = np.cumsum(sorted_T)
    normalized_cumulative_sum = cumulative_sum / cumulative_sum[-1]

    quant = np.sort(quant)

    # --- frontières de quantiles ---
    boundaries = []
    for q in quant:
        idx = np.searchsorted(normalized_cumulative_sum, q)
        boundaries.append(sorted_T[idx])

    boundaries = np.array(boundaries)

    # --- classification ---
    classes = np.zeros_like(flat_T, dtype=int)
    for i, q in enumerate(quant):
        classes[normalized_cumulative_sum >= q] = i + 1
    classes[normalized_cumulative_sum > quant[-1]] = len(quant) + 1

    class_array = np.zeros_like(array, dtype=int)
    class_array.flat[sorted_indices] = classes

    cumulative_sum_array = np.zeros_like(flat_T, dtype=float)
    cumulative_sum_array[sorted_indices] = normalized_cumulative_sum
    cumulative_sum_array = cumulative_sum_array.reshape(array.shape)

    return class_array, cumulative_sum_array, boundaries

## Formatter graphe
def log_125_formatter(val, pos=None):
    if val <= 0:
        return ""

    exponent = np.floor(np.log10(val))
    mantissa = val / 10**exponent

    # Tolérance numérique
    if np.isclose(mantissa, [1, 2, 5]).any():
        return rf"${mantissa:g}.10^{{{int(exponent)}}}$"
    else:
        return ""