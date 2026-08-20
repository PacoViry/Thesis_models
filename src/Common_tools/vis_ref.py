#module for colors and markers.
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import gaussian_filter
from scipy import integrate, optimize
import numpy as np
import networkx as nx
import pandas as pd
print(sys.path)

###Définition des couleurs
colors_5 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
colors_10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
colors_22 = [ '0.1',"#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5", '0.95', 'purple']
colors_26 = [
    "#E41A1C",  # rouge vif
    "#377EB8",  # bleu intense
    "#4DAF4A",  # vert vif
    "#984EA3",  # violet profond
    "#FF7F00",  # orange vif
    "#FFFF33",  # jaune lumineux
    "#A65628",  # brun foncé
    "#F781BF",  # rose vif
    "#00FFFF",  # cyan pur
    "#FF00FF",  # magenta pur
    "#008080",  # turquoise foncé
    "#3F51B5",  # indigo
    "#B2FF00",  # vert lime
    "#0A2A4F",  # bleu nuit,
    "0.85", #blanc gris
    "#FF6F61",  # corail
    "#808000",  # vert olive
    "#87CEEB",  # bleu ciel
    "#B22222",  # rouge brique
    "#CFA0E9",  # lavande
    "#FFD700",  # jaune or
    "#228B22",  # vert forêt
    "#003F5C",  # bleu pétrole
    "#D100D1",  # fuchsia
    "#000000",  # noir
]
marker_type = ['s','o','^','P','x','*']
sizes = [30,30,35,40,40,60]
symb = ['o', 's', '^', 'v', '<', '>', 'd', 'p', 'h', '*', 'D', 'x', 'P', 'h', '8']

## Visualisation des couleurs
def vis_colors(color):
    for i in range(len(color)):
        plt.barh(i, 1, color=color[i], edgecolor='k')
    plt.show()

## Lissage 1D
def smooth_ln_1d(x, z, limits_x, reso, factor=4, smooth_param_x=1.0):
    h, xedges = np.histogram(
        np.log(x),
        bins=reso * factor,
        weights=z,
        range=np.log(limits_x)
    )
    h_smooth = gaussian_filter(h, sigma=smooth_param_x/((xedges[-1]-xedges[0])/len(xedges)))
    h_small = h_smooth.reshape(h_smooth.shape[0] // factor, factor).sum(axis=1)
    xi = 0.5 * (xedges[1:] + xedges[:-1])
    xi_small = xi.reshape(-1, factor).mean(axis=1)
    return h_small, xi_small

## Lissage 2D
def smooth_ln_2d(x, y, z, limits_x, limits_y, reso, smooth_param_x, smooth_param_y, factor = 4):
    h, xedges, yedges = np.histogram2d(np.log(x), np.log(y), bins=reso*factor, weights = z, range=[np.log(limits_x), np.log(limits_y)])
    z = gaussian_filter(h, sigma=[smooth_param_x/((xedges[-1]-xedges[0])/len(xedges)), smooth_param_y/((yedges[-1]-yedges[0])/len(yedges))])
    xi = 0.5 * (xedges[1:] + xedges[:-1])
    yi = 0.5 * (yedges[1:] + yedges[:-1])
    xi_small = xi.reshape(-1, factor).mean(axis=1)
    yi_small = yi.reshape(-1, factor).mean(axis=1)
    z_small = z.reshape(z.shape[0]//factor, factor, z.shape[1]//factor, factor).sum(axis=(1,3))
    return z_small, xi_small, yi_small

## Quantile matrix
def quantile_classification(array, quant):
    flat_t = array.flatten()

    # tri décroissant
    sorted_indices = np.argsort(flat_t)[::-1]
    sorted_t = flat_t[sorted_indices]

    cumulative_sum = np.cumsum(sorted_t)
    normalized_cumulative_sum = cumulative_sum / cumulative_sum[-1]

    quant = np.sort(quant)

    # --- frontières de quantiles ---
    boundaries = []
    for q in quant:
        idx = np.searchsorted(normalized_cumulative_sum, q)
        boundaries.append(sorted_t[idx])

    boundaries = np.array(boundaries)

    # --- classification ---
    classes = np.zeros_like(flat_t, dtype=int)
    for i, q in enumerate(quant):
        classes[normalized_cumulative_sum >= q] = i + 1
    classes[normalized_cumulative_sum > quant[-1]] = len(quant) + 1

    class_array = np.zeros_like(array, dtype=int)
    class_array.flat[sorted_indices] = classes

    cumulative_sum_array = np.zeros_like(flat_t, dtype=float)
    cumulative_sum_array[sorted_indices] = normalized_cumulative_sum
    cumulative_sum_array = cumulative_sum_array.reshape(array.shape)

    return class_array, cumulative_sum_array, boundaries

## Formatter graphe
def log_125_formatter(val, pos=None):
    if val <= 0:
        return ""

    exponent = np.floor(np.log10(val))
    mantissa = val / 10**exponent

    # tolérance numérique
    if np.isclose(mantissa, [1, 2, 5]).any():
        return rf"${mantissa:g}.10^{{{int(exponent)}}}$"
    else:
        return ""

def color_lists(lists):
    """
    lists : liste de p listes, chaque sous-liste ayant n éléments distincts.
    Retourne : dict {élément: couleur} si trouvé, sinon None
    """
    # Construire le graphe d'intersection
    g = nx.Graph()
    for L in lists:
        for u in L:
            g.add_node(u)
        # Ajouter toutes les arêtes d'une clique
        for i in range(len(L)):
            for j in range(i+1, len(L)):
                g.add_edge(L[i], L[j])

    try:
        # Coloration gloutonne (heuristique)
        coloring = nx.coloring.greedy_color(g, strategy="largest_first")
    except Exception as e:
        print("Échec du coloriage heuristique :", e)
        return None

    # Vérification : chaque liste doit avoir toutes ses couleurs distinctes
    for L in lists:
        couleurs = [coloring[x] for x in L]
        # print(len(set(couleurs)))
        if len(set(couleurs)) < len(couleurs):
            print("Coloriage invalide pour au moins une liste.")
            return None

    return coloring

def weighted_quantile(x, w, q):
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]

    cw = np.cumsum(w_sorted)
    cw /= cw[-1]  # normalisation

    return np.interp(q, cw, x_sorted)

def date_to_float_year(date):
    if type(date)!= float :
        start_of_year = pd.Timestamp(year=date.year, month=1, day=1)
        start_of_next_year = pd.Timestamp(year=date.year + 1, month=1, day=1)
        year_length = (start_of_next_year - start_of_year).days
        days_since_start_of_year = (date - start_of_year).days
        return date.year + days_since_start_of_year / year_length
    else :
        return date


def I1(gamma, m):
    """∫₀ᵐ t(m-t) exp(γt) dt"""
    val, _ = integrate.quad(lambda t: t * (m - t) * np.exp(gamma * t), 0, m)
    return val

def I_d(gamma, m, a, b):
    """∫₀ᵐ t(m-t) exp(γt) dt"""
    val, _ = integrate.quad(lambda t: t * (m - t) * np.exp(gamma * t), a, b)
    return val

def I2(gamma, m):
    """∫₀ᵐ t²(m-t) exp(γt) dt"""
    val, _ = integrate.quad(lambda t: t**2 * (m - t) * np.exp(gamma * t), 0, m)
    return val


def solve_deliv(B, m, gamma_bounds=(-10, 10)):

    if not (0 < B / 1 < m):
        raise ValueError(
            f"B/A = {B:.4f} doit être strictement dans (0, {m}) "
            "(la moyenne de t sous f doit rester dans l'intervalle)."
        )

    ratio = B
    def g(gamma):
        i1 = I1(gamma, m)
        if abs(i1) < 1e-14:
            return np.inf
        return I2(gamma, m) / i1 - ratio
    # Vérification que g change de signe sur l'intervalle
    ga, gb = gamma_bounds
    fa, fb = g(ga), g(gb)
    if fa * fb > 0:
        raise ValueError(
            f"Impossible de trouver gamma dans [{ga}, {gb}] : "
            f"g({ga})={fa:.3f}, g({gb})={fb:.3f}. "
            "Essayez d'élargir gamma_bounds."
        )

    gamma_sol = optimize.brentq(g, ga, gb, xtol=1e-12, rtol=1e-12)
    mu_sol    = 1 / I1(gamma_sol, m)

    return mu_sol, gamma_sol