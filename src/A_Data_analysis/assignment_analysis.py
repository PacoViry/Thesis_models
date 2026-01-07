import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors
from PIL import Image
import glob
import time as tm
from scipy.interpolate import interp1d
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
# import networkx as nx
# import imageio.v2 as imageio
# from itables import show
import importlib
from src.Common_tools import vis_ref
importlib.reload(vis_ref)


#The entry dataset can contain the following columns :
# 'ADEP', 'ADEP Name',  'ADES', 'ADES Name', 'Period', 'Aircraft Type', 'Aircraft Type Name', 'Aircraft Operator','Aircraft Operator Name',
# 'N_flights', 'Seats', 'ASK', 'Activity', 'Distance_conn (km)', 'Flights_conn_p',
# 'Seats_conn_p', 'ASK_conn_p', 'Activity_conn_p', 'Weights'
#The following module allows :
# 1) to vizualize the market composition regarding aircraft operator or aircraft type depending on different indicators
# 2) to visualize the market shares of a selection regarding aircraft operator or aircraft type depending on different indicators
# 3) to observe market composition and share over longer periods
# the market composition observations need to be harmonious in terms of colors and heights


def market_vis(df, name_fig ='test_market', title_fig = None,  color_mix = vis_ref.colors_22, market = 'Aircraft Type',
                  Observation = 'ASK', Dist_limits =[4e2, 1.5e4], Capac_limits = [9e3, 4e6],
                  ask_d_ref = None, M_x_ref = None, M_y_ref = None,
                  N_market = 12, reso = 400, smooth_param = 0.05, period_duration = 1, weight = False):
    #On peut étudier les flux par aéroport, opérateur, avion. Selon les ASKs, sieges ou nombre de vols
    #dimensions du graphe et du lissage
    fontsize_legend  = int(60/N_market**0.7)
    dimensions = (14,10)
    width_grid, height_grid = 21,8
    width_main, height_main = 17, 5
    width_sec, height_sec = 3, 5
    if weight == True :
        print('cas a construire')
    smooth_x = (np.log(Dist_limits[1]-np.log(Dist_limits[0])))*smooth_param/(dimensions[0]*width_main/width_grid)
    smooth_y = (np.log(Capac_limits[1]-np.log(Capac_limits[0])))*smooth_param/(dimensions[1]*height_main/height_grid)
    fig = plt.figure(figsize=dimensions)
    grid = plt.GridSpec(height_grid, width_grid, hspace=0.0, wspace=0.0)
    main_ax = fig.add_subplot(grid[height_grid-height_main:height_grid, 0:width_main])
    x_density_ax = fig.add_subplot(grid[0:height_grid-height_main, 0:width_main])
    y_density_ax = fig.add_subplot(grid[height_grid-height_sec:height_grid, width_main:width_main+width_sec])
    cax = fig.add_subplot(grid[height_grid-height_main:height_grid, width_main+width_sec:width_grid])

    #Calcul de la distribution globales
    int_traff = df[list(set(['ADEP', 'ADES', 'Distance_conn (km)', 'Seats_conn_p', Observation + '_conn_p']))].drop_duplicates(
        subset=['ADES', 'ADEP'], keep='first')[list(set(['Distance_conn (km)', 'Seats_conn_p', Observation + '_conn_p']))]
    market_types = df[[market, market+' Name', Observation]].groupby([market, market+' Name']).sum().sort_values(by=Observation,ascending=False).reset_index()
    selec = market_types[:N_market][market]
    selec_n = market_types[:N_market][market+' Name']
    traff_matrix, x, y = vis_ref.smooth_ln_2d(np.array(int_traff['Distance_conn (km)']), np.array(int_traff['Seats_conn_p']),
                                np.array(int_traff[Observation + '_conn_p']), Dist_limits, Capac_limits,reso,
                                smooth_x, smooth_y)
    traff_matrix = traff_matrix.transpose()
    X = np.exp(x)
    Y = np.exp(y)

    #Calcul des distributions marginales de chaque marché
    Input_ac_n = np.array(df[market])
    Input_ac = np.array(df[['Distance_conn (km)', 'Seats_conn_p', Observation]])
    Moys = []
    Res_x_e = np.zeros((N_market + 2, reso))
    Res_y_e = np.zeros((N_market + 2, reso))
    Obs_tots = []
    for i in range(N_market):
        Input_i = Input_ac[Input_ac_n == selec[i],:]
        Moys.append(np.array([(Input_i[:, 0]* Input_i[:, 2]).sum() / (Input_i[:, 2].sum()),
                    (Input_i[:, 1]* Input_i[:, 2]).sum() / (Input_i[:, 2].sum())]))
        Res_x_e[i + 1, :] = vis_ref.smooth_ln_1d(Input_i[:, 0],Input_i[:, 2],Dist_limits,reso,  smooth_param_x = smooth_x)
        Res_y_e[i + 1, :] = vis_ref.smooth_ln_1d(Input_i[:, 1],Input_i[:, 2], Capac_limits,reso,  smooth_param_x = smooth_y)
        Obs_tots.append(Input_i[:, 2].sum())
    Input_i = Input_ac[~(np.isin(Input_ac_n, np.array(selec))), :]
    Res_x_e[N_market + 1, :] = vis_ref.smooth_ln_1d(Input_i[:, 0],Input_i[:, 2],Dist_limits,reso,  smooth_param_x = smooth_x)
    Res_y_e[N_market + 1, :] = vis_ref.smooth_ln_1d(Input_i[:, 1],Input_i[:, 2], Capac_limits,reso,  smooth_param_x = smooth_y)
    Obs_tots.append(Input_i[:, 2].sum())
    Res_x_e = Res_x_e.cumsum(axis=0)
    Res_y_e = Res_y_e.cumsum(axis=0)




    # Calcul des quantiles
    if ask_d_ref == None:
        ask_d_ref = traff_matrix.max()
    if M_x_ref == None:
        M_x_ref = Res_x_e.max() * 1.05
    if M_y_ref == None:
        M_y_ref = Res_y_e.max() * 1.2
    traff_matrix = traff_matrix/ask_d_ref
    quanti = np.array([0.5, 0.8, 0.95, 0.99])
    traf_quant, traf_cum, boundaries = vis_ref.quantile_classification(traff_matrix, quanti)

    # Visualisation des données
    pcm = main_ax.pcolormesh(X, Y, traff_matrix,
                        cmap='Spectral_r',
                        shading='gouraud')
    contour = main_ax.contour(X, Y, 100*np.array(traf_cum),levels = 100*np.array(quanti), colors = 'black',
                              linewidths=[1.6, 0.9, 0.9, 0.5], linestyles = ['-', '-', '--', '--'],zorder = 2)
    contour2 = main_ax.contour(X, Y, 100*np.array(traf_cum), levels=100*np.array(quanti), colors='black',alpha = 0.3,
                               linewidths=[1.6, 0.9, 0.9, 0.5], linestyles = ['-', '-', '--', '--'], zorder=1)
    manual_positions = [
        (Dist_limits[0] * (Dist_limits[1] / Dist_limits[0]) ** 0.45,
         Capac_limits[0] * (Capac_limits[1] / Capac_limits[0]) ** 0.40),
        (Dist_limits[0] * (Dist_limits[1] / Dist_limits[0]) ** 0.55,
         Capac_limits[0] * (Capac_limits[1] / Capac_limits[0]) ** 0.45),
        (Dist_limits[0] * (Dist_limits[1] / Dist_limits[0]) ** 0.6,
         Capac_limits[0] * (Capac_limits[1] / Capac_limits[0]) ** 0.55),
    ]
    clabels1 = main_ax.clabel(contour, inline=True, fontsize=12, inline_spacing=8, manual=manual_positions)
    for label in clabels1:
        label.set_bbox({'facecolor': 'none', 'alpha': 0.9, 'edgecolor': 'none'})
        label.set_text(label.get_text() + '%')
    for j in range(len(Moys)):
        main_ax.scatter(Moys[j][0], Moys[j][1], alpha=1, marker='o',
                        color=color_mix[j], s=250, linewidth=1, edgecolor='black', zorder=3)
        main_ax.scatter(Moys[j][0], Moys[j][1], alpha=1, marker='+', s=250, color='black', zorder=3)
    main_ax.set_xlabel('Route distance (km)', fontsize=17, color='darkblue')
    main_ax.set_ylabel('Route capacity (seats/year)', fontsize=17, color='darkred')
    main_ax.set_xscale('log')
    main_ax.set_yscale('log')
    # Tous les ticks logarithmiques
    main_ax.xaxis.set_major_locator(mticker.LogLocator(base=10,subs=(1,2,5)))
    main_ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(3,4,6,7,8,9)))
    main_ax.yaxis.set_major_locator(mticker.LogLocator(base=10,subs=(1,2,5)))
    main_ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(3,4,6,7,8,9)))
    # Formatter commun aux majeurs et mineurs
    formatter = mticker.FuncFormatter(vis_ref.log_125_formatter)
    main_ax.xaxis.set_major_formatter(formatter)
    main_ax.xaxis.set_minor_formatter(formatter)
    main_ax.yaxis.set_major_formatter(formatter)
    main_ax.yaxis.set_minor_formatter(formatter)
    main_ax.grid(
        True,
        axis='both',
        which= 'major',
        linestyle='--',
        linewidth=0.5,
        color='0.2'
    )

    exponent = int(np.log(market_types[Observation][0])/np.log(10))
    for j in range(len(Moys)): #posera peut être un jour problème pour bien garder un ordre correct, ne pas trier et reordonner? (regarder l'ancien code)
        x_density_ax.fill_between(X,Res_x_e[j], Res_x_e[j+1],color = color_mix[j],
                    label = str(selec_n[j])+' '+str(int(10**-(exponent-2)*(market_types[Observation][j])+0.5)/100)+'e'+str(exponent)+ ' '+ Observation + ', '
                     + str(int(1000*(market_types[Observation][j])/market_types[Observation].sum()+0.5)/10)+'%',edgecolor='black', linewidth=0.5)
    x_density_ax.fill_between(X, Res_x_e[len(Moys)], Res_x_e[len(Moys) + 1], color='0.5',
                              label= 'Others ' + str(int(10 ** -(exponent-2) * (
                              market_types[Observation][j:].sum()) + 0.5) / 100) + 'e'+str(exponent)+ ' ' + Observation + ', '
                                    + str(int(1000 * (market_types[Observation][j:].sum()) / market_types[Observation].sum() + 0.5) / 10) + '%', edgecolor='black', linewidth=0.5)
    x_density_ax.plot(X, Res_x_e[len(Moys) + 1], color='0', linewidth=1.5)
    leg = x_density_ax.legend(loc='upper left',bbox_to_anchor = (1,1.19),ncols = 1, fontsize = fontsize_legend, framealpha = 0,edgecolor='none')
    leg.set_zorder(5)

    x_density_ax.set_ylabel(Observation +'\n'+'distribution (1)', fontsize=14)
    x_density_ax.set_xscale('log')
    x_density_ax.set_ylim(0, M_x_ref)
    x_density_ax.set_xlim(Dist_limits[0], Dist_limits[1])
    x_density_ax.set_yticks([])  # Cacher les ticks sur cet axe
    # x_density_ax.set_xticks([])  # Cacher les ticks sur cet axe
    x_density_ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
    x_density_ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9)))

    # Masquer ticks et labels
    x_density_ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False
    )
    # Grille verticale uniquement
    x_density_ax.grid(
        True,
        axis='x',
        which='major',
        linestyle='--',
        linewidth=0.5,
        color='0.3'
    )
    x_density_ax.set_axisbelow(True)


    for j in range(len(Moys)):
        y_density_ax.fill_betweenx(Y,Res_y_e[j], Res_y_e[j+1],color = color_mix[j],edgecolor='black', linewidth=0.5)
    y_density_ax.fill_betweenx(Y, Res_y_e[len(Moys)], Res_y_e[len(Moys)+ 1], color='0.5', edgecolor='black', linewidth=1)
    y_density_ax.set_yscale('log')
    y_density_ax.set_xlim(0, M_y_ref)
    y_density_ax.set_ylim(Capac_limits[0], Capac_limits[1])
    y_density_ax.set_xticks([])  # Cacher les ticks sur cet axe
    y_density_ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
    y_density_ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9)))
    y_density_ax.set_xlabel(Observation +'\n'+'distribution (2)', fontsize=14)
    y_density_ax.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=False
    )
    # Grille horizontale uniquement
    y_density_ax.grid(
        True,
        axis='y',
        which='major',
        linestyle='--',
        linewidth=0.5,
        color='0.3'
    )
    y_density_ax.set_axisbelow(True)


    cbar = plt.colorbar(pcm, cax=cax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(Observation+' density (normalized)', fontsize=14)
    for b, style, width in zip(boundaries, ['-','-','--','--'], [1.6, 0.9, 0.9, 0.5]):
        cbar.ax.hlines(
            y=b,
            xmin=0,
            xmax=1,
            colors='black',
            linestyles=style,
            linewidth=width,
            transform=cbar.ax.get_yaxis_transform()
        )
    if title_fig != None :
        fig.suptitle(title_fig, fontsize=18, fontweight='bold', y=0.95, x = 0.3, ha = 'left')
    plt.savefig('figures/'+name_fig+'.pdf', format = 'pdf')
    plt.show()
    return(None)

def assign_vis(df, market_seg, name_fig ='test_assign', title_fig = None, market = 'Aircraft Type', Observation = 'ASK',
               Dist_limits =[4e2, 1.5e4], Capac_limits = [9e3, 4e6],reso=400,
               smooth_param = 0.05, period_duration = 1, weight = False, vmax = None):
    # market_seg est une liste contenant tous les identifiants à étudier.

    dimensions = (14, 10)
    width_grid, height_grid = 21, 8
    width_main, height_main = 17, 5
    if weight == True:
        print('cas a construire')
    smooth_x = (np.log(Dist_limits[1] - np.log(Dist_limits[0]))) * smooth_param / (
                dimensions[0] * width_main / width_grid)
    smooth_y = (np.log(Capac_limits[1] - np.log(Capac_limits[0]))) * smooth_param / (
                dimensions[1] * height_main / height_grid) #seulement pour comier le lissage de la figure précédent, peut être un sujet à adapter
    dimensions = (9, 6)
    fig = plt.figure(figsize=dimensions)

    # Calcul de la distribution globales
    int_traff = df[list(set(['ADEP', 'ADES', 'Distance_conn (km)', 'Seats_conn_p', Observation + '_conn_p']))].drop_duplicates(
        subset=['ADES', 'ADEP'], keep='first')[
        list(set(['ADEP', 'ADES', 'Distance_conn (km)', 'Seats_conn_p', Observation + '_conn_p']))]
    traff_matrix, x, y = vis_ref.smooth_ln_2d(np.array(int_traff['Distance_conn (km)']),
                                              np.array(int_traff['Seats_conn_p']),
                                              np.array(int_traff[Observation + '_conn_p']), Dist_limits, Capac_limits,
                                              reso,
                                              smooth_x, smooth_y)
    traff_matrix = traff_matrix.transpose()
    X = np.exp(x)
    Y = np.exp(y)
    quanti = np.array([0.5, 0.8, 0.95, 0.99])
    traf_quant, traf_cum, boundaries = vis_ref.quantile_classification(traff_matrix, quanti)

    # Calcul de la distribution du segment de marché
    df_selec = df[df[market].isin(market_seg)]
    df_selec = pd.merge(df_selec[['ADEP', 'ADES', Observation]], int_traff, on=['ADEP', 'ADES'], how='left')
    traff_matrix_selec, x, y = vis_ref.smooth_ln_2d(np.array(df_selec['Distance_conn (km)']),
                                              np.array(df_selec['Seats_conn_p']),
                                              np.array(df_selec[Observation]), Dist_limits, Capac_limits,
                                              reso,
                                              smooth_x, smooth_y)
    traff_matrix_selec = traff_matrix_selec.transpose()


    ratio = traff_matrix_selec/traff_matrix
    if vmax == None :
        vmax = np.max(ratio)
    plt.pcolormesh(X, Y, traff_matrix_selec/traff_matrix,
                cmap='Spectral_r', shading='gouraud', vmax=vmax)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(Observation + ' share (%)', fontsize=14)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 100:.0f}%'))
    contour = plt.contour(X, Y, 100 * np.array(traf_cum), levels=100 * np.array(quanti), colors=['black','black','black','0.3'],
                              linewidths=[1.6, 0.9, 0.9, 1.5], linestyles=['-', '-', '--', '-'], zorder=4)
    contour2 = plt.contour(X, Y, 100 * np.array(traf_cum), levels=100 * np.array(quanti), colors='black',alpha = 0.3,
                               linewidths=[1.6, 0.9, 0.9, 1.5], linestyles=['-', '-', '--', '-'], zorder=3)
    manual_positions = [
        (Dist_limits[0] * (Dist_limits[1] / Dist_limits[0]) ** 0.45,
         Capac_limits[0] * (Capac_limits[1] / Capac_limits[0]) ** 0.40),
        (Dist_limits[0] * (Dist_limits[1] / Dist_limits[0]) ** 0.55,
         Capac_limits[0] * (Capac_limits[1] / Capac_limits[0]) ** 0.45),
        (Dist_limits[0] * (Dist_limits[1] / Dist_limits[0]) ** 0.6,
         Capac_limits[0] * (Capac_limits[1] / Capac_limits[0]) ** 0.55),
    ]
    clabels1 = plt.clabel(contour, inline=True, fontsize=12, inline_spacing=4, manual=manual_positions)
    for label in clabels1:
        label.set_bbox({'facecolor': 'none', 'alpha': 0.9, 'edgecolor': 'none'})
        label.set_text(label.get_text() + '%')
    contourf = plt.contourf(X,Y,traf_cum, levels=[0.99, 1], colors='white', aspect='auto', zorder=2, alpha=1)

    L_boundaries = [[Dist_limits[0],Dist_limits[1]], [Capac_limits[0],Capac_limits[1]]]
    results = []
    L_deb = [5,2]
    L_fin = [1,2,5]
    for lims in L_boundaries:
        deb = np.exp(np.log(lims[0])-np.log(10)*int(np.log10(lims[0])))
        if deb > 5 :
            ind_1 = 0
        elif deb > 2 :
            ind_1 = 1
        else :
            ind_1 = 2
        fin = np.exp(np.log(lims[1]/5)-np.log(10)*int(np.log10(lims[1]/5)))
        if fin < 2:
            ind_2 = 0
        elif fin < 4 :
            ind_2 = 1
        else :
            ind_2 = 2
        expo =  int(np.log10(lims[0]))+1
        delta_e = int(np.log10(lims[1]/5))- int(np.log10(lims[0]))
        results.append([ind_1, ind_2, expo, delta_e])
    for i in range(results[0][0]):
        val = L_deb[i]*10**(results[0][2]-1)
        plt.plot([val, val], [Capac_limits[0],Capac_limits[1]], color='black', linestyle='--', linewidth=0.3, alpha=0.5, zorder = 3)
    for j in range(results[0][3]):
        for u in L_fin :
            val = u * 10**(results[0][2] + j)
            plt.plot([val, val], [Capac_limits[0],Capac_limits[1]], color='black', linestyle='--', linewidth=0.3, alpha=0.5, zorder = 3)
    for i in range(results[0][1]):
        val = L_fin[i] * 10 ** (results[0][2] +results[0][3])
        plt.plot([val, val], [Capac_limits[0], Capac_limits[1]], color='black', linestyle='--', linewidth=0.3, zorder = 3,
                 alpha=0.5)
    for i in range(results[1][0]):
        val = L_deb[i]*10**(results[1][2]-1)
        plt.plot([Dist_limits[0],Dist_limits[1]], [val, val], color='black', linestyle='--', linewidth=0.3, alpha=0.5, zorder = 3)
    for j in range(results[1][3]):
        for u in L_fin :
            val = u * 10**(results[1][2] + j)
            plt.plot([Dist_limits[0],Dist_limits[1]],[val, val], color='black', linestyle='--', linewidth=0.3, alpha=0.5, zorder = 3)
    for i in range(results[1][1]):
        val = L_fin[i] * 10 ** (results[1][2] +results[1][3])
        plt.plot([Dist_limits[0], Dist_limits[1]],[val, val], color='black', linestyle='--', linewidth=0.3, zorder = 3,
                 alpha=0.5)
    plt.xlim(Dist_limits[0], Dist_limits[1])
    plt.ylim(Capac_limits[0], Capac_limits[1])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Route distance (km)', fontsize=14, color='darkblue')
    plt.ylabel('Route capacity (seats/year)', fontsize=14, color='darkred')
    plt.gca().xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
    plt.gca().xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9)))
    plt.gca().yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
    plt.gca().yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9)))
    # Formatter commun aux majeurs et mineurs
    formatter = mticker.FuncFormatter(vis_ref.log_125_formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_minor_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_minor_formatter(formatter)
    if title_fig != None :
        fig.suptitle(title_fig, fontsize=18, fontweight='bold', y=0.95, x = 0.3, ha = 'left')
    plt.savefig('figures/'+name_fig+'.pdf', format = 'pdf')
    plt.show()
    return None

def dynamic_market_visualisation():
    return(None)