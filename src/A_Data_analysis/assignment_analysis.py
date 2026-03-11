import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from PIL import Image
import glob
from matplotlib.ticker import FuncFormatter
import imageio.v2 as imageio
# import importlib
from src.Common_tools import vis_ref
import random as rd
# importlib.reload(vis_ref)


#The entry dataset can contain the following columns :
# 'ADEP', 'ADEP Name',  'ADES', 'ADES Name', 'Period', 'Aircraft Type', 'Aircraft Type Name', 'Aircraft Operator','Aircraft Operator Name',
# 'N_flights', 'Seats', 'ASK', 'Activity','Av ac seats', 'Distance', 'Distance_conn (km)', 'Flights_conn_p',
# 'Seats_conn_p', 'ASK_conn_p', 'Activity_conn_p', 'Weight'
#The following module allows :
# 1) to vizualize the market composition regarding aircraft operator or aircraft type depending on different indicators
# 2) to visualize the market shares of a selection regarding aircraft operator or aircraft type depending on different indicators
# 3) to observe market composition and share over longer periods
# the market composition observations need to be harmonious in terms of colors and heights

def agg_ind_1(df):
    g = df.groupby(['ADEP', 'ADES', 'Period'])
    distance_conn = (pd.concat([g['Seats'].sum().rename('Seats_sum'),
                                (df['Seats'] * df['Distance'])
                               .groupby([df['ADEP'], df['ADES'], df['Period']])
                               .sum()
                               .rename('SeatsDist_sum')], axis=1)
    .assign(**{'Distance_conn (km)': lambda df: df['SeatsDist_sum'] / df['Seats_sum']})
    .reset_index()[['ADEP', 'ADES', 'Period', 'Distance_conn (km)']])

    seats_conn_p = (
        df.groupby(['ADEP', 'ADES', 'Period'])['Seats']
        .sum()
        .reset_index()
        .rename(columns={'Seats': 'Seats_conn_p'})
    )
    print('synthesis...')
    df = pd.merge(
        df,
        distance_conn,
        on=['ADEP', 'ADES', 'Period'],
        how='left'
    )
    df = pd.merge(
        df,
        seats_conn_p,
        on=['ADEP', 'ADES', 'Period'],
        how='left'
    )
    df.drop(columns=['Distance'], inplace=True)
    # df['ASK_conn_p'] = df['Seats_conn_p'] * df['Distance_conn (km)']
    df['ASK'] = df['Seats'] * df['Distance_conn (km)']
    return df

def agg_ind_s(df, observation = 'ASK'):
    obs_conn_p = (
        df.groupby(['ADEP', 'ADES', 'Period'])[observation]
        .sum()
        .reset_index()
        .rename(columns={observation: observation+'_conn_p'})
    )

    df = pd.merge(
        df,
        obs_conn_p,
        on=['ADEP', 'ADES', 'Period'],
        how='left'
    )

    return df

def market_vis(df, name_fig ='test_market', title_fig = None,  color_mix = vis_ref.colors_26, rank = None, color_rank =None,
               market = 'Aircraft Type', observation = 'ASK', dist_limits =(4e2, 1.9e4), capac_limits = (9e3, 4e6),
                  ask_d_ref = None, m_x_ref = None, m_y_ref = None, video = False, fun_mode = False,
                  n_market = 24, reso = 400, smooth_param = 0.05, period_duration = 1, weight = False, format_fig = 'pdf',
               label_size_param = 1):
    #On peut étudier les flux par aéroport, opérateur, avion. Selon les ASKs, sieges ou nombre de vols
    #dimensions du graphe et du lissage
    plt.style.use('default')
    fontsize_legend  = min(int(65/n_market**0.7),int(65/12**0.7))*label_size_param
    dimensions = (14,10)
    width_grid, height_grid = 21,8
    width_main, height_main = 17, 5
    width_sec, height_sec = 3, 5
    smooth_x = (np.log(dist_limits[1]-np.log(dist_limits[0])))*smooth_param/(dimensions[0]*width_main/width_grid)
    smooth_y = (np.log(capac_limits[1]-np.log(capac_limits[0])))*smooth_param/(dimensions[1]*height_main/height_grid)
    fig = plt.figure(figsize=dimensions)
    if fun_mode :
        fig.patch.set_facecolor((color_mix[int(rd.random()*21)]))
        fig.patch.set_alpha(0.2)  # Transparence if needed
    grid = plt.GridSpec(height_grid, width_grid, hspace=0.0, wspace=0.0)
    main_ax = fig.add_subplot(grid[height_grid-height_main:height_grid, 0:width_main])
    x_density_ax = fig.add_subplot(grid[0:height_grid-height_main, 0:width_main])
    y_density_ax = fig.add_subplot(grid[height_grid-height_sec:height_grid, width_main:width_main+width_sec])
    cax = fig.add_subplot(grid[height_grid-height_main:height_grid, width_main+width_sec:width_grid])
    if weight :
        df[observation] = df[observation] * df['Weight']
    #Calcul de la distribution globales
    market_types = df[[market, market+' Name', observation]].groupby([market, market+' Name']).sum().sort_values(by=observation,ascending=False).reset_index()
    selec_0 = market_types[:n_market]
    if rank is not None: #rearrangement de l'ordre si nécessaire
        selec_0 = selec_0.assign(_rank=selec_0[market].map(rank)).sort_values('_rank').drop(columns='_rank').reset_index()
    selec = selec_0[market]
    selec_n = selec_0[market+' Name']
    if weight:
        int_traff = df[list({'ADEP', 'ADES', 'Distance_conn (km)', 'Seats_conn_p', observation + '_conn_p',
                        'Weight'})].drop_duplicates(subset=['ADES', 'ADEP'], keep='first')[
            list({'Distance_conn (km)', 'Seats_conn_p', observation + '_conn_p', 'Weight'})]
        traff_matrix, x, y = vis_ref.smooth_ln_2d(np.array(int_traff['Distance_conn (km)']),
                                        np.array(int_traff['Seats_conn_p']) / period_duration,
                                    np.array(int_traff[observation + '_conn_p']) * np.array(
                                                int_traff['Weight']), dist_limits,capac_limits, reso, smooth_x, smooth_y)
    else:
        int_traff = df[list({'ADEP', 'ADES', 'Distance_conn (km)', 'Seats_conn_p', observation + '_conn_p'})].drop_duplicates(
        subset = ['ADES', 'ADEP'], keep = 'first')[list({'Distance_conn (km)', 'Seats_conn_p', observation + '_conn_p'})]
        traff_matrix, x, y = vis_ref.smooth_ln_2d(np.array(int_traff['Distance_conn (km)']), np.array(int_traff['Seats_conn_p']) / period_duration,
        np.array(int_traff[observation + '_conn_p']), dist_limits, capac_limits, reso, smooth_x, smooth_y)
    traff_matrix = traff_matrix.transpose()
    X = np.exp(x)
    Y = np.exp(y)
    if color_rank is None:
        color_rank = {ac: k for k, ac in enumerate(selec)}

    #Calcul des distributions marginales de chaque marché
    input_ac_n = np.array(df[market])
    input_ac = np.array(df[['Distance_conn (km)', 'Seats_conn_p', observation]])
    moys = []
    res_x_e = np.zeros((n_market + 2, reso))
    res_y_e = np.zeros((n_market + 2, reso))
    obs_tots = []
    for i in range(n_market):
        input_i = input_ac[input_ac_n == selec[i],:]
        moys.append(np.array([(input_i[:, 0]* input_i[:, 2]).sum() / (input_i[:, 2].sum()),
                    (input_i[:, 1]* input_i[:, 2]).sum() / (input_i[:, 2].sum())]))
        res_x_e[i + 1, :] = vis_ref.smooth_ln_1d(input_i[:, 0],input_i[:, 2],dist_limits,reso,  smooth_param_x = smooth_x)[0]
        res_y_e[i + 1, :] = vis_ref.smooth_ln_1d(input_i[:, 1]/period_duration,input_i[:, 2], capac_limits,reso,  smooth_param_x = smooth_y)[0]
        obs_tots.append(input_i[:, 2].sum())
    input_i = input_ac[~(np.isin(input_ac_n, np.array(selec))), :]
    res_x_e[n_market + 1, :] = vis_ref.smooth_ln_1d(input_i[:, 0],input_i[:, 2],dist_limits,reso,  smooth_param_x = smooth_x)[0]
    res_y_e[n_market + 1, :] = vis_ref.smooth_ln_1d(input_i[:, 1]/period_duration,input_i[:, 2], capac_limits,reso,  smooth_param_x = smooth_y)[0]
    obs_tots.append(input_i[:, 2].sum())
    res_x_e = res_x_e.cumsum(axis=0)
    res_y_e = res_y_e.cumsum(axis=0)

    # Calcul des quantiles
    if ask_d_ref is None:
        ask_d_ref = traff_matrix.max()
    if m_x_ref is None:
        m_x_ref = res_x_e.max() * 1.05
    if m_y_ref is None:
        m_y_ref = res_y_e.max() * 1.2
    traff_matrix = traff_matrix/ask_d_ref
    quanti = np.array([0.5, 0.8, 0.95, 0.99])
    traf_quant, traf_cum, boundaries = vis_ref.quantile_classification(traff_matrix, quanti)

    # Visualisation des données
    pcm = main_ax.pcolormesh(X, Y, traff_matrix, vmin=0, vmax=1,
                        cmap='Spectral_r', shading='gouraud')
    contour = main_ax.contour(X, Y, 100*np.array(traf_cum),levels = 100*np.array(quanti), colors = 'black',
                              linewidths=[1.6, 0.9, 0.9, 0.5], linestyles = ['-', '-', '--', '--'],zorder = 2)
    main_ax.contour(X, Y, 100*np.array(traf_cum), levels=100*np.array(quanti), colors='black',alpha = 0.3,
                               linewidths=[1.6, 0.9, 0.9, 0.5], linestyles = ['-', '-', '--', '--'], zorder=1)
    manual_positions = [
        (dist_limits[0] * (dist_limits[1] / dist_limits[0]) ** 0.2,
         capac_limits[0] * (capac_limits[1] / capac_limits[0]) ** 0),
        (dist_limits[0] * (dist_limits[1] / dist_limits[0]) ** 0.33,
         capac_limits[0] * (capac_limits[1] / capac_limits[0]) ** 0.30),
        (dist_limits[0] * (dist_limits[1] / dist_limits[0]) ** 0.43,
         capac_limits[0] * (capac_limits[1] / capac_limits[0]) ** 0.45),
        (dist_limits[0] * (dist_limits[1] / dist_limits[0]) ** 0.55,
         capac_limits[0] * (capac_limits[1] / capac_limits[0]) ** 0.55),
    ]

    clabels1 = main_ax.clabel(contour, inline=True, fontsize=12, inline_spacing=8, manual=manual_positions)
    for label in clabels1:
        label.set_bbox({'facecolor': 'none', 'alpha': 0.9, 'edgecolor': 'none'})
        label.set_text(label.get_text() + '%')
    sizes_markers = (np.array(selec_0[observation]) / market_types[observation].sum())**0.6
    for j in range(len(moys)):
        main_ax.scatter(moys[j][0], moys[j][1], alpha=1, marker='o',
                        color=color_mix[color_rank[selec[j]]], s=1500*sizes_markers[j], linewidth=1, edgecolor='black', zorder=3)
        main_ax.scatter(moys[j][0], moys[j][1], alpha=1, marker='+', s=1500*sizes_markers[j], linewidth=2*sizes_markers[j]**0.5, color='0.05', zorder=3)
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
    main_ax.grid(True, axis='both', which= 'major',linestyle='--',linewidth=0.5,color='0.2')
    exponent = int(np.log10(market_types[observation][0]))
    for j in range(len(moys)): #posera peut être un jour problème pour bien garder un ordre correct, ne pas trier et reordonner? (regarder l'ancien code)
        x_density_ax.fill_between(X,res_x_e[j], res_x_e[j+1],color = color_mix[color_rank[selec[j]]],
                    label = str(selec[j]).replace(" ", "")+' ' + str(selec_n[j])+': '+str(int(10**-(exponent-2)*(selec_0[observation][j])+0.5)/100)+'e'+str(exponent)+ ' '+ observation + ', '
                     + str(int(1000*(selec_0[observation][j])/market_types[observation].sum()+0.5)/10)+'%',edgecolor='black', linewidth=0.5)
    x_density_ax.fill_between(X, res_x_e[len(moys)], res_x_e[len(moys) + 1], color='0.5',
                              label= 'Others: ' + str(int(10 ** -(exponent-2) * (
                              market_types[observation][len(moys):].sum()) + 0.5) / 100) + 'e'+str(exponent)+ ' ' + observation + ', '
                                    + str(int(1000 * (market_types[observation][len(moys):].sum()) / market_types[observation].sum() + 0.5) / 10) + '%', edgecolor='black', linewidth=0.5)
    x_density_ax.plot(X, res_x_e[len(moys) + 1], color='0', linewidth=1.5)

    if rank is not None:
        handles, labels = x_density_ax.get_legend_handles_labels()
        selec_0[market] = selec_0[market].astype(str).str.replace(' ', '', regex=False)
        rank_map = selec_0.drop_duplicates(market).set_index(market)["index"].to_dict()
        sorted_pairs = sorted(
            zip(handles, labels),
            key=lambda x_i: (
                999 if x_i[1].startswith("Others") else rank_map.get(x_i[1].split()[0], 998)
            ))
        handles, labels = zip(*sorted_pairs)
        labels = tuple(
            l if l.startswith("Others") else l.split(" ", 1)[1]
            for l in labels
        )
        leg = x_density_ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1,1.10), ncols=1,
                                  fontsize=fontsize_legend,framealpha=1, edgecolor='none',labelspacing=0.3)
    else :
        handles, labels = x_density_ax.get_legend_handles_labels()
        labels = tuple(
            l if l.startswith("Others") else l.split(" ", 1)[1]
            for l in labels
        )
        leg = x_density_ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1.10), ncols=1,
                                  fontsize=fontsize_legend, framealpha=1, edgecolor='none',labelspacing=0.3)
    leg.set_zorder(5)

    x_density_ax.set_ylabel(observation +'\n'+'distribution (1)', fontsize=14)
    x_density_ax.set_xscale('log')
    x_density_ax.set_ylim(0, m_x_ref)
    x_density_ax.set_xlim(dist_limits[0], dist_limits[1])
    x_density_ax.set_yticks([])  # Cacher les ticks sur cet axe
    # x_density_ax.set_xticks([])  # Cacher les ticks sur cet axe
    x_density_ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
    x_density_ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9)))

    # Masquer ticks et labels
    x_density_ax.tick_params(axis='x', which='both',bottom=False, top=False, labelbottom=False)
    # Grille verticale uniquement
    x_density_ax.grid(True, axis='x', which='major',linestyle='--',linewidth=0.5,color='0.3')
    x_density_ax.set_axisbelow(True)

    for j in range(len(moys)):
        y_density_ax.fill_betweenx(Y,res_y_e[j], res_y_e[j+1],color = color_mix[color_rank[selec[j]]],edgecolor='black', linewidth=0.5)
    y_density_ax.fill_betweenx(Y, res_y_e[len(moys)], res_y_e[len(moys)+ 1], color='0.5', edgecolor='black', linewidth=1)
    y_density_ax.set_yscale('log')
    y_density_ax.set_xlim(0, m_y_ref)
    y_density_ax.set_ylim(capac_limits[0], capac_limits[1])
    y_density_ax.set_xticks([])  # Cacher les ticks sur cet axe
    y_density_ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
    y_density_ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9)))
    y_density_ax.set_xlabel(observation +'\n'+'distribution (2)', fontsize=14)
    y_density_ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    # Grille horizontale uniquement
    y_density_ax.grid(True, axis='y', which='major', linestyle='--', linewidth=0.5, color='0.3')
    y_density_ax.set_axisbelow(True)

    cbar = plt.colorbar(pcm, cax=cax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(observation+' density (normalized)', fontsize=14)
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
    if title_fig is not None:
        fig.suptitle(title_fig, fontsize=18, fontweight='bold', y=0.95, x = 0.3, ha = 'left')
    if video:
        plt.savefig('figures/video_storage/'+name_fig+'.png', format = 'png')
    else :
        plt.savefig('figures/assignment_figures/'+name_fig+'.'+ format_fig, format = format_fig)
    plt.close()
    return None

def assign_vis(df, market_seg, name_fig ='test_assign', title_fig = None, market = 'Aircraft Type', observation = 'ASK',
               dist_limits =(4e2, 1.9e4), capac_limits = (9e3, 4e6),reso=400, smooth_param = 0.05, video = False,
               period_duration = 1, weight = False, vmax = None, color = 'pink', format_fig = 'pdf'):
    # market_seg est une liste contenant tous les identifiants à étudier.
    dimensions = (14, 10)
    width_grid, height_grid = 21, 8
    width_main, height_main = 17, 5
    if weight == True:
        df[observation] = df[observation]* df['Weight']
    smooth_x = (np.log(dist_limits[1] - np.log(dist_limits[0]))) * smooth_param / (
                dimensions[0] * width_main / width_grid)
    smooth_y = (np.log(capac_limits[1] - np.log(capac_limits[0]))) * smooth_param / (
                dimensions[1] * height_main / height_grid) #seulement pour comier le lissage de la figure précédent, peut être un sujet à adapter
    dimensions = (9, 6)
    fig = plt.figure(figsize=dimensions)

    # Calcul de la distribution globales
    if weight:
        int_traff = df[list({'ADEP', 'ADES', 'Distance_conn (km)', 'Seats_conn_p', observation + '_conn_p',
                        'Weight'})].drop_duplicates(subset=['ADES', 'ADEP'], keep='first')[
            list({'Distance_conn (km)', 'Seats_conn_p', observation + '_conn_p', 'Weight'})]
        traff_matrix, x, y = vis_ref.smooth_ln_2d(np.array(int_traff['Distance_conn (km)']),
                                        np.array(int_traff['Seats_conn_p']) / period_duration,
                                    np.array(int_traff[observation + '_conn_p']) * np.array(
                                                int_traff['Weight']), dist_limits,capac_limits, reso, smooth_x, smooth_y)
    else:
        int_traff = df[list({'ADEP', 'ADES', 'Distance_conn (km)', 'Seats_conn_p', observation + '_conn_p'})].drop_duplicates(
        subset = ['ADES', 'ADEP'], keep = 'first')[list({'Distance_conn (km)', 'Seats_conn_p', observation + '_conn_p'})]
        traff_matrix, x, y = vis_ref.smooth_ln_2d(np.array(int_traff['Distance_conn (km)']),
            np.array(int_traff['Seats_conn_p']) / period_duration, np.array(int_traff[observation + '_conn_p']),
            dist_limits, capac_limits, reso, smooth_x, smooth_y)
    traff_matrix = traff_matrix.transpose()
    X = np.exp(x)
    Y = np.exp(y)
    quanti = np.array([0.5, 0.8, 0.95, 0.99])
    traf_quant, traf_cum, boundaries = vis_ref.quantile_classification(traff_matrix, quanti)

    # Calcul de la distribution du segment de marché
    df_selec = df[df[market].isin(market_seg)]
    df_selec = pd.merge(df_selec[['ADEP', 'ADES', 'Period', observation]], int_traff, on=['ADEP', 'ADES', 'Period'], how='left')
    selec_D = (df_selec[observation]*df_selec['Distance_conn (km)']).sum()/df_selec[observation].sum()
    selec_C = (df_selec[observation]*df_selec['Seats_conn_p']).sum()/df_selec[observation].sum()

    traff_matrix_selec, x, y = vis_ref.smooth_ln_2d(np.array(df_selec['Distance_conn (km)']),
                                              np.array(df_selec['Seats_conn_p']),
                                              np.array(df_selec[observation]), dist_limits, capac_limits,
                                              reso,
                                              smooth_x, smooth_y)
    traff_matrix_selec = traff_matrix_selec.transpose()


    ratio = traff_matrix_selec/traff_matrix
    if vmax is None:
        vmax = np.max(ratio)
    plt.pcolormesh(X, Y, traff_matrix_selec/traff_matrix,
                cmap='Spectral_r', shading='gouraud', vmax=vmax)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(observation + ' share (%)', fontsize=14)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x_i, _: f'{x_i * 100:.0f}%'))
    contour = plt.contour(X, Y, 100 * np.array(traf_cum), levels=100 * np.array(quanti), colors=['black','black','black','0.3'],
                              linewidths=[1.6, 0.9, 0.9, 1.5], linestyles=['-', '-', '--', '-'], zorder=4)
    contour2 = plt.contour(X, Y, 100 * np.array(traf_cum), levels=100 * np.array(quanti), colors='black',alpha = 0.3,
                               linewidths=[1.6, 0.9, 0.9, 1.5], linestyles=['-', '-', '--', '-'], zorder=3)
    manual_positions = [
        (dist_limits[0] * (dist_limits[1] / dist_limits[0]) ** 0.45,
         capac_limits[0] * (capac_limits[1] / capac_limits[0]) ** 0.40),
        (dist_limits[0] * (dist_limits[1] / dist_limits[0]) ** 0.55,
         capac_limits[0] * (capac_limits[1] / capac_limits[0]) ** 0.45),
        (dist_limits[0] * (dist_limits[1] / dist_limits[0]) ** 0.6,
         capac_limits[0] * (capac_limits[1] / capac_limits[0]) ** 0.55),
    ]
    clabels1 = plt.clabel(contour, inline=True, fontsize=12, inline_spacing=4, manual=manual_positions)
    for label in clabels1:
        label.set_bbox({'facecolor': 'none', 'alpha': 0.9, 'edgecolor': 'none'})
        label.set_text(label.get_text() + '%')
    contourf = plt.contourf(X,Y,traf_cum, levels=[0.99, 1], colors='white', aspect='auto', zorder=2, alpha=1)
    plt.scatter(selec_D, selec_C, alpha=1, marker='o',
                color=color, s=250, linewidth=1, edgecolor='black', zorder=4)
    plt.scatter(selec_D, selec_C, alpha=1, marker='+', s=250, color='black', zorder=4)

    L_boundaries = [[dist_limits[0],dist_limits[1]], [capac_limits[0],capac_limits[1]]]
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
        plt.plot([val, val], [capac_limits[0],capac_limits[1]], color='black', linestyle='--', linewidth=0.3, alpha=0.5, zorder = 3)
    for j in range(results[0][3]):
        for u in L_fin :
            val = u * 10**(results[0][2] + j)
            plt.plot([val, val], [capac_limits[0],capac_limits[1]], color='black', linestyle='--', linewidth=0.3, alpha=0.5, zorder = 3)
    for i in range(results[0][1]):
        val = L_fin[i] * 10 ** (results[0][2] +results[0][3])
        plt.plot([val, val], [capac_limits[0], capac_limits[1]], color='black', linestyle='--', linewidth=0.3, zorder = 3,
                 alpha=0.5)
    for i in range(results[1][0]):
        val = L_deb[i]*10**(results[1][2]-1)
        plt.plot([dist_limits[0],dist_limits[1]], [val, val], color='black', linestyle='--', linewidth=0.3, alpha=0.5, zorder = 3)
    for j in range(results[1][3]):
        for u in L_fin :
            val = u * 10**(results[1][2] + j)
            plt.plot([dist_limits[0],dist_limits[1]],[val, val], color='black', linestyle='--', linewidth=0.3, alpha=0.5, zorder = 3)
    for i in range(results[1][1]):
        val = L_fin[i] * 10 ** (results[1][2] +results[1][3])
        plt.plot([dist_limits[0], dist_limits[1]],[val, val], color='black', linestyle='--', linewidth=0.3, zorder = 3,
                 alpha=0.5)
    plt.xlim(dist_limits[0], dist_limits[1])
    plt.ylim(capac_limits[0], capac_limits[1])
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
    if title_fig is not None:
        fig.suptitle(title_fig, fontsize=18, fontweight='bold', y=0.95, x = 0.3, ha = 'left')
    if video == True:
        plt.savefig('figures/video_storage/' + name_fig + '.png')
    else:
        plt.savefig('figures/assignment_figures/' + name_fig +'.'+ format_fig)
    plt.close()
    return None

def dynamic_market_vis(df, periods, name_periods = None, video_name = 'test_video_market', format_v = 'gif', color_mix = vis_ref.colors_22, market = 'Aircraft Type',
                    observation = 'ASK', dist_limits = (4e2, 1.9e4), capac_limits = (9e3, 4e6), frame_s = 2, period_ref = None, n_market = 12,
                       reso = 400, smooth_param = 0.05, period_duration = 1, weight = False, type_obs = None, obs_names = None,label_size_param = 1):
    selec_df = df[df['Period'].isin(periods)]
    if name_periods is None:
       name_periods = periods
    print('Color choice...')
    dimensions = (14, 10)
    width_grid, height_grid = 21, 8
    width_main, height_main = 17, 5
    if weight is True:
        selec_df[observation] = selec_df[observation]* selec_df['Weight']
    smooth_x = (np.log(dist_limits[1] - np.log(dist_limits[0]))) * smooth_param / (
            dimensions[0] * width_main / width_grid)
    smooth_y = (np.log(capac_limits[1] - np.log(capac_limits[0]))) * smooth_param / (
            dimensions[
                1] * height_main / height_grid)  # seulement pour comier le lissage de la figure précédent, peut être un sujet à adapter

    # Ordonnancement des couleurs
    Lists_c = []
    Lists_r = []
    for period in periods:
        df_p = selec_df[selec_df['Period'] == period]
        group = df_p[[market,  observation]].groupby([market]).sum().sort_values(by=market, ascending=False)
        market_types = group.sort_values(by=observation, ascending=False).reset_index()
        selec = market_types[:n_market][market]
        selec_r = market_types[market]
        Lists_r.append(list(selec_r))
        Lists_c.append(list(selec))
    sol = vis_ref.color_lists(Lists_c)
    if sol is not None:
        print("Coloriage trouvé :", sol)
    else:
        print("Pas de solution simple trouvée, attribution random")
        sol = dict(zip(Lists_c, range(2,len(Lists_c)+2)))
    #Ordonnancement des markets pour des transitions smooth entre avions ou opérateurs
    first = {}
    last = {}
    for i, list_m in enumerate(Lists_r):
        for m in list_m :
            if m not in first:
                first[m] = i
            last[m] = i
    # trier avec nouvelle règle
    types_sorted = sorted(first.keys(), key=lambda ac: (-first[ac], -last[ac]))
    rank = {ac: k for k, ac in enumerate(types_sorted)}

    #Hauteurs de références
    if period_ref is None :
        print('a coder, trouver la valeur maximum')
        return None
    else :
        df_p = selec_df[selec_df['Period']==period_ref]
        if weight :
            int_traff = df_p[list(
                list({'ADEP', 'ADES', 'Distance_conn (km)', 'Seats_conn_p', observation + '_conn_p', 'Weight'}))].drop_duplicates(
                subset=['ADES', 'ADEP'], keep='first')[
                list({'Distance_conn (km)', 'Seats_conn_p', observation + '_conn_p', 'Weight'})]
            traff_matrix, x, y = vis_ref.smooth_ln_2d(np.array(int_traff['Distance_conn (km)']),
                                                      np.array(int_traff['Seats_conn_p']),
                                                      np.array(int_traff[observation + '_conn_p'])*np.array(int_traff['Weight']), dist_limits,
                                                      capac_limits, reso,
                                                      smooth_x, smooth_y)
        else :
            int_traff = df_p[list({'ADEP', 'ADES', 'Distance_conn (km)', 'Seats_conn_p', observation + '_conn_p'})].drop_duplicates(
                subset=['ADES', 'ADEP'], keep='first')[list({'Distance_conn (km)', 'Seats_conn_p', observation + '_conn_p'})]
            traff_matrix, x, y = vis_ref.smooth_ln_2d(np.array(int_traff['Distance_conn (km)']),
                                                      np.array(int_traff['Seats_conn_p']),
                                                      np.array(int_traff[observation + '_conn_p']), dist_limits,
                                                      capac_limits, reso,
                                                      smooth_x, smooth_y)
        traff_matrix = traff_matrix.transpose()
        ask_max = traff_matrix.max()
        traff_x_max = traff_matrix.sum(axis = 0).max()*1.05
        traff_y_max = traff_matrix.sum(axis=1).max()*1.2
    selec_df = df[df['Period'].isin(periods)]
    #nettoyage des images existantes
    for filename in os.listdir('figures/video_storage/'):
        if filename.endswith(".png"):
            os.remove(os.path.join('figures/video_storage/', filename))
    #Créations des images
    for i in range(len(periods)):
        if i%3 == 0 :
            print(str(int(1000*i/len(periods)+0.5)/10)+'%')
        market_vis(selec_df[selec_df['Period'] == periods[i]],name_fig ='video_'+str(100+i), color_mix = color_mix, rank = rank, color_rank = sol,
                   market = market, observation = observation, dist_limits = dist_limits, capac_limits = capac_limits, n_market = n_market,
                   m_x_ref= traff_x_max, m_y_ref = traff_y_max, ask_d_ref = ask_max, video = True, title_fig = name_periods[i],
                    smooth_param = smooth_param, weight = weight, reso = reso,label_size_param=label_size_param)

    #Animation des images
    image_files = sorted(glob.glob("figures/video_storage/*.png"))
    if format_v == 'video':
        with imageio.get_writer('figures/integrated_scenario/'+video_name + '.mp4', fps=frame_s) as writer:
            # répéter la première image 3 fois
            for _ in range(6):
                writer.append_data(imageio.imread(image_files[0]))

            # toutes les images
            for img in image_files:
                writer.append_data(imageio.imread(img))

            # répéter la dernière image 3 fois
            for _ in range(6):
                writer.append_data(imageio.imread(image_files[-1]))

    else :
        # Charger les images
        frames = [Image.open(img) for img in image_files]
        # Sauvegarder en GIF animé
        frames[0].save('figures/integrated_scenario/'+ video_name + '.gif', save_all=True,
                       append_images=[frames[0] for i in range(3)] + frames[:] + [frames[-1] for i in range(3)],
                       duration=1000/frame_s, loop=0)
    if type_obs is not None:
        type_obs = np.array(type_obs)
        print(rank)
        # --- TRI SELON RANK ---
        # ordered_indices = sorted(
        #     range(len(rank)),
        #     key=lambda i: rank[i]
        # ) #on teste autre chose
        ordered_indices = sorted(
            rank.keys(),
            key=lambda i: rank[i]
        )


        type_obs = type_obs[:, ordered_indices]
        obs_names = [obs_names[i] for i in ordered_indices]

        n_bars = type_obs.shape[0]
        p_categories = type_obs.shape[1]



        x = np.arange(n_bars) * period_duration + 2024
        bottom = np.zeros(n_bars)
        top = np.zeros(n_bars)
        fig, ax = plt.subplots(figsize=(9,6))
        plt.grid(axis='y', color='grey', linestyle='--')

        for i in range(p_categories):
            top = top + type_obs[:, i]
            if i < n_market+7:
                if ordered_indices[i] in sol :
                    ax.fill_between(x,bottom,top,label=obs_names[i],
                        color = color_mix[sol[ordered_indices[i]]],linewidth = 0,alpha=0.9)
                else :
                    ax.fill_between(x,bottom,top,label=obs_names[i],
                        linewidth=0,alpha=0.9)
                bottom = top
        mpl.rcParams['hatch.color'] = 'white'
        ax.fill_between(x,bottom,top,alpha=0.9, color='grey',linewidth = 0, edgecolor='white', label = 'Others',hatch='//')
        ax.legend(framealpha=1,bbox_to_anchor=(1.05, 1),
                  loc='upper left',borderaxespad=0.)
        plt.xlim(x.min(), x.max())
        plt.ylim(0, 1.1*top.max())
        plt.xlabel('Years')
        plt.ylabel('Quantity of ' + observation)
        plt.tight_layout()
        plt.savefig('figures/integrated_scenario/'+video_name + '.pdf')
        plt.show()

    return None

def dynamic_assign_vis(df, market_seg, periods, name_periods = None, video_name = 'test_video_assign', format_v = 'gif', market ='Aircraft Type',
                       observation = 'ASK', dist_limits = (4e2, 1.9e4), capac_limits = (9e3, 4e6), frame_s = 2, color = 'pink',
                        reso = 400, smooth_param = 0.05, period_duration = 1, weight = False, vmax = None):
    #prépa data
    selec_df = df[(df['Period'].isin(periods))]
    for filename in os.listdir('figures/video_storage/'):
        if filename.endswith(".png"):
            os.remove(os.path.join('figures/video_storage/', filename))
    #Créations des images
    if name_periods is None:
        name_periods = periods
    for i in range(len(periods)):
        if i%3 == 0 :
            print(str(100*i/len(periods))+'%')
        assign_vis(selec_df[selec_df['Period'] == periods[i]],market_seg=market_seg,name_fig ='video_'+str(100+i),title_fig= name_periods[i],
                   market = market, observation = observation, dist_limits = dist_limits, capac_limits = capac_limits,
                   video = True, smooth_param = smooth_param, period_duration = period_duration, weight = weight, reso = reso, vmax=vmax)

        # Animation des images
    image_files = sorted(glob.glob("figures/video_storage/*.png"))
    if format_v == 'video':
        with imageio.get_writer('figures/' + video_name + '.mp4', fps=frame_s) as writer:
            # répéter la première image 3 fois
            for _ in range(6):
                writer.append_data(imageio.imread(image_files[0]))

            # toutes les images
            for img in image_files:
                writer.append_data(imageio.imread(img))

            # répéter la dernière image 3 fois
            for _ in range(6):
                writer.append_data(imageio.imread(image_files[-1]))

    else:
        # Charger les images
        frames = [Image.open(img) for img in image_files]
        # Sauvegarder en GIF animé
        frames[0].save('figures/' + video_name + '.gif', save_all=True,
                       append_images=3*[frames[0]] + frames[:] + 3*[frames[-1]],
                       duration=1000 / frame_s, loop=0)
    return None