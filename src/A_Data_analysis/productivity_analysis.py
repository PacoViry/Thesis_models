import numpy as np
import importlib
import seaborn as sns
import pandas as pd
from fractions import Fraction
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from matplotlib.ticker import FixedLocator
from scipy.ndimage import gaussian_filter1d
from src.Common_tools import vis_ref
# importlib.reload(vis_ref)

main_Aircrafts = ['B738', 'A320', 'A319', 'A20N', 'A321', 'A21N', 'B38M', 'AT76', 'E190',
       'B77W', 'B789', 'A333', 'E195', 'CRJ9', 'AT75', 'DH8D', 'BCS3', 'A332',
       'B788', 'A359', 'B77L', 'DH8A', 'B763', 'B734', 'B772', 'CRJX', 'B752',
       'C56X', 'B737', 'S92', 'B739', 'B744', 'E55P', 'E75L', 'SU95', 'A388',
       'AT72', 'E145', 'E170', 'A306', 'AT45', 'E295', 'SF34', 'B748', 'A318',
       'C25A', 'A339', 'F2TH', 'B733', 'C68A', 'CL35', 'E35L', 'A35K', 'E75S',
       'E290', 'PC12', 'GLEX', 'C510', 'C525', 'DH8B', 'B78X', 'A343', 'CRJ2',
       'BE20', 'BCS1', 'CL60', 'PC24', 'C25B', 'A139', 'AT43',
        'B736','B735', 'RJ1H','B712', 'SB20','F100','A346','CRJ7','H25B',
          'B753','B762','B764','F50','JS41','ATP' , 'DH8C',
        'D328', 'A310','MD11','C680','JS32','MD82','FA7X','J328',
        'BE40','D228','CL30','C550','E120','SW4',
         'E135', 'E50P','GLF5','L410','F900','AT46','C650',
        'LJ60','B463','DHC6','B39M','B742', 'B773', 'A30B', 'F70'] #115 avions, 98% des vols
Appr_max_seats = [175, 180, 145, 194, 228, 244, 200, 78, 100,
            368, 420, 440, 146, 90, 78, 78, 160, 406,
            380, 440, 440, 78, 328, 150, 440, 100, 240,
            9, 143, 24, 167, 539, 10, 80, 103, 853, #A380
            78, 50, 70, 266, 50, 146, 36, 605, 132,
            10, 440, 19, 140, 9, 12, 37, 480, 76, #E75S
            100, 8, 18, 5, 6, 78, 440, 300, 50,
            10, 135, 30, 10, 10, 15, 48,
            132, 140, 100, 134, 58, 109, 350, 68, 9,#H25B
            290, 290, 400, 56, 30, 68, 56,
            30, 280, 380, 11, 19, 172, 19, 43, #J328
             4, 19, 16, 8, 30, 19,
               37, 7, 14, 19, 19, 48, 15, #C650
              8, 100, 20, 240, 550, 550, 345,70] #configurations denses en classe éco.
L_WB = ['B77W', 'B789', 'A333','A332','A310',
       'B788', 'A359', 'B77L', 'B763', 'B772','B744',  'A388','B742','B764',
       'A306', 'B748','A339','A35K','B78X', 'A343', 'A346','B762','B764''B742', 'B773', 'A30B' ]
L_NB = ['B738', 'A320', 'A319', 'A20N', 'A321', 'A21N', 'B38M', 'B752','B737',  'B739','A318',
        'B712','B733','B736','B735','B753','MD11','MD82','B39M','B734'] #NB classiques
L_RJ = [ 'E190','E195', 'CRJ9', 'BCS3',  'CRJX', 'C56X', 'B737', 'E55P', 'E75L','B463', 'SU95',
        'E145', 'E170', 'E295', 'SF34', 'E35L','E75S','E290','CRJ2', 'BCS1', 'RJ1H', 'F100','CRJ7','F50','E120','E135', 'E50P','F70']
L_TP =['AT76', 'AT75', 'DH8D', 'DH8A', 'AT72', 'AT45', 'C25A', 'F2TH', 'C68A', 'CL35', 'PC12', 'DH8B', 'BE20', 'PC24',
       'AT43', 'SB20', 'JS41', 'ATP', 'DH8C', 'D328','JS32', 'J328', 'BE40', 'D228', 'SW4', 'L410', 'AT46',  'DHC6']
top_airlines = ['RYR', 'DLH', 'EZY', 'AFR', 'SAS', 'BAW', 'KLM',
                'VLG', 'AZA', 'WZZ', 'BEE', 'SWR', 'EWG', 'AUA',
               'TAP', 'NAX', 'WIF', 'FIN', 'LOT', 'EXS', 'PGT',
               'IBE', 'AEA','ANE', 'BER', 'BEL', 'EIN', 'AEE', 'GWI',
               'TRA', 'SXS', 'EZS', 'LOG', 'OAL', 'BTI', 'TVF',
               'CFG', 'ROT', 'CFE', 'SHT', 'ASL', 'IBB', 'LGL',
               'SEH', 'CTN', 'STK', 'TAY', 'JAF', 'TUI', 'VIR',
               'CSA', 'CCM', 'BMS', 'RYS', 'TOM', 'TFL', 'BLX',
                'WAU']
names = ['Ryanair', 'Lufthansa', 'Easy Jet', 'Air France', 'Scandinavian Airlines', 'British Airways', 'KLM',
         'Vueling', 'Alitalia', 'Wizz Air', 'Flybe', 'Swissair', 'Eurowings', 'Austrian Airlines',
        'Air Portugal', 'Norvegian Air Shuttle', 'Wideroe', 'Finnair', 'Polish airline', 'Jet2', 'Pegasus Airlines',
        'Iberia', 'Air Europa', 'Air Nostrum', 'Fly Air 41', 'Brussel Airlines', 'Aer Lingus', 'Agean Airlines', 'German Wings',
        'Transavia', 'SunExpress', 'Easy jet switzerland', 'Loganair', 'Olympic Air', 'Air Baltic', 'Transavia France',
        'Condor', 'Tarom', 'BA City Flier', 'British Airways shuttle', 'Air Serbia', 'Binter Canarias', 'Lux Air',
        'Sky Express', 'Croatia Airlines', 'Stobart Air', 'ASL airlines belgium', 'TUI Fly belgium', 'TUI Fly', 'Virgin Atlantic',
        'Czech Airlines', 'Air Corsica', 'Blue Air', 'Buzz', 'TUI Airways', 'TUI Netherlands', 'TUI Nordic',
        'Wizz air Ukraine']

# Two types of entry dataset, individual flights, and flight streaks series.
# Individual flights contain the following columns : A REPRENDRE AVEC L ESSENTIEL
#     'ADEP', 'ADES','ADEP -1', 'ADES -1', 'ADEP +1', 'ADES +1', 'FILED OFF BLOCK TIME', 'FILED ARRIVAL TIME',
#     'ACTUAL OFF BLOCK TIME', 'ACTUAL ARRIVAL TIME', 'Aircraft Type',
#     'Aircraft Operator', 'Aircraft Registration', 'Period',
#     'Real Flight Duration', 'Real Flight Duration', 'Ground Time',
#     'Ground Time -1', 'Real Flight Duration +1',
#     'Real Flight Duration +1', 'Real Flight Duration -1',
#     'Real Flight Duration -1'
# Flights streaks series contain less columns :


#The following module allows :
# 1) to vizualize the density of ground times
# 2) to vizualize back and forth densities and associated ground times patterns
# 3) to vizualize individual flight series operationnal pattern
# 4) to vizualize flight streaks patterns

def GT_density(df, name_fig = 'test_GT_density', GT_min = 0.2,GT_max = 24, FT_min = 0.5, FT_max = 14, reso = 900, sigma = 0.02):
    sns.set(style='whitegrid')
    plt.figure(figsize=(14, 8))

    largeur_x = np.log(FT_max) - np.log(FT_min)
    largeur_y = np.log(GT_max) - np.log(GT_min)
    FT_min_eq = FT_min * np.exp(- largeur_x / 2 / (
                reso - 1))  # a adapter selon le type de colormesh, on répartit avec un facteur 2 sinon
    GT_min_eq = GT_min * np.exp(- largeur_y / 2 / (reso - 1))
    FT_max_eq = FT_max * np.exp( largeur_x / 2 / (reso - 1))
    GT_max_eq = GT_max * np.exp(  largeur_y / 2 / (reso - 1))

    angle = np.arctan((FT_max-FT_min) / (0.65 * 24)) * 180 / np.pi
    df_selec = df[(df['Planned Flight Duration +1']>FT_min_eq)&(df['Ground Time']>GT_min_eq)&(df['Ground Time']<GT_max_eq)]
    sigma_x, sigma_y = sigma, 5/2*sigma

    #density calculations
    x = np.array(df_selec['Planned Flight Duration +1'])*(1 - sigma_x ** 2 / 2)
    y = np.array(df_selec['Ground Time']) *(1- sigma_y ** 2 / 2)
    GE_matrix, x_, y_ = vis_ref.smooth_ln_2d(x, y, np.ones(x.shape[0]), [FT_min_eq, FT_max_eq], [GT_min_eq, GT_max_eq], reso, sigma_x, sigma_y)
    x_abs = np.exp(x_)
    y_abs = np.exp(y_)
    GE_matrix = GE_matrix.transpose()
    GE_d_matrix =GE_matrix/GE_matrix.sum(axis=0, keepdims=True)*reso/(y_[-1]-y_[0])
    GE_d_matrix = np.where(GE_d_matrix > 1.8, 1.8, GE_d_matrix)
    plt.plot(x_abs, ((GE_d_matrix) * y_abs[:, np.newaxis]).sum(axis=0) / (GE_d_matrix).sum(axis=0),linewidth=2, color='fuchsia')

    # additionnal details
    plt.annotate(
        'Average ground times',  # Le label à afficher
        (0.9, 3.8),  # Position (valeur réelle, prédiction)
        textcoords="offset points",  # Décalage par rapport à la position
        xytext=(0, 0),  # Décalage en pixels (x, y)
        fontsize=13 + 2,  # Taille de la police
        color='fuchsia'  # Couleur des annotations
    )
    plt.plot(np.linspace(0.5, 5.5, 25), np.linspace(22, 12, 25), linewidth=3, linestyle='--', color='lime')
    plt.annotate(
        'Daily "back and forth" 1h stop',  # Le label à afficher
        (3, 15),  # Position (valeur réelle, prédiction)
        textcoords="offset points",  # Décalage par rapport à la position
        xytext=(5, 5),  # Décalage en pixels (x, y)
        fontsize=12 + 2,  # Taille de la police
        color="lime"  # Couleur des annotations
        , rotation=-angle
    )
    plt.plot(np.linspace(0.5, 5.5, 25), np.linspace(1, 1, 25), linewidth=3, linestyle='--', color='black')
    plt.annotate(
        'TAT = 1h',  # Le label à afficher
        (1.5, 1.3),  # Position (valeur réelle, prédiction)
        textcoords="offset points",  # Décalage par rapport à la position
        xytext=(5, 5),  # Décalage en pixels (x, y)
        fontsize=12 + 2,  # Taille de la police
        color="black"  # Couleur des annotations
    )
    plt.plot(np.linspace(6, 15, 25), np.linspace(2.5, 2.5, 25), linewidth=3, linestyle='--', color='black')
    plt.annotate(
        'TAT = 2.5h',  # Le label à afficher
        (8, 0.9),  # Position (valeur réelle, prédiction)
        textcoords="offset points",  # Décalage par rapport à la position
        xytext=(5, 5),  # Décalage en pixels (x, y)
        fontsize=12 + 2,  # Taille de la police
        color="black"  # Couleur des annotations
    )
    plt.plot(np.linspace(6, 10.2, 25), np.linspace(21.5 - 2 * 6, 21.5 - 2 * 10.2, 25), linewidth=3, linestyle='--',
             color='lime')
    plt.annotate(
        'Daily "back and forth" 2.5h stop',  # Le label à afficher
        (7, 5),  # Position (valeur réelle, prédiction)
        textcoords="offset points",  # Décalage par rapport à la position
        xytext=(5, 5),  # Décalage en pixels (x, y)
        fontsize=12 + 2,  # Taille de la police
        color="lime"  # Couleur des annotations
        , rotation=-angle
    )
    plt.plot(np.linspace(0.5, 5.5, 25), np.linspace(47, 36, 25), linewidth=3, linestyle='--', color='lime')
    plt.plot(np.linspace(6, 14.2, 25), np.linspace(24 + 21.5 - 2 * 6, 24 + 21.5 - 2 * 14.2, 25), linewidth=3,
             linestyle='--', color='lime')

    plt.annotate(
        '"Night cloud"',  # Le label à afficher
        (1.3, 8.5),  # Position (valeur réelle, prédiction)
        textcoords="offset points",  # Décalage par rapport à la position
        xytext=(0, 0),  # Décalage en pixels (x, y)
        fontsize=12 + 2,  # Taille de la police
        color="black"  # Couleur des annotations
    )
    plt.annotate(
        '"Daily back and forth triangle"',  # Le label à afficher
        (5.5, 3.5),  # Position (valeur réelle, prédiction)
        textcoords="offset points",  # Décalage par rapport à la position
        xytext=(0, 0),  # Décalage en pixels (x, y)
        fontsize=12 + 2,  # Taille de la police
        color="black"  # Couleur des annotations
    )

    plt.annotate(
        '"Back and forth" every 2 days, 2.5h stop',  # Le label à afficher
        (9.5, 15),  # Position (valeur réelle, prédiction)
        textcoords="offset points",  # Décalage par rapport à la position
        xytext=(5, 5),  # Décalage en pixels (x, y)
        fontsize=12 + 2,  # Taille de la police
        color="lime"  # Couleur des annotations
        , rotation=-angle
    )

    levs = (np.array([0, 0.1, 0.2, 0.4, 0.6, 1, 1.4, 1.8]))
    plt.contourf(x_abs, y_abs, GE_d_matrix, levs, cmap='Spectral_r')
    plt.colorbar(label='Density of contribution to average ground time (adimensionnal)', spacing='proportional')
    plt.tick_params(axis='both', which='both', color='0.7', length=7, width=1, direction='out', bottom=True, left=True,
                    labelsize=13)
    plt.xlabel('Following filed block time (h)', fontsize=14)
    plt.ylabel('Ground time (h)', fontsize=14)
    plt.xlim(0, FT_max)  # xmax
    plt.ylim(0, GT_max)
    plt.savefig('figures/productivity_figures/' + name_fig + '.pdf', format='pdf')
    plt.close()
    return(None)

def TATs_matrix(df, name_fig = 'test', x_min = 0.5,x_max = 14, y_min = 0.5, y_max = 14,z_max = 24, reso = 900, sigma = 0.02):
    df_2 = df[df['ADEP -1'].notna()]
    mask1 = (df['ADEP -1'].notna()) & (df['ADES'] != df['ADEP -1']) & (df['ADEP'] != df['ADES +1'])
    mask2 = (df['ADEP -1'].notna())
    print(str(int(1000*mask1.sum()/mask2.sum()+0.5)/10)+'% are not b&f flights')

    sigma_x, sigma_y = sigma, sigma
    largeur_x = np.log(x_max) - np.log(x_min)
    largeur_y = np.log(y_max) - np.log(y_min)
    x_min_eq = x_min * np.exp(- largeur_x / 2 / (reso-1))  # a adapter selon le type de colormesh, on répartit avec un facteur 2 sinon
    y_min_eq = y_min * np.exp(- largeur_y / 2 / (reso-1))
    x_max_eq = x_max * np.exp(largeur_x / 2 / (reso-1))
    y_max_eq = y_max * np.exp(largeur_y / 2 / (reso-1))

    x = np.array(df_2['Planned Flight Duration -1']) * (1 - sigma_x ** 2 / 2)
    y = np.array(df_2['Planned Flight Duration +1']) * (1 - sigma_y ** 2 / 2)
    z = (np.array(df_2['Ground Time -1']))
    mask_z = (z<z_max)&(df_2['Planned Flight Duration -1']>0)
    x, y, z = x[mask_z], y[mask_z], z[mask_z]

    GE_matrix, x_, y_ = vis_ref.smooth_ln_2d(x, y, np.ones(x.shape[0]), [x_min_eq, x_max_eq], [y_min_eq, y_max_eq],
                                             reso, sigma_x, sigma_y)
    x_abs = np.exp(x_)
    y_abs = np.exp(y_)
    GE_matrix = GE_matrix.transpose()
    X, Y = np.meshgrid(x_abs, y_abs)
    GE_d_matrix = GE_matrix / x_abs[:, np.newaxis] / y_abs[np.newaxis,:] * reso ** 2  ## Pour normaliser à l'échelle
    d_ref = 10 ** int(np.log10(GE_d_matrix.max()+0.5))
    GE_d_matrix = np.where(GE_d_matrix > d_ref, d_ref, GE_d_matrix)
    levs = (np.array(
        [u for u in [0.01, 0.018, 0.031, 0.056, 0.1, 0.18, 0.31, 0.56, 1, 1.8, 3.1, 5.6, 10, 18, 30, 56, 100] if
         u >= 0]) / 100 * d_ref)
    GE_d_matrix_nan = np.where(GE_d_matrix < levs[0], np.nan, GE_d_matrix)

    sns.set(style='whitegrid')
    plt.figure(figsize=(9, 7))
    plt.contourf(x_abs, y_abs, GE_d_matrix_nan, levs, norm=mcolors.LogNorm(), cmap='gist_ncar')
    cbar = plt.colorbar()
    plt.contour(x_abs, y_abs, GE_d_matrix_nan, levels=levs[1:], zorder=2, linewidths=0.5,
                           colors='black', linestyles='-')

    for level in levs:
        cbar.ax.hlines(level, *cbar.ax.get_xlim(), colors='black', linewidth=0.5, linestyle='-')
    cbar.set_label(r'Stops density ($h^{-2}$)', fontsize=12)
    labels = [f"{x:.1e}" for x in levs]
    cbar.set_ticks(levs)
    cbar.set_ticklabels(labels)

    plt.contour(X, Y, GE_d_matrix, levels=[levs[0]], zorder=2, linewidths=[3], colors='black')
    plt.contour(X, Y, GE_matrix, levels=[levs[0] * y_max * x_max / (reso ** 2)], zorder=3, linewidths=[3],
                           colors='red')

    plt.tick_params(axis='both', which='both', color='0.7', length=6, width=1, direction='out')
    extr = 2 * (int(x_max) // 2)
    if x_min < 1:
        x_ticks = [0.5, 1, 1.5, 2, 3, 4] + list(np.linspace(6, extr, (extr - 6) // 2 + 1))
        y_ticks = [0.5, 1, 1.5, 2, 3, 4] + list(np.linspace(6, extr, (extr - 6) // 2 + 1))
        plt.xticks(x_ticks, ['0.5'] + [str(v) for v in x_ticks[1:]], fontsize=12)
        plt.yticks(y_ticks, ['0.5'] + [str(v) for v in y_ticks[1:]], fontsize=12)
        plt.annotate('Statistically significant \n after log-smoothing', (3, 13), fontsize=14, color="red")
    else:
        x_ticks = list(np.linspace(6, extr, (extr - 6) // 2 + 1))
        y_ticks = list(np.linspace(6, extr, (extr - 6) // 2 + 1))
        plt.xticks(x_ticks, [str(u) for u in x_ticks], fontsize=12)
        plt.yticks(y_ticks, [str(u) for u in y_ticks], fontsize=12)

    plt.xlabel('Average duration of back and forth serie N (h)', fontsize=14)
    plt.ylabel('Average duration of back and forth serie N+1 (h)', fontsize=14)
    ax = plt.gca()
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    plt.savefig('figures/productivity_figures/' + f'transition_matrix_{100 * sigma}_' + name_fig + '.pdf', format='pdf')
    plt.close()

    sns.set(style='whitegrid')
    plt.figure(figsize=(9, 7))
    GE_matrix_w, x_, y_ = vis_ref.smooth_ln_2d(x, y, z, [x_min_eq, x_max_eq], [y_min_eq, y_max_eq],
                                             reso, sigma_x, sigma_y)
    AVG_GT = GE_matrix_w.transpose()/GE_matrix
    AVG_GT_0 = np.where(GE_matrix<levs[0] * y_max * x_max / (reso ** 2), 4.5, AVG_GT)
    AVG_GT_nan = np.where(GE_matrix<levs[0] * y_max * x_max / (reso ** 2), np.nan, AVG_GT)

    v_max = min(14, AVG_GT_0.max())
    v_min = max(1.5, AVG_GT_0.min())
    if v_max > 9:
        levels_contours = list(np.arange(2, 3, 0.5)) + list(np.arange(int(v_min) + 2, 6, 1)) + list(
            range(6, int(v_max) + 1, 1))
    else:
        levels_contours = list(np.arange(2, 3, 0.5)) + list(np.arange(int(v_min) + 2, 6, 1)) + list(
            np.arange(5, int(2 * v_max) / 2 + 1, 1))
    level_linestyles = ['-', '-'] * (len(levels_contours) // 2) + ['-'] * (len(levels_contours) % 2)
    AVG_GT_nan = np.where(AVG_GT_nan < int(v_min) + 1, int(v_min) + 1, AVG_GT_nan)
    AVG_GT_nan = np.where(AVG_GT_nan > int(v_max), int(v_max), AVG_GT_nan)
    plt.contourf(x_abs, y_abs, AVG_GT_nan, levels_contours, norm=mcolors.PowerNorm(gamma=1),cmap='Spectral_r')
    cbar = plt.colorbar()

    plt.contour(X, Y, GE_matrix, levels=[levs[0] * y_max * x_max / (reso ** 2)], zorder=3, linewidths=[3],
                            colors='red')
    plt.contour(X, Y, AVG_GT_nan, levels=levels_contours[1:], zorder=2, linewidths=0.5, colors='black',
                           linestyles=level_linestyles[1:])
    for level, linestyle in zip(levels_contours, level_linestyles):
        cbar.ax.hlines(level, *cbar.ax.get_xlim(), colors='black', linewidth=0.5, linestyle=linestyle)
    cbar.set_label('Average GTs post-flight in back and forth serie N', fontsize=12)
    cbar.update_ticks()

    plt.tick_params(axis='both', which='both', color='0.7', length=6, width=1, direction='out')
    if x_min < 1:
        x_ticks = [0.5, 1, 2] + list(range(4, 2 * (int(x_max) // 2 + 1), 2))
        y_ticks = [0.5, 1, 2] + list(range(4, 2 * (int(x_max) // 2 + 1), 2))
        plt.xticks(x_ticks, [str(u) for u in x_ticks], fontsize=12)
        plt.yticks(y_ticks, [str(u) for u in y_ticks], fontsize=12)
        plt.annotate('Statistically significant \n after log-smoothing', (3, 13), fontsize=14, color="red")
    else:
        x_ticks = list(range(2 * (int(x_min) // 2 + 1), 2 * (int(x_max) // 2 + 1), 2))
        y_ticks = list(range(2 * (int(x_min) // 2 + 1), 2 * (int(x_max) // 2 + 1), 2))
        plt.xticks(x_ticks, [str(u) for u in x_ticks], fontsize=12)
        plt.yticks(y_ticks, [str(u) for u in y_ticks], fontsize=12)

    plt.xlabel('Average duration of back and forth serie N (h)', fontsize=14)
    plt.ylabel('Average duration of back and forth serie N+1 (h)', fontsize=14)
    ax = plt.gca()
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    plt.savefig('figures/productivity_figures/' +f'TATs_MATRIX_{100 * sigma}_' + name_fig + '.pdf', format='pdf')
    plt.close()
    return(None)

def visu(exemple, titre = 'ac_calendar', visu = True, x_min = None):
    exemple["ACTUAL OFF BLOCK TIME"] = pd.to_datetime(exemple["ACTUAL OFF BLOCK TIME"], utc=True).dt.tz_convert(
        'Europe/Paris')
    exemple["ACTUAL ARRIVAL TIME"] = pd.to_datetime(exemple["ACTUAL ARRIVAL TIME"], utc=True).dt.tz_convert(
        'Europe/Paris')

    # Début et fin du mois en France
    first_date = exemple["ACTUAL OFF BLOCK TIME"].min()
    start_time = pd.Timestamp(first_date.year, first_date.month, 1, tz='Europe/Paris')

    # Fin du mois = 1er jour du mois suivant - 1 nanoseconde
    end_time = (start_time + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)

    # Création de la figure
    fig, ax = plt.subplots(figsize=(15, 3))
    # Dégradé de 23h à 11h (heure locale France)
    gradient_steps = 200

    colors_l = [mcolors.to_rgba(c) for c in ["black", "#FFD700"]]
    gradient = np.linspace(0, 1, gradient_steps).reshape(1, -1)
    cmap = mcolors.LinearSegmentedColormap.from_list("black_yellow", colors_l)

    # Ajout du fond pour chaque jour
    current_time = start_time
    L_date = ['M', 'T', 'W', 'T', 'F', 'S', 'S']  # décalé d'un pour fitter
    if x_min is not None:
        lim_time = start_time + pd.Timedelta(days=0, hours=(x_min + 1 + 2 / 3) % 24)
    else :
        lim_time = None
    while current_time <= end_time + pd.Timedelta(days=1):
        # Plage de nuit : de 23h aujourd’hui à 11h le lendemain
        night_start = current_time + pd.Timedelta(days=-1, hours=15)
        night_end = current_time + pd.Timedelta(days=0, hours=3)
        evening = current_time + pd.Timedelta(days=0, hours=15)

        x_start = mdates.date2num(night_start)
        x_end = mdates.date2num(night_end)
        x_end_2 = mdates.date2num(evening)

        ax.imshow(1 - gradient, aspect="auto", extent=(x_start, x_end, 0, 1), cmap=cmap)
        ax.imshow(gradient, aspect="auto", extent=(x_end, x_end_2, 0, 1), cmap=cmap)
        # Marquage du jour en niveau de gris (ex : pour montrer progression des jours)
        gray = str(1 - current_time.weekday() / 6)
        ax.barh(y=0.05, left=mdates.date2num(current_time) + 2 / 24, width=1, height=0.1, color=gray, linewidth=0)
        if current_time <= end_time:
            d = int(current_time.weekday())
            plt.text(x=mdates.date2num(current_time) + 9.4 / 24, y=0.02, s=L_date[d], color='orange', fontsize=14)
        if x_min is not None:
            ax.barh(y=0.55, left=mdates.date2num(lim_time), width=4 / 3 / 24, height=0.85, color='red', linewidth=0)
            lim_time = lim_time + pd.Timedelta(days=1)
        current_time += pd.Timedelta(days=1)

    # Tracer les vols
    for i, row in exemple.iterrows():
        start = row['ACTUAL OFF BLOCK TIME']
        end = row['ACTUAL ARRIVAL TIME']
        ax.barh(y=0.55, left=start, width=(end - start), height=0.7, color="blue", alpha=1, linewidth=0)

    # Configuration de l'axe temporel
    ax.set_xlim(start_time, end_time)  # Pour inclure le dernier jour
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=MO, tz="Europe/Paris"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d", tz="Europe/Paris"))
    plt.xticks(fontsize=15, ha='left')

    # Suppression axe y
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)

    # Label
    ax.set_xlabel("Weeks", fontsize=18)
    if visu:
        plt.savefig('figures/productivity_figures/' + titre + '.pdf', format='pdf')
        plt.close()
    else :
        plt.show()
    return(None)

def heures_en_vol(dep, arr, sigma_h, res_minutes):
    """Renvoie un tableau 24h comptant les minutes où l'avion est en vol."""
    minutes = np.arange(0, 24 * 60, res_minutes)
    counts = np.zeros_like(minutes, dtype=float)

    dep_min = (dep.dt.hour.to_numpy() * 60 + dep.dt.minute.to_numpy() - 30) % (
                24 * 60)  ### 20mn de moins sur l'heure de départ
    arr_min = (arr.dt.hour.to_numpy() * 60 + arr.dt.minute.to_numpy() + 30) % (
                24 * 60)  ### 20mn de plus sur l'heure d'arrivée
    cross_midnight = (dep_min > arr_min)

    # pour les vols ne traversant pas minuit
    mask1 = ~cross_midnight
    for t1, t2 in zip(dep_min[mask1], arr_min[mask1]):
        counts[t1:t2] += 1.0

    # pour les vols traversant minuit
    mask2 = cross_midnight
    for t1, t2 in zip(dep_min[mask2], arr_min[mask2]):
        counts[t1:] += 1.0
        counts[:t2] += 1.0

    sigma_min = sigma_h * 60 / res_minutes
    counts_smooth = gaussian_filter1d(counts, sigma=sigma_min, mode='wrap')
    heures = minutes / 60
    return pd.Series(counts_smooth, index=heures)

def visu_distrib(exemple, sigma = 0.8, titre = 'ac_fh_distrib', visu = False, visu_c = False):
    distribution = heures_en_vol(exemple['FILED OFF BLOCK TIME'], exemple['FILED ARRIVAL TIME'], sigma, res_minutes=1)
    x_min = distribution.argmin()
    y_min = distribution.min()
    if visu:
        sns.set(style='whitegrid')
        plt.figure(figsize=(9, 2.5))  # Adjusted figure size
        if visu_c:
            distribution2 = heures_en_vol(exemple['FILED OFF BLOCK TIME'], exemple['FILED ARRIVAL TIME'], 0.0005, res_minutes=1)
            plt.plot(distribution2.index, distribution2.values, label='raw data')
            plt.plot(distribution.index, distribution.values, label='smoothed data')
            plt.scatter(x_min*1/60, y_min, color='red', label='reference hour', zorder=3)
        else:
            plt.plot(distribution.index, distribution.values)
        plt.legend(loc='best', framealpha = 1)  # Adjusted legend position
        plt.xlabel("Hour of the day")
        plt.ylabel("Flights occuring")
        plt.grid(True)
        plt.xlim([distribution.index.min(), distribution.index.max()])  # Set x-axis limits
        plt.ylim([0, distribution.values.max() * 1.1])  # Set y-axis limits
        if visu_c:
            plt.savefig('figures/productivity_figures/' + titre + '.pdf', format='pdf', bbox_inches='tight')
            plt.close()
        else :
            plt.show()
    return x_min*1/60, y_min

def constru_series_vols(exemple, x_min):
    exemple = exemple.sort_values('FILED OFF BLOCK TIME').reset_index(drop=True)

    # Bornes temporelles
    start = exemple['FILED ARRIVAL TIME'].min().normalize()
    end = exemple['FILED OFF BLOCK TIME'].max().normalize() + pd.Timedelta(days=1)

    # Temps de référence journaliers
    ref_times = pd.date_range(start=start, end=end, freq='D') + pd.to_timedelta(x_min, unit='h')
    ref_times_np = ref_times.to_numpy()

    # Périodes d'inactivité (avec la marge nécessaire)
    end_np = exemple['FILED OFF BLOCK TIME'][1:].to_numpy() + np.timedelta64(0,
                                                                             'm')  ### 0mn autour de l'emplacement de l'inactivité
    start_np = exemple['FILED ARRIVAL TIME'][:-1].to_numpy() - np.timedelta64(0,
                                                                              'm')  ### 0mn autour de l'emplacement de l'inactivité


    cond = (ref_times_np[:, None] >= start_np[None, :]) & (ref_times_np[:, None] < end_np[None, :])
    ref_inactive = ref_times_np[np.any(cond, axis=1)]
    if len(ref_inactive) < 2:
        return np.empty((0, 4))

    # Vols en numpy
    arr_np = exemple['FILED ARRIVAL TIME'].to_numpy()
    dep_np = exemple['FILED OFF BLOCK TIME'].to_numpy()
    durations = (arr_np - dep_np) / np.timedelta64(1, 'h')

    # Intervalles [t1, t2)
    t1s = ref_inactive[:-1]
    t2s = ref_inactive[1:]
    n_intervals = len(t1s)

    # Identifier pour chaque vol l’intervalle où il tombe
    idx_start = np.searchsorted(t1s, arr_np, side='right') - 1
    idx_valid = (idx_start >= 0) & (dep_np < t2s[idx_start])
    idx_start = idx_start[idx_valid]
    durations_valid = durations[idx_valid]

    if len(idx_start) == 0:
        return np.empty((0, 4))

    # Comptage par intervalle
    n_vols = np.bincount(idx_start, minlength=n_intervals)
    sum_durations = np.bincount(idx_start, weights=durations_valid, minlength=n_intervals)
    mean_durations = np.divide(sum_durations, n_vols, out=np.zeros_like(sum_durations), where=n_vols > 0)

    # Durée (en jours) de chaque intervalle
    n_days = (t2s - t1s) / np.timedelta64(1, 'D')

    # Identifier les séries d’intervalles consécutifs non vides
    active = n_vols > 0
    series_id = np.cumsum((~active).astype(int))
    # Calcul de la durée totale de chaque série (somme des n_days)

    # Attribution aux vols
    taille_groupe = n_vols[idx_start]
    series_duration_days = n_days[idx_start]

    duree_moyenne_groupe = mean_durations[idx_start]
    # Départs et arrivées des vols retenus
    dep_valid = dep_np[idx_valid]
    arr_valid = arr_np[idx_valid]

    # Détection du recoupement de l'heure h_ref
    pos = np.searchsorted(ref_times_np, dep_valid, side='left')
    cuts_ref = np.zeros(len(dep_valid), dtype=int)
    mask = (pos < ref_times_np.size) & (ref_times_np[pos] < arr_valid)
    cuts_ref[mask] = 1

    tableau_vols = np.column_stack((
        durations_valid,
        taille_groupe,
        duree_moyenne_groupe,
        series_duration_days,
        cuts_ref
    ))
    return tableau_vols

def constru_flight_streak(df):
    grouped_aircrafts = {u: ac for u, ac in df.groupby('Aircraft Registration')}
    L_id = df['Aircraft Registration'].unique()
    n_tot = len(L_id)
    frac = n_tot//20
    series_tot = []

    for i in range(n_tot):
        if i % frac == 0: print(str(i//frac*5)+'%,', end = " ")
        aircraft_ref = grouped_aircrafts.get(L_id[i])
        for t, aircraft_ex in aircraft_ref.groupby('Period'):
            if aircraft_ex.shape[0] > 10:
                x_min, _ = visu_distrib(aircraft_ex, sigma = 0.8, visu = False,visu_c = False)
                series2 = constru_series_vols(aircraft_ex, x_min)
                if series2.size > 0:
                    series_tot.append(series2)
    series_tot = np.concatenate(series_tot, axis=0)
    print('100 %, ok.')
    return series_tot

def visu_flight_streaks(array, min_fh = 0.5, max_fh = 14, reso = 200, sigma = 0.02, title = 'regimes_all', n_s = None):
    plt.style.use('default')
    cmap = cm.get_cmap('gist_ncar')
    n_d = 10
    n_v = 10
    n_c = 10
    hatch_l = ['..', '///', None]
    edge_colors = ['0', 'black', 'black']
    F = [str(0)]
    order = [0.0]
    types = [None]
    largeur_x =np.log(max_fh)-np.log(min_fh)
    min_fh_eq = min_fh * np.exp(- largeur_x / 2 / (reso - 1))  # a adapter selon le type de colormesh, on répartit avec un facteur 2 sinon
    max_fh_eq = max_fh * np.exp(largeur_x / 2 / (reso - 1))

    y_l_i = [[np.zeros(reso)]]
    y_l_s = [[np.zeros(reso)]]

    test, x_grid = vis_ref.smooth_ln_1d(np.array((min_fh_eq*max_fh_eq)**0.5), np.array(1), [min_fh_eq, max_fh_eq],reso, smooth_param_x =sigma)
    for j in range(1, n_d + 1):
        series_selec = array[array[:, 3] == j]
        print(j, end = ", ")
        for i in range(1, n_v * j + 1):
            series_r = series_selec[series_selec[:, 1] == i]
            if series_r.shape[0] > 0:
                L_i = np.zeros(reso)
                L_s = np.zeros(reso)
                for k in range(n_c):
                    indices = np.arange(k, series_r.shape[0], n_c)
                    if indices.shape[0] > 0:
                        L_i += vis_ref.smooth_ln_1d(series_r[indices, 0] * (1 - (sigma ** 2) / 2), np.ones(indices.shape[0]), [min_fh_eq, max_fh_eq],reso, smooth_param_x =sigma)[0] # 0 pour la version individuelle, 2 série
                        L_s += vis_ref.smooth_ln_1d(series_r[indices, 2] * (1 - (sigma ** 2) / 2), np.ones(indices.shape[0]), [min_fh_eq, max_fh_eq],reso, smooth_param_x =sigma)[0]
                y_l_i.append([L_i]) # 0 pour la version individuelle, 2 série
                y_l_s.append([L_s])  # 0 pour la version individuelle, 2 série
                frac = Fraction(i, j)
                num = frac.numerator # 2
                denom = frac.denominator # 3
                order.append(i / j)
                if denom == 1:
                    F.append(str(num))
                    if num % 2 == 0:
                        types.append(2)
                    else:
                        types.append(0)
                else:
                    F.append(str(num) + '/' + str(denom))
                    types.append(1)

    indices = np.argsort(order)
    order_sorted = [order[i] for i in indices]
    y_l_i_sorted = [y_l_i[i] for i in indices]
    y_l_s_sorted = [y_l_s[i] for i in indices]

    F_sorted = [F[i] for i in indices]
    type_sorted = [types[i] for i in indices]
    # Fusionner les valeurs identiques
    order_unique = []
    y_l_i_merged = []
    y_l_s_merged = []

    F_merged = []
    type_merged = []
    for o, y_i, y_s, f, h in zip(order_sorted, y_l_i_sorted, y_l_s_sorted, F_sorted, type_sorted):
        if order_unique and np.abs(o - order_unique[-1]) < 0.1:
            y_l_i_merged[-1] += y_i[0]
            y_l_s_merged[-1] += y_s[0]
        else:
            order_unique.append(o)
            y_l_i_merged.append(y_i[0])
            y_l_s_merged.append(y_s[0])
            F_merged.append(f)
            type_merged.append(h)
    colors_sorted = [((order_unique[i] - 0.5) / (n_v)) ** 0.8 for i in range(len(order_unique))]

    y_l_i_merged = np.array(y_l_i_merged)
    y_l_i_merged = np.cumsum(y_l_i_merged, axis=0)
    y_l_i_merged_f = 100 * y_l_i_merged / y_l_i_merged[-1, :]
    y_l_s_merged = np.array(y_l_s_merged)
    y_l_s_merged = np.cumsum(y_l_s_merged, axis=0)
    y_l_s_merged_f = 100 * y_l_s_merged / y_l_s_merged[-1, :]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, height_ratios=[3, 3, 3], figsize=(7, 8))

    for i in range(len(order_unique) - 1):
        ax2.fill_between(np.exp(x_grid), y_l_i_merged_f[i], y_l_i_merged_f[i + 1], label=F_merged[i + 1], linewidth=0.2,
                         color=cmap(colors_sorted[i + 1]), edgecolors=edge_colors[type_merged[i + 1]],
                         hatch=hatch_l[type_merged[i + 1]])
    ax2.set_ylim((0, 100))
    ax2.set_ylabel("Flights per day \n" r"depending on $BT$ (%)")

    for i in range(len(order_unique) - 1):
        ax3.fill_between(np.exp(x_grid), y_l_s_merged_f[i], y_l_s_merged_f[i + 1], linewidth=0.2,
                         color=cmap(colors_sorted[i + 1]), edgecolors=edge_colors[type_merged[i + 1]],
                         hatch=hatch_l[type_merged[i + 1]])
    ax3.set_ylim((0, 100))
    # ax3.text(1.65,50,'6', color = 'white', fontsize = 15)
    ax3.text(3.2, 50, '4', color='white', fontsize=15)
    ax3.text(6, 50, '2', color='white', fontsize=15)
    # ax3.text(13,25,'1', color = 'white', fontsize = 15)

    ax3.set_ylabel("Flights per day \n" r"depending on $BT_{avg}$ (%)")

    x_ticks = np.arange(1, 15, 1)
    ax1.plot(np.exp(x_grid), y_l_i_merged[-1] / np.exp(x_grid)*reso/largeur_x, color='0.5', label='Block time')
    ax1.plot(np.exp(x_grid), y_l_s_merged[-1] / np.exp(x_grid)*reso/largeur_x, color='0.1', linestyle='--',
             label='Series avg block time')
    ax1.legend(framealpha=1)
    ind_max = int(np.log10(max(max(y_l_i_merged[-1] / np.exp(x_grid)*reso/largeur_x),max(y_l_s_merged[-1] / np.exp(x_grid)*reso/largeur_x))))+1
    # max_y = 10 ** ind_max
    if n_s is None :
        min_y = 10 ** (ind_max-5)
    else : min_y = 10**n_s
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=min_y)
    ax1.set_ylabel('Flights densities ' r'($h^{-1}$)')
    ax1.grid(True, axis='both', linestyle='--', linewidth=0.3, color='gray')

    plt.xticks(x_ticks, x_ticks)
    plt.xlim((min_fh, max_fh))

    plt.xlabel(r'Filed block time $BT$ or $BT_{avg}$ (h)')
    plt.savefig('figures/productivity_figures/' + title + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print('ok.')
    return(None)

def streaks_matrix(array, min_fh = 0.5, max_fh = 14, sigma = 0.02, reso = 1000, title = 'streak_matrix'):
    largeur_x = np.log(max_fh) - np.log(min_fh)
    min_fh_eq = min_fh * np.exp( - largeur_x / 2 / (reso - 1))  # a adapter selon le type de colormesh, on répartit avec un facteur 2 sinon
    max_fh_eq = max_fh * np.exp( largeur_x / 2 / (reso - 1))
    sigma_x, sigma_y = sigma, sigma
    x = array[:,0]* (1 - sigma_x ** 2 / 2)
    y = array[:,2] * (1 - sigma_y ** 2 / 2)
    # Histogramme 2D sur grille régulière
    grouping_matrix, x_, y_ = vis_ref.smooth_ln_2d(x, y, np.ones(x.shape[0]), [min_fh_eq, max_fh_eq],
                                        [min_fh_eq, max_fh_eq], reso, sigma_x, sigma_y)
    grouping_matrix = grouping_matrix.transpose()
    x_abs = np.exp(x_)
    y_abs = np.exp(y_)
    X, Y = np.meshgrid(x_abs, y_abs)

    grouping_flights = grouping_matrix.sum(axis=0) / x_abs * reso/largeur_x
    grouping_series = grouping_matrix.sum(axis=1) / y_abs * reso/largeur_x
    grouping_matrix_d = grouping_matrix / x_abs[:, np.newaxis] / y_abs[np.newaxis,:] * reso ** 2/largeur_x**2  ## Pour normaliser à l'échelle

    z_ref = 10 ** (int(np.log10(grouping_matrix_d.max()) + 0.5))
    grouping_matrix_d = np.where(grouping_matrix_d > z_ref, z_ref, grouping_matrix_d)
    # Z_small_nan = np.where(Z_small < 0, np.nan, Z_small)
    levs = (np.array(
        [u for u in [0.0056, 0.01, 0.018, 0.03, 0.056, 0.1, 0.18, 0.3, 0.56, 1, 1.8, 3, 5.6, 10, 18, 30, 56, 100] if
         u >= 0]) / 100 * z_ref)

    sns.set(style='whitegrid')
    fig = plt.figure(figsize=(10, 8))
    grid = plt.GridSpec(4, 18, hspace=0.15, wspace=0.5)  # 5 colonnes pour la colorbar à droite
    main_ax = fig.add_subplot(grid[1:4, 0:13])
    main_ax.grid(True, linestyle='--', linewidth=0.5, color='0.2')
    im = main_ax.contourf(x_abs, y_abs, grouping_matrix_d, levs, norm=mcolors.LogNorm(),cmap='gist_ncar')
    cax = fig.add_subplot(grid[1:4, 17])
    cbar = plt.colorbar(im, cax)
    main_ax.contour(x_abs, y_abs, grouping_matrix_d, levels=levs[1:], zorder=2,linewidths=0.5, colors='black', linestyles='-')
    puissance_ref = int(np.log10(max(grouping_flights.max(), grouping_flights.max()))) + 1

    x_density_ax = fig.add_subplot(grid[0, 0:13])
    x_density_ax.plot(x_abs, grouping_flights, color='mediumblue', linewidth=3)
    x_density_ax.plot(x_abs, grouping_series, color='firebrick', linewidth=1, linestyle='--')
    x_density_ax.set_ylabel('Marg. distrib.' + r' ($h^{-1}$)', fontsize=13)
    x_density_ax.set_yscale('log')
    x_density_ax.grid(True, linestyle='--', linewidth=0.6, color='grey', axis='x')
    x_density_ax.set_xlim((min_fh, max_fh))
    x_density_ax.set_ylim((10 ** (puissance_ref-4), 10 ** puissance_ref))

    y_density_ax = fig.add_subplot(grid[1:4, 13:17])
    y_density_ax.plot(grouping_series, y_abs, color='firebrick', linewidth=3)
    y_density_ax.plot(grouping_flights, y_abs, color='mediumblue', linewidth=1, linestyle='--')
    y_density_ax.set_xlabel('Marg. distrib.' + r' ($h^{-1}$)', fontsize=13)
    y_density_ax.set_xscale('log')
    y_density_ax.set_xlim((10 ** (puissance_ref-4), 10 ** puissance_ref))
    y_density_ax.grid(True, linestyle='--', linewidth=0.6, color='grey', axis='y')
    y_density_ax.set_ylim((min_fh, max_fh))
    for u in range(1,4):
        y_density_ax.plot([10 ** (puissance_ref - u), 10 ** (puissance_ref - u)], [min_fh, max_fh], linestyle='--',
                          color='grey', linewidth=0.6)
        x_density_ax.plot([min_fh, max_fh],[10 ** (puissance_ref - u), 10 ** (puissance_ref - u)], linestyle='--',
                          color='grey', linewidth=0.6)

    for level in levs:
        cbar.ax.hlines(level, *cbar.ax.get_xlim(), colors='black', linewidth=0.5, linestyle='-')
    cbar.set_label('Flights density' + r' ($h^{-2}$)', fontsize=13)
    labels = [f"{x:.1e}" for x in levs]
    cbar.set_ticks(levs)
    cbar.set_ticklabels(labels)
    # plt.xscale('log')
    # plt.yscale('log')
    main_ax.tick_params(axis='both', which='both', color='0.7', length=6, width=1, direction='out')
    extr = 2 * (int(max_fh) // 2)

    x_ticks = [1 / 2, 1, 2, 3, 4, 5] + list(np.arange(6,extr+2,2))
    y_ticks = [1 / 2, 1, 2, 3, 4, 5] + list(np.arange(6,extr+2,2))
    main_ax.set_xticks(x_ticks, ['30mn'] + [str(v) for v in x_ticks[1:]], fontsize=12)
    main_ax.set_yticks(y_ticks, ['30mn'] + [str(v) for v in y_ticks[1:]], fontsize=12)
    y_density_ax.set_yticks(y_ticks, [''] * len(y_ticks))
    x_density_ax.set_xticks(x_ticks, [''] * len(x_ticks))
    x_density_ax.set_yticks([10 ** u for u in range(puissance_ref-4, puissance_ref + 1)],
                            [fr'$10^{exposant}$' for exposant in range(puissance_ref-4, puissance_ref + 1)],
                             fontsize = 11)
    y_density_ax.set_xticks([10 ** u for u in range(puissance_ref-4, puissance_ref + 1)],
                            [fr'$10^{exposant}$' for exposant in range(puissance_ref-4, puissance_ref + 1)],
                            fontsize = 11)

    main_ax.plot([min_fh, max_fh], [min_fh, max_fh], color='black', linestyle='--')
    main_ax.set_xlabel('Flight filed block time (h)', fontsize=16, color='mediumblue')
    main_ax.set_ylabel('Series avg filed block time (h)', fontsize=16, color='firebrick')
    main_ax.set_xlim((min_fh, max_fh))
    main_ax.set_ylim((min_fh, max_fh))
    plt.savefig('figures/productivity_figures//' + title + '.pdf')
    plt.close()
    return(None)

