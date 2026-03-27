#module Fleet_visualisation
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from src.Common_tools.vis_ref import colors_5, colors_10,colors_22, colors_26, vis_colors
import numpy as np
import matplotlib.ticker as mtick


warnings.filterwarnings("ignore")

Class_engines = ['AVON 527',0, 'AVON 531B',0, 'AVON 533R',0, 'BR715A1-30',2, 'BR715C1-30',2, 'CF6-45A2',0, ###mapping des types de moteurs dans la bdd
                 'CF6-50C',0, 'CF6-50C1',0, 'CF6-50C2',0, 'CF6-50C2B',0, 'CF6-50C2F',0, 'CF6-50C2R',0,
                 'CF6-50E',0, 'CF6-50E2',0, 'CF6-6D',0, 'CF6-6D1A',0, 'CF6-80A',1, 'CF6-80A2',1, 'CF6-80A3',1,
                 'CF6-80C2',1, 'CF6-80C2A1',1, 'CF6-80C2A2',1, 'CF6-80C2A3',1, 'CF6-80C2A5',1,
                 'CF6-80C2A5F',1, 'CF6-80C2A8',1, 'CF6-80C2B1', 1,'CF6-80C2B1F',1, 'CF6-80C2B2',1,
                 'CF6-80C2B2F',1, 'CF6-80C2B4', 1,'CF6-80C2B4F',1, 'CF6-80C2B5F',1, 'CF6-80C2B6',1,
                 'CF6-80C2B6F',1, 'CF6-80C2B7F',1, 'CF6-80C2B8F',1, 'CF6-80C2D1F',1, 'CF6-80E1',1,
                 'CF6-80E1A2',1, 'CF6-80E1A3',1, 'CF6-80E1A4',1, 'CF6-80E1A4B',1, 'CFM56-2',0 ,
                 'CFM56-2A',0, 'CFM56-2C1',0, 'CFM56-3B1',0, 'CFM56-3B2',0, 'CFM56-3C1',0, 'CFM56-5A1',0,
                 'CFM56-5A3',0, 'CFM56-5A4',0, 'CFM56-5A5',0, 'CFM56-5B', 2,'CFM56-5B1',2, 'CFM56-5B1/2',2,
                 'CFM56-5B1/2P',2, 'CFM56-5B1/3',2, 'CFM56-5B1/P',2, 'CFM56-5B2',2, 'CFM56-5B2/3',2,
                 'CFM56-5B2/P',2, 'CFM56-5B3',2, 'CFM56-5B3/2P',2, 'CFM56-5B3/3', 2,'CFM56-5B3/3B1',2,
                 'CFM56-5B3/P',2, 'CFM56-5B4',2, 'CFM56-5B4/2',2, 'CFM56-5B4/2P',2, 'CFM56-5B4/3',2,
                 'CFM56-5B4/P',2, 'CFM56-5B5/3',2, 'CFM56-5B5/P',2, 'CFM56-5B6/2',2, 'CFM56-5B6/2P',2,
                 'CFM56-5B6/3',2, 'CFM56-5B6/P',2, 'CFM56-5B7',2, 'CFM56-5B7/3',2, 'CFM56-5B7/P',2,
                 'CFM56-5B8/3',2, 'CFM56-5B8/P',2, 'CFM56-5B9/3', 2,'CFM56-5B9/P',2, 'CFM56-5C2',1,
                 'CFM56-5C2F',1, 'CFM56-5C3/F',1, 'CFM56-5C3/G',1, 'CFM56-5C4',1, 'CFM56-5C4/P',1,
                 'CFM56-7B',2, 'CFM56-7B20',2, 'CFM56-7B20/3',2, 'CFM56-7B22',2, 'CFM56-7B22/3', 2,
                 'CFM56-7B22E',2, 'CFM56-7B24', 2,'CFM56-7B24/3',2, 'CFM56-7B24E',2, 'CFM56-7B26',2,
                 'CFM56-7B26/2',2, 'CFM56-7B26/3',2, 'CFM56-7B26/B1',2, 'CFM56-7B26E',2, 'CFM56-7B27',2,
                 'CFM56-7B27/3',2, 'CFM56-7B27/B1', 2,'CFM56-7B27/B3',2, 'CFM56-7B27E', 2,
                 'CFM56-7B27E/3B3',2, 'Conway 508', 0,'Conway 509',0, 'D-30', 0,'D-30-II',0,
                 'D-30-III', 0,'D-30KP',0, 'D-30KU',0, 'D-36',0, 'D-436',2 ,'D-436-148',2, 'GE90-110B1',2,
                 'GE90-110B1L',2, 'GE90-115B',2, 'GE90-115BL2',2, 'GE90-76B',2, 'GE90-85B',2, 'GE90-90B',2,
                 'GE90-94B',2, 'GEnx-1B64',3, 'GEnx-1B67',3, 'GEnx-1B70',3, 'GEnx-1B74',3, 'GEnx2B67',3,
                 'GEnx2B67B',3, 'GEnx2B67BG01',3, 'GEnx2B67G01',3, 'GEnx2B67G02', 3,'GP7270',2, 'GP7270E',2,
                 'JT3C', 0,'JT3C-7',0, 'JT3D',0, 'JT3D-1',0, 'JT3D-3B',0, 'JT3D-3BH', 0,'JT3D-3C',0,
                 'JT3D-3D',0, 'JT3D-4A',0, 'JT3D-7',0, 'JT3D-7H',0, 'JT3DH',0, 'JT4A',0, 'JT4D',0,
                 'JT8D',0, 'JT8D-11',0, 'JT8D-15',0, 'JT8D-15A',0, 'JT8D-15AH',0, 'JT8D-15H',0, 'JT8D-17',0,
                 'JT8D-17A',0, 'JT8D-17H',0, 'JT8D-17R',0, 'JT8D-17RH', 0,'JT8D-217',0, 'JT8D-217A',0,
                 'JT8D-217C',0, 'JT8D-219',0, 'JT8D-219H',0, 'JT8D-7', 0,'JT8D-7A',0, 'JT8D-7B', 0,
                 'JT8D-7BH',0, 'JT8D-9',0, 'JT8D-9A',0, 'JT8D-9AH',0, 'JT8D-9H', 0,'JT9D-20',0, 'JT9D-3A',0,
                 'JT9D-59A',0, 'JT9D-7',0, 'JT9D-70',0, 'JT9D-70A',0, 'JT9D-7A', 0,'JT9D-7AH',0, 'JT9D-7AW',0,
                 'JT9D-7F',0, 'JT9D-7J',0, 'JT9D-7Q',0, 'JT9D-7Q3',0, 'JT9D-7R4D', 0,'JT9D-7R4D1',0,
                 'JT9D-7R4E',0, 'JT9D-7R4E1',0, 'JT9D-7R4E4',0, 'JT9D-7R4G2',0, 'JT9D-7R4H1',0, 'JT9D-7W',0,
                 'LEAP-1A',3, 'LEAP-1B', 3,'LEAP-X1C',3, 'NK-8-2U',0, 'NK-8-4',0, 'NK-86',0, 'Olympus 593',0,
                 'PS-90',2, 'PS-90A',2, 'PS-90A-76',2, 'PS-90A2',2, 'PW1000G',3, 'PW1100G-JM',3,
                 'PW1124G-JM',3, 'PW1127G', 3,'PW1127G-JM',3, 'PW1130G',3, 'PW1133G',3, 'PW1133G-JM',3,
                 'PW1200G', 3, 'PW1500G',3, 'PW1521G',3, 'PW1524G',3, 'PW1900G',3, 'PW1921G',3, 'PW2037',0,
                 'PW2040',0, 'PW2043',0, 'PW4052',1, 'PW4056',0, 'PW4060',1, 'PW4062',0, 'PW4062A',0,
                 'PW4074',2, 'PW4077',2, 'PW4084',2, 'PW4090',2, 'PW4098',2, 'PW4152',1, 'PW4156A',1,
                 'PW4158',0, 'PW4164',1, 'PW4168',1, 'PW4168A',1, 'PW4168A-1D',1, 'PW4170',1, 'PW4460',1,
                 'PW4462',1, 'PW6122A',2, 'PW6124',2, 'RB211-22B',0, 'RB211-524B',0, 'RB211-524C2',0,
                 'RB211-524D4',0, 'RB211-524G',0, 'RB211-524GH-T',0, 'RB211-524H',0, 'RB211-535C',1,
                 'RB211-535E4',0, 'RB211-535E4B',0, 'RB211-535E4C',0, 'Spey 1', 0,'Spey 506-14',0,
                 'Spey 506-14A',0, 'Spey 511',0, 'Spey 511-14',0, 'Spey 511-14W', 0,'Spey 512',0,
                 'Spey 512-14',0, 'Spey 512-14DW',0, 'Spey 555-15',0, 'Spey 555-15H',0, 'Spey 555-15N',0,
                 'Spey 555-15P', 0,'TAY 620-15',0, 'TAY 650-15',0, 'TAY 651-54',0 , 'Trent 1000',3,
                 'Trent 1000-A',3, 'Trent 1000-C',3, 'Trent 1000-G', 3,'Trent 1000-H1',3, 'Trent 1000-J2',3,
                 'Trent 1000-K2',3, 'Trent 1000-TEN', 3,'Trent 553-61', 2,'Trent 553A2-61',2,
                 'Trent 556-61', 2,'Trent 556A2-61',2, 'Trent 7000',3, 'Trent 768-60',1, 'Trent 772', 1,
                 'Trent 772-60',1, 'Trent 772B',1, 'Trent 772B-60', 1,'Trent 772C',1, 'Trent 772C-60',1,
                 'Trent 875',2, 'Trent 877',2, 'Trent 884', 2,'Trent 890',2, 'Trent 892',2, 'Trent 895',2,
                 'Trent 970-84',2, 'Trent 972-84',2, 'Trent XWB-79',3, 'Trent XWB-84',3, 'Trent XWB-97',3,
                 'Trent-XWB-75',3, 'V2500-A1',0, 'V2522-A5',2, 'V2524-A5',2, 'V2525-D5',2, 'V2527-A5',2, 'V2527E-A5',2,
                 'V2527M-A5',2, 'V2528-D5',2, 'V2530-A5',2, 'V2533-A5',2]

L_engines = Class_engines[::2]
Gen_engines = Class_engines[1::2]
mapping_engines = dict(zip(L_engines, Gen_engines))
### the entry dataset is composed of several columns : 'Aircraft Type','Engine', 'Delivery Date', 'Status', 'Event Date'
### Status vaut written off, stored ou active
### Each column describes an engine which is or was part of the fleet
# The following module contains :
# 1) A function to vizualize the fleet composition
# 2) Functions to vizualize the aircraft deliveries and retirements
# 3) Function to create a ranking of all retirements
# 4) Function to visualize the age heterogeneity of retirements

def visu_fleet(df, mapping = mapping_engines, title = 'fleet_ex'):
    df = df.copy()
    df['gen'] = df['Engine'].map(mapping)

    deliveries = df[['gen', 'Delivery Date']].copy()
    # deliveries['Delivery Date'] = pd.to_datetime(deliveries['Delivery Date'], errors='coerce')
    deliveries['change'] = 1 #possibility to weight by the number of available seats, if it is in the dataset
    deliveries = deliveries.rename(columns={'Delivery Date': 'date'})

    retirements = df[(df['Event Date'].notna())&(df['Status']=='Written Off')][['gen', 'Event Date']].copy()
    # retirements['Event Date'] = pd.to_datetime(retirements['Event Date'], errors='coerce')
    retirements['change'] = -1#possibility to weight by the number of available seats, if it is in the dataset
    retirements = retirements.rename(columns={'Event Date': 'date'})

    events = pd.concat([deliveries, retirements])

    # 4. Calcul de l'historique par génération
    # On calcule d'abord la somme quotidienne des changements
    daily_changes = events.groupby(['date', 'gen'])['change'].sum().reset_index()
    # On pivote pour avoir les dates en index et les générations en colonnes
    history_pivoted = daily_changes.pivot(index='date', columns='gen', values='change').fillna(0)
    # Somme cumulée sur le temps pour chaque génération (nombre actif par gen)
    history_active = history_pivoted.cumsum()
    # 5. Réorganiser les colonnes (Générations les plus élevées en bas)
    # On trie les colonnes par ordre décroissant
    sorted_cols = sorted(history_active.columns, reverse=True)
    history_active = history_active[sorted_cols]

    # 6. Visualisation avec stackplot (équivalent cumulé de fill_between)
    plt.figure(figsize=(9, 5))

    # stackplot prend les x (index) et les y (les valeurs de chaque colonne)
    plt.stackplot(history_active.index,
                  [history_active[col] for col in history_active.columns],
                  labels=history_active.columns, colors = colors_10,
                  alpha=0.8)

    x_max = history_active.index.max()
    x_min = 1970
    x_ticks = np.arange(x_min, x_max + 1, 5)
    plt.xticks(x_ticks,x_ticks)
    plt.title("Aggregate fleet size evolution, per technology")
    plt.xlabel("Year")
    plt.ylim(ymin=0)
    plt.xlim(xmin = x_min, xmax=x_max)
    plt.ylabel("Number of active aircraft")
    plt.legend(loc='upper left', title="Generation")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('figures/fleet_figures' + title + '.pdf', format='pdf')
    plt.close()

def visu_prod(df, mapping = mapping_engines, reso = 1, title = 'fleet_prod_ex'):
    df = df.copy()
    df['gen'] = df['Engine'].map(mapping)

    deliveries = df[['gen', 'Delivery Date']].copy()
    deliveries['Delivery Date'] = ((deliveries['Delivery Date']-1970)/reso).astype(int)*reso+1970+reso/2

    deliveries['change'] = 1 #possibility to weight by the number of available seats, if it is in the dataset
    deliveries = deliveries.rename(columns={'Delivery Date': 'date'})

    daily_changes = deliveries.groupby(['date', 'gen'])['change'].sum().reset_index()
    # On pivote pour avoir les dates en index et les générations en colonnes
    history = daily_changes.pivot(index='date', columns='gen', values='change').fillna(0)
    sorted_cols = sorted(history.columns, reverse=True)
    history = history[sorted_cols]

    # 6. Visualisation avec stackplot (équivalent cumulé de fill_between)
    plt.figure(figsize=(9, 5))

    # stackplot prend les x (index) et les y (les valeurs de chaque colonne)
    plt.stackplot(history.index,
                  [history[col] for col in history.columns],
                  labels=history.columns, colors=colors_10,
                  alpha=0.8)
    x_max = history.index.max()
    x_min = 1970
    x_ticks = np.arange(x_min, x_max + 1, 5)
    plt.xticks(x_ticks, x_ticks)
    plt.title("Delivery volumes per generation")
    plt.xlabel("Year")
    plt.ylim(ymin=0)
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylabel("Yearly delivered aircraft")
    plt.legend(loc='upper left', title="Générations")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('figures/fleet_figures/' + title + '.pdf', format='pdf')
    plt.close()

def visu_retirements(df, mapping = mapping_engines, reso_1 = 1, reso_2 = 1, title = 'fleet_retirements_ex'):
    df = df.copy()
    df['gen'] = df['Engine'].map(mapping)

    retirements = df[(df['Event Date'].notna())&(df['Status']=='Written Off')][['gen', 'Delivery Date','Event Date']].copy()
    retirements['Event Date'] = ((retirements['Event Date']-1990)/reso_1).astype(int)*reso_1+1990+reso_1/2
    retirements['Delivery Date'] = ((retirements['Delivery Date']-1970)/reso_2).astype(int)*reso_2+1970+reso_2/2
    retirements['change'] = 1 #possibility to weight by the number of available seats, if it is in the dataset
    retirements = retirements.rename(columns={'Event Date': 'date'})

    daily_changes = retirements.groupby(['date', 'gen'])['change'].sum().reset_index()
    # On pivote pour avoir les dates en index et les générations en colonnes
    history = daily_changes.pivot(index='date', columns='gen', values='change').fillna(0)
    sorted_cols = sorted(history.columns, reverse=True)
    history = history[sorted_cols]

    # 6. Visualisation avec stackplot (équivalent cumulé de fill_between)
    plt.figure(figsize=(9,5))

    # stackplot prend les x (index) et les y (les valeurs de chaque colonne)
    plt.stackplot(history.index,
                  [history[col] for col in history.columns],
                  labels=history.columns, colors=colors_10,
                  alpha=0.8)


    daily_changes_completed = retirements.groupby(['date', 'gen','Delivery Date'])['change'].sum().reset_index()
    date, gen, deliv, change = daily_changes_completed['date'], daily_changes_completed['gen'], daily_changes_completed['Delivery Date'], daily_changes_completed['change']
    dates_u = np.sort(np.unique(date))#croissant
    gens_u = np.sort(np.unique(gen))[::-1]
    deliv_u = np.sort(np.unique(deliv))[::-1]   # décroissant
    t, g, d = len(dates_u), len(gens_u), len(deliv_u)
    m = np.zeros((t, g, d), dtype=float)
    t_idx = np.searchsorted(dates_u, date)
    g_idx = np.searchsorted(gens_u[::-1], gen)
    d_idx = np.searchsorted(deliv_u[::-1], deliv)
    np.add.at(m, (t_idx, g_idx, d_idx), change)
    m_flat = m.reshape(m.shape[0], -1)
    m_flat = np.cumsum(m_flat[:,::-1], axis=1)
    for i in range(m_flat.shape[1]):
        plt.plot(dates_u, m_flat[:,i], color = 'black', linestyle='--', linewidth=0.5)


    x_max = history.index.max()
    x_min = 1990
    x_ticks = np.arange(x_min, x_max + 1, 5)
    plt.xticks(x_ticks, x_ticks)
    plt.title("Retirements volumes per generation")
    plt.xlabel("Year")
    plt.ylim(ymin=0)
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylabel("Yearly retired aircraft")
    plt.legend(loc='upper left', title="Generations")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('figures/fleet_figures/' + title + '.pdf', format='pdf')
    plt.close()

def retirement_ranking(df):
    df_a = df[df['Status'].isin(['Active', 'Stored'])]
    df_r = df[df['Status'] == 'Written Off']

    df_r = df_r.sort_values(by='Event Date')
    df_r['rank'] = df_r['Event Date'].rank(method='min').astype(int)

    df_a['rank'] = max(df_r['rank']) + 1
    df_f = pd.concat([df_r, df_a], sort=False).reset_index(drop=True)
    return df_f


def visu_deliveries_array(deliveries, obs_sizes, obs_names, period_duration, n_market,graph_name = 'ex', color_mix = colors_26):
    T, M = deliveries.shape
    deliveries = deliveries * obs_sizes[None,:]
    col_sums = deliveries.sum(axis=0)
    N_m = (col_sums>0).sum()

    # indices des plus gros volumes
    top_idx = np.argsort(-col_sums)[:min(n_market * 2-3,N_m)]
    # calcul de la date moyenne d'utilisation
    mean_date = []

    for m in range(M):
        if col_sums[m] > 0:
            t_vals = np.arange(T)
            mean_date.append((t_vals * deliveries[:, m]).sum() / col_sums[m])
        else:
            mean_date.append(-np.inf)

    # masque top
    is_top = np.zeros(M, dtype=bool)
    is_top[top_idx] = True

    # tri
    cols_sorted = sorted(
        range(M),
        key=lambda m: (
            not is_top[m],  # priorité au top
            -mean_date[m]  # date moyenne décroissante
        )
    )

    rank = {m: k for k, m in enumerate(cols_sorted)}

    # --- TRI SELON RANK ---
    ordered_indices = sorted(
        range(len(rank)),
        key=lambda i: rank[i]
    )

    deliveries = deliveries[:, ordered_indices]
    obs_names = [obs_names[i] for i in ordered_indices]
    n_bars = deliveries.shape[0]
    p_categories = deliveries.shape[1]
    x = np.arange(n_bars) * period_duration + 2024
    bottom = np.zeros(n_bars)
    top = np.zeros(n_bars)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plt.grid(axis='y', color='grey', linestyle='--')
    for i in range(min(p_categories, N_m)):
        top = top + deliveries[:, i]
        if i < n_market:
            ax.fill_between(x, bottom, top, label=obs_names[i],
                            color=color_mix[i], linewidth=0, alpha=1, zorder=2)
            bottom = top
        elif i < 2 * n_market - 3:
            ax.fill_between(x, bottom, top, label=obs_names[i],
                            color=color_mix[i - n_market], linewidth=0, edgecolor='0.2', alpha=1, hatch='..', zorder=2)
            bottom = top
    ax.fill_between(x, bottom, top, alpha=0.8, color='grey', linewidth=0, edgecolor='white', label='Others', hatch='//',
                    zorder=2)
    ax.legend(framealpha=1, bbox_to_anchor=(1.05, 1),
              loc='upper left', borderaxespad=0., ncol=2)
    plt.xlim(x.min(), x.max())
    plt.ylim(0, 1.05 * top.max())
    plt.xlabel('Years')
    plt.ylabel('Produced aircraft seats')
    plt.tight_layout()
    plt.savefig('figures/integrated_observation/scenario/' + graph_name + '_prod_mod.pdf')
    plt.show()
    return None

def visu_retirements_array(vol_obs, obs_names, period_duration, n_market,graph_name = 'ex', color_mix = colors_26):
    vol_obs_p = np.maximum.accumulate(vol_obs[::-1, :], axis=0)[::-1, :] #effet de cumulé sur le retrait.
    retirement_seats_volumes = -np.diff(vol_obs_p, axis=0)
    retirement_seats_volumes_pos = np.where(retirement_seats_volumes < 0, 0, retirement_seats_volumes)
    type_obs = retirement_seats_volumes_pos.sum(axis=1)

    retirement_seats_volumes2 = -np.diff(vol_obs, axis=0)
    n_y_mod = retirement_seats_volumes2.shape[0]
    n_y_ac = retirement_seats_volumes2.shape[1]
    n_y_past = n_y_ac - n_y_mod
    i_idx = np.arange(n_y_mod)[:, None]  # shape (n, 1)
    j_idx = np.arange(n_y_ac)[None, :]  # shape (1, m)
    mask = j_idx == i_idx+n_y_past  # shape (n, m)
    retirement_seats_volumes2[mask,:]=0
    type_obs2 = retirement_seats_volumes2.sum(axis=1)

    T, M = type_obs.shape

    col_sums = type_obs.sum(axis=0)

    # indices des plus gros volumes
    top_idx = np.argsort(-col_sums)[:n_market *2-3]

    # calcul de la date moyenne d'utilisation
    mean_date = []

    for m in range(M):
        if col_sums[m] > 0:
            t_vals = np.arange(T)
            mean_date.append((t_vals * type_obs[:, m]).sum() / col_sums[m])
        else:
            mean_date.append(-1)

    # masque top
    is_top = np.zeros(M, dtype=bool)
    is_top[top_idx] = True

    # tri
    cols_sorted = sorted(
        range(M),
        key=lambda m: (
            not is_top[m],  # priorité au top
            -mean_date[m]  # date moyenne décroissante
        )
    )

    rank = {m: k for k, m in enumerate(cols_sorted)}

    # --- TRI SELON RANK ---
    ordered_indices = sorted(
        range(len(rank)),
        key=lambda i: rank[i]
    )

    type_obs = type_obs[:, ordered_indices]
    type_obs2 = type_obs2[:, ordered_indices]
    obs_names = [obs_names[i] for i in ordered_indices]

    n_bars = type_obs.shape[0]
    p_categories = type_obs.shape[1]

    x = np.arange(n_bars) * period_duration + 2024
    bottom = np.zeros(n_bars)
    bottom2 = np.zeros(n_bars)
    top = np.zeros(n_bars)
    top2 = np.zeros(n_bars)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plt.grid(axis='y', color='grey', linestyle='--')
    for i in range(p_categories):
        top = top + type_obs[:, i]
        top2 = top2 + type_obs2[:, i]
        if i < n_market:
            bottom_visu = np.where(bottom<=0,np.nan, bottom)
            top_visu = np.where(top <= 0, np.nan, top)
            ax.fill_between(x, bottom_visu, top_visu, label=obs_names[i],
                                color=color_mix[i], linewidth=0, alpha=1, zorder = 2)
            ax.fill_between(x, bottom2, top2,color=color_mix[i], linewidth=0, alpha=0.4, zorder=1)
            bottom = top
            bottom2 = top2
        elif i < 2*n_market-3:
            bottom_visu = np.where(bottom <= 0, np.nan, bottom)
            top_visu = np.where(top <= 0, np.nan, top)
            ax.fill_between(x, bottom_visu, top_visu, label=obs_names[i],
                                color=color_mix[i-n_market], linewidth=0, edgecolor='0.2', alpha=1, hatch='..', zorder = 2)
            ax.fill_between(x, bottom2, top2, color=color_mix[i-n_market], linewidth=0, alpha=0.4, zorder=1)
            bottom = top
            bottom2 = top2
        # else :
        #     ax.fill_between(x, bottom, top, linewidth=0, alpha=0.9)
        #     bottom = top
    ax.fill_between(x, bottom, top, alpha=0.8, color='grey', linewidth=0, edgecolor='white', label='Others', hatch='//', zorder = 2)
    ax.fill_between(x, bottom2, top2, alpha=0.2, color='grey', linewidth=0, edgecolor='white', hatch='//', zorder = 1)
    ax.fill_between(x, 0, 0, color='grey', linewidth=0.5, edgecolor='black', alpha=0.1, label='Temporary \n storage')
    ax.legend(framealpha=1, bbox_to_anchor=(1.05, 1),
              loc='upper left', borderaxespad=0., ncol = 2)
    plt.xlim(x.min(), x.max())
    plt.ylim(min(0,1.05*top2.min()), 1.05 * top.max())
    plt.xlabel('Years')
    plt.ylabel('Retired aircraft seats')
    plt.tight_layout()
    plt.savefig('figures/integrated_observation/scenario/' + graph_name+'_retir_mod.pdf')
    plt.show()
    return None

def constraint_plot(traff_arrays, active_fleet_arrays, ranges_c, title ='test', years = [2024,2050]):
    N_p = traff_arrays.shape[0]
    values_to_sort2 = np.array(ranges_c)
    sorted_indices2 = np.argsort(values_to_sort2)
    for t in range(N_p): #tri par distance puis somme cumulée sur les deux arrays
        values_to_sort = traff_arrays[t, :, 0]
        sorted_indices = np.argsort(values_to_sort)[::-1]
        traff_arrays[t, :, :] = traff_arrays[t, sorted_indices, :]
        traff_arrays[t,:,1] = np.cumsum(traff_arrays[t,:,1], axis=0)

        active_fleet_arrays[t, :] = active_fleet_arrays[t, sorted_indices2[::-1]]
        active_fleet_arrays[t, :] = np.cumsum(active_fleet_arrays[t, :], axis=0)
    years_array = np.linspace(years[0], years[1], N_p)
    norm = mpl.colors.Normalize(vmin=years[0], vmax=years[1])
    cmap = plt.get_cmap('turbo')  # ou 'Spectral', 'viridis', 'plasma', etc.
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.grid(axis='both', color='grey', linestyle='--')
    for t in range(N_p):
        color = cmap(norm(years_array[t]))
        max_value =  traff_arrays[t, -1, 1]
        ax.step(traff_arrays[t, :, 0], traff_arrays[t, :, 1]/max_value, linestyle ='-',where='pre', color=color, linewidth=0.5)
        ax.step(np.concatenate([np.array([0]),ranges_c[sorted_indices2],np.array([ranges_c[sorted_indices2][-1]])]),
                np.concatenate([np.array([1]),active_fleet_arrays[t][::-1]/max_value,np.array([0])]),where='pre', linestyle = '--', color=color, linewidth=0.5)
    ax.set_xlabel('Connection distance /Max. Range (km)', fontsize = 13)
    ax.set_ylabel('Share of the total seats', fontsize = 13)
    ax.plot([0,0],[0,0], label = 'Active fleet capabilities', linestyle = '--', color = 'black', linewidth=1)
    ax.plot([0,0],[0,0], label = 'Traffic distances', linestyle = '-', color = 'black', linewidth=1)
    plt.legend(loc='upper right', framealpha=1, fontsize = 13)
    plt.xlim(0)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.2))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # requis pour matplotlib

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Year', fontsize = 13)
    plt.savefig('figures/integrated_observation/scenario/range_constraints/constraint_plot_'+title+'.pdf')