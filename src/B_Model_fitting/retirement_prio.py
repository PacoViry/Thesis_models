import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
### the entry dataset is composed of several columns : 'Aircraft Type','Engine', 'Delivery Date', 'Status', 'Event Date'
### Status vaut written off, stored ou active
### Each column describes an engine which is or was part of the fleet
# The following module contains :
# 1) A function to compute individual coefficients fitting a statistical model
# 2) A function to compute these coefficients based on a linear age representation.


def setting_df(ac_df):
    Liste_classements = ac_df['rank'].unique()
    # Séparer les lignes où 'C' vaut 'a' et 'b'
    filter = ac_df['rank'] == Liste_classements[-1]
    df_a = ac_df[~filter]
    df_b = ac_df[filter]

    df_a['Type_Year'] = df_a['Aircraft Type'] +' '+ df_a['Delivery Date'].astype(int).astype(str)
    df_b['Type_Year'] = df_b['Aircraft Type'] +' '+ df_b['Delivery Date'].astype(int).astype(str)

    df_b_filtered = df_b[df_b['Type_Year'].isin(df_a['Type_Year'].unique())]

    # Fusionner les DataFrames pour obtenir le résultat final
    file_f = pd.concat([df_a, df_b_filtered])
    file_f.reset_index(inplace= True, drop=True)
    return(file_f)


def random_list(liste_r, params):
    year_min = liste_r['Delivery Date'].min()
    year_max = liste_r['Delivery Date'].max()
    liste_modeles = sorted(liste_r['Aircraft Type'].unique())
    propensions = ini_propensions(params)
    classements = liste_r['rank'].unique()
    lignes = []
    l_e = liste_r[liste_r['rank'] <= classements[-1]][['Aircraft Type', 'Delivery Date', 'rank']]
    for c in classements[:-1]:
        extract = l_e[l_e['rank'] == c]
        N = extract.shape[0]
        if N > 1:
            pivot_table = extract[['Aircraft Type', 'Delivery Date', 'rank']].pivot_table(values='rank',
                                                                                       index='Aircraft Type',
                                                                                       columns='Delivery Date',
                                                                                       aggfunc='count', fill_value=0)
            pivot_table = pivot_table.reindex(index=liste_modeles, columns=list(range(year_min, year_max + 1)),
                                              fill_value=0)
            flotte = pivot_table.to_numpy()
            probas1 = propensions * flotte  ###on teste pour voir si ça bugue pas et si on fait mieux que 94s
            s = probas1.shape
            for i in range(N):  ### tirage simplifié ici, à améliorer et réfléchir
                probas2 = probas1 / probas1.sum()
                probas_cum = probas2.cumsum()
                tirage = rd.random()
                modele, date = np.unravel_index(np.argmax(probas_cum >= tirage), s)
                probas1[modele, date] += -propensions[modele, date]
                lignes.append([liste_modeles[modele], date + year_min, 1])
        else:
            lignes.append(extract.iloc[0].tolist())
    nouveau_df = pd.DataFrame(lignes, columns=['Aircraft Type', 'Delivery Date', 'rank'])
    nouveau_df['rank'] = nouveau_df.index + 1
    nouveau_df = pd.concat([nouveau_df, liste_r[liste_r['rank'] == classements[-1]]], sort=False)
    nouveau_df = nouveau_df.reset_index(drop=True)
    return (nouveau_df)


def heatmap_retirements(liste, n = 480, ymin = 0):
    resultats = np.zeros((n + 10, n + 10))
    resultats = np.where(resultats == 0.0, np.nan, resultats)
    # Calculer les valeurs de la fonction f2 pour toutes les combinaisons de a et b
    for i in range(len(liste['Delivery Date'])):
        if (i // n) % 2 == 0:
            if i % n == n - 1:
                resultats[n - 20 * (i // n) - 33:n - 20 * (i // n), i % n + 5:i % n + 7] = liste['Delivery Date'].iloc[
                                                                                               i] + ymin
            else:
                resultats[n - 20 * (i // n) - 13:n - 20 * (i // n), i % n + 5] = liste['Delivery Date'].iloc[i] + ymin
        else:
            if i % n == n - 1:
                resultats[n - 20 * (i // n) - 33:n - 20 * (i // n), n - 1 - i % n + 3:n - 1 - i % n + 6] = \
                liste['Delivery Date'].iloc[i] + ymin
            else:
                resultats[n - 20 * (i // n) - 13:n - 20 * (i // n), n - 1 - i % n + 5] = liste['Delivery Date'].iloc[
                                                                                             i] + ymin
    # Créer la heatmap avec seaborn
    sns.heatmap(resultats, cmap='Spectral')
    # Ajouter des étiquettes
    sns.set(style='whitegrid')
    plt.gcf().set_size_inches(10, 6)
    plt.axis('off')
    # Enlever la grille
    plt.grid(False)
    # plt.savefig('data_input_TAA.png', bbox_inches='tight', format='png')
    plt.show()
    return None

def ini_propensions(params):
    propensions = np.exp(-params)
    return (propensions)


def grad_minibatch(liste_r, params):  # mode 0, classique
    year_min = liste_r['Delivery Date'].min()
    year_max = liste_r['Delivery Date'].max()
    classements = liste_r['rank'].unique()[:-1]

    liste_modeles = sorted(liste_r['Aircraft Type'].unique())
    propensions = ini_propensions(params)

    pivot_table = liste_r[['Aircraft Type', 'Delivery Date', 'rank']].pivot_table(values='rank', index='Aircraft Type',
                                                                               columns='Delivery Date', aggfunc='count',
                                                                               fill_value=0)
    pivot_table = pivot_table.reindex(index=liste_modeles, columns=list(range(year_min, year_max + 1)), fill_value=0)
    flotte = pivot_table.to_numpy()
    grad_a = np.zeros((len(liste_modeles), year_max - year_min + 1))
    Probas = (propensions * flotte)

    liste_r['Delivery Date'] = liste_r['Delivery Date'] - year_min
    mapping = {value: idx for idx, value in enumerate(liste_modeles)}
    liste_r['Aircraft Type'] = liste_r['Aircraft Type'].map(mapping)
    data = liste_r[['Aircraft Type', 'Delivery Date']].values[0:classements[-1]]
    for row in data:
        mod = row[0]
        mill = row[1]
        grad_a += grad_uni(mod, mill, Probas)
        Probas[mod, mill] += -propensions[mod, mill]
    return (grad_a)


def grad_uni(mod, mill, Probas):
    grad = np.zeros(Probas.shape)
    quotient = Probas.sum().sum()
    grad[mod, mill] = -1
    grad += Probas / quotient
    return (grad)

def l_v(liste_r, params):
    year_min = liste_r['Delivery Date'].min()
    year_max = liste_r['Delivery Date'].max()
    classements = liste_r['rank'].unique()[:-1]
    liste_modeles = sorted(liste_r['Aircraft Type'].unique())

    l_v = 0
    propensions = ini_propensions(params)

    pivot_table = liste_r[['Aircraft Type', 'Delivery Date', 'rank']].pivot_table(values='rank', index='Aircraft Type',
                                                                               columns='Delivery Date', aggfunc='count',
                                                                               fill_value=0)
    pivot_table = pivot_table.reindex(index=liste_modeles, columns=list(range(year_min, year_max + 1)), fill_value=0)
    flotte = pivot_table.to_numpy()
    Probs = propensions * flotte

    liste_r['Delivery Date'] = liste_r['Delivery Date'] - year_min
    mapping = {value: idx for idx, value in enumerate(liste_modeles)}
    liste_r['Aircraft Type'] = liste_r['Aircraft Type'].map(mapping)
    data = liste_r[['Aircraft Type', 'Delivery Date']].values[0:classements[-1]]
    for row in data:
        mod = row[0]
        mill = row[1]
        l_v += np.log(flotte[mod, mill]) - params[mod, mill] - np.log((Probs).sum().sum())
        Probs[mod, mill] += -propensions[mod, mill]
    return (l_v)

def fit_type_y(df, N = 50, M = 30):
    list_ac = sorted(df['Aircraft Type'].unique())
    df['Delivery Date'] = df['Delivery Date'].astype(int)
    year_min = df['Delivery Date'].min()
    year_max = df['Delivery Date'].max()

    vals = np.zeros((len(list_ac), year_max - year_min + 1))
    v_list = []
    for i in range(N):
        print(str(i), end = ': ')
        rd_file = random_list(df, vals)
        v_list.append(l_v(rd_file, vals))
        print(v_list[-1], end=', ')
        m = 1
        for q in range(1, M):
            m+= -1
            u = grad_minibatch(rd_file, vals)
            l_v_ref = l_v(rd_file, vals + 0.01 * u * 2 ** m)
            if l_v_ref > v_list[-1] :
                l_v_ref_2 = l_v(rd_file, vals + 0.01 * u * 2 ** (m+1))
                while l_v_ref_2 > l_v_ref :
                    m+=1
                    l_v_ref = l_v_ref_2
                    l_v_ref_2 = l_v(rd_file, vals + 0.01 * u * 2 ** (m+1))
            else :
                print('reduction needed')
                while l_v_ref < v_list[-1]:
                    m -= 1
                    l_v_ref = l_v(rd_file, vals + 0.01 * u * 2 ** m)
            vals += 0.01 * u * 2 ** (m)
            print(l_v_ref, end = ', ')
            v_list.append(l_v_ref)
    print ('ok')
    v_list.append(l_v(df, vals))
    vals3 = np.where(vals == 0.0, np.nan, vals)
    for j in range(len(list_ac)):
        plt.scatter(np.arange(year_max - year_min + 1) + year_min, vals3[j, :], label=list_ac[j])
    plt.show()
    plt.plot(v_list)
    plt.show()
    return([year_min,year_max], list_ac, vals3)

def save_prop(vals, list_ac, y_lims, title = 'propensions_test'):
    save = pd.DataFrame(vals, index=np.arange(y_lims[0], y_lims[1] + 1), columns=list_ac)
    excel_path = 'data//retirement_propensions//'+title+'.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        save.to_excel(writer, sheet_name='propensions', index=True)
    return None
