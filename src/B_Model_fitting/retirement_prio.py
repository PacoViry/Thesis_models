import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
### the entry dataset is composed of several columns : 'Aircraft Type','Engine', 'Delivery Date', 'Status', 'Event Date'
### Status vaut written off, stored ou active
### Each column describes an engine which is or was part of the fleet
# The following module contains :
# 1) A function to compute individual coefficients fitting a statistical model
# 2) A function to compute these coefficients based on a linear age representation. (a remplir!)


def setting_df(ac_df):
    list_ranks = ac_df['rank'].unique()
    # Séparer les lignes où 'C' vaut 'a' et 'b'
    filter = ac_df['rank'] == list_ranks[-1]
    df_r = ac_df[~filter]
    df_a = ac_df[filter]

    df_r['Type_Year'] = df_r['Aircraft Type'].astype(str) +' '+ df_r['Delivery Date'].astype(int).astype(str)
    df_a['Type_Year'] = df_a['Aircraft Type'].astype(str) +' '+ df_a['Delivery Date'].astype(int).astype(str)

    df_a_filtered = df_a[df_a['Type_Year'].isin(df_r['Type_Year'].unique())]
    # df_b_filtered = df_b.copy() #on pose des problèmes sur la convergence.
    # Fusionner les DataFrames pour obtenir le résultat final
    file_f = pd.concat([df_r, df_a_filtered])
    file_f.reset_index(inplace= True, drop=True)
    return file_f

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


def random_list(liste_r, params):
    list_ac = sorted(liste_r['Aircraft Type'].unique())
    list_dates = sorted(liste_r['Delivery Date'].unique())
    propensions = ini_propensions(params)
    classements = liste_r['rank'].unique()
    lignes = []
    l_e = liste_r[liste_r['rank'] <= classements[-1]][['Aircraft Type', 'Delivery Date', 'rank']]
    for c in classements[:-1]:
        extract = l_e[l_e['rank'] == c]
        n_r = extract.shape[0]
        if n_r > 1:
            pivot_table = (extract[['Aircraft Type', 'Delivery Date', 'rank']].pivot_table(values='rank',
                                                                                          index='Aircraft Type',
                                                                                          columns='Delivery Date',
                                                                                          aggfunc='count',
                                                                                          fill_value=0)
                           .reindex(index=list_ac, columns=list_dates, fill_value=0))
            pivot_table.sort_index(axis=0, inplace=True)
            pivot_table.sort_index(axis=1, inplace=True)
            flotte = pivot_table.to_numpy()
            s = propensions.shape
            probas1 = propensions * flotte  ###on teste pour voir si ça bugue pas et si on fait mieux que 94s
            for i in range(n_r):  ### tirage simplifié ici, à améliorer et réfléchir
                probas2 = probas1 / probas1.sum()
                probas_cum = probas2.cumsum()
                tirage = rd.random()
                modele, date = np.unravel_index(np.argmax(probas_cum >= tirage), s)
                probas1[modele, date] += -propensions[modele, date]
                lignes.append([list_ac[modele], list_dates[date], c+i])
        else:
            lignes.append(extract.iloc[0].tolist())
    nouveau_df = pd.DataFrame(lignes, columns=['Aircraft Type', 'Delivery Date', 'rank'])
    nouveau_df['rank'] = nouveau_df.index + 1
    nouveau_df = pd.concat([nouveau_df, liste_r[liste_r['rank'] == classements[-1]]], sort=False)
    nouveau_df = nouveau_df.reset_index(drop=True)
    return nouveau_df

def ini_propensions(params):
    propensions = np.exp(-params)
    return (propensions)

def grad_minibatch(liste_r, params):  # mode 0, classique
    n_ac = params.shape[0]
    list_ac = np.arange(n_ac)
    list_dates = sorted(liste_r['Delivery Date'].unique())
    classements = liste_r['rank'].unique()[:-1]
    propensions = ini_propensions(params)

    pivot_table = (liste_r[['Aircraft Type', 'Delivery Date', 'rank']].pivot_table(values='rank',
                                                                                   index='Aircraft Type',
                                                                                   columns='Delivery Date',
                                                                                   aggfunc='count',
                                                                                   fill_value=0)
                   .reindex(index=list_ac, columns=list_dates, fill_value=0))
    pivot_table.sort_index(axis=0, inplace=True)
    pivot_table.sort_index(axis=1, inplace=True)

    flotte = pivot_table.to_numpy()
    grad_a = np.zeros((len(list_ac),len(list_dates)))
    probas = (propensions * flotte)

    liste_r['Delivery Date'] = liste_r['Delivery Date']
    mapping = {value: idx for idx, value in enumerate(list_ac)}
    liste_r['Aircraft Type'] = liste_r['Aircraft Type'].map(mapping)
    mapping2 = {value: idx for idx, value in enumerate(list_dates)}
    liste_r['Delivery Date'] = liste_r['Delivery Date'].map(mapping2)
    data = liste_r[['Aircraft Type', 'Delivery Date']].values[0:classements[-1]]
    for row in data:
        mod = row[0]
        mill = row[1]
        grad_a += grad_uni(mod, mill, probas)
        probas[mod, mill] += -propensions[mod, mill]
    return grad_a

def grad_uni(mod, mill, probas):
    grad = np.zeros(probas.shape)
    quotient = probas.sum().sum()
    grad[mod, mill] = -1
    grad += probas / quotient
    return grad

def l_v(liste_r, params):
    list_ac = sorted(liste_r['Aircraft Type'].unique())
    list_dates = sorted(liste_r['Delivery Date'].unique())
    classements = liste_r['rank'].unique()[:-1]
    llike = 0
    propensions = ini_propensions(params)
    pivot_table = (liste_r[['Aircraft Type', 'Delivery Date', 'rank']].pivot_table(values='rank',
                                                                                   index='Aircraft Type',
                                                                                   columns='Delivery Date',
                                                                                   aggfunc='count',
                                                                                   fill_value=0)
                   .reindex(index=list_ac, columns=list_dates, fill_value=0))
    pivot_table.sort_index(axis = 0, inplace =True)
    pivot_table.sort_index(axis=1, inplace =True)
    flotte = pivot_table.to_numpy()
    probs = propensions * flotte

    liste_r['Delivery Date'] = liste_r['Delivery Date']
    mapping = {value: idx for idx, value in enumerate(list_ac)}
    liste_r['Aircraft Type'] = liste_r['Aircraft Type'].map(mapping)
    mapping2 = {value: idx for idx, value in enumerate(list_dates)}
    liste_r['Delivery Date'] = liste_r['Delivery Date'].map(mapping2)
    data = liste_r[['Aircraft Type', 'Delivery Date']].values[0:classements[-1]]
    for row in data:
        mod = row[0]
        mill = row[1]
        llike += np.log(flotte[mod, mill]) - params[mod, mill] - np.log(probs.sum().sum())
        probs[mod, mill] += -propensions[mod, mill]
    return llike

def fit_type_y(df_o, epoch = 50, rep = 30, vals = None, n_ac = None, title = 'TAA_free'):
    df = df_o.copy()
    n_r = df['rank'].max()
    df['Delivery Date'] = df['Delivery Date'].astype(int)
    list_dates = sorted(df['Delivery Date'].unique())
    if n_ac is None :
        n_ac = len(sorted(df['Aircraft Type'].unique()))
        list_ac = sorted(df['Aircraft Type'].unique())
    else :
        list_ac = np.arange(n_ac)
    n_dates = len(list_dates)
    if vals is None :
        vals = np.zeros((n_ac, n_dates))

    # --- Adam hyperparameters ---
    learning_rate = 0.02
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # --- Adam states ---
    m_vals = np.zeros_like(vals)
    v_vals = np.zeros_like(vals)
    t = 0

    v_list = []
    for i in range(epoch):
        print(str(i), end = ': ')
        rd_file = random_list(df, vals)
        v_list.append(l_v(rd_file, vals))
        print(int(10**4*v_list[-1]/n_r)/10**4, end=', ')

        for q in range(1, rep):
            grad_vals = grad_minibatch(rd_file, vals)

            t += 1
            m_vals = beta1 * m_vals + (1 - beta1) * grad_vals
            v_vals = beta2 * v_vals + (1 - beta2) * (grad_vals ** 2)

            m_vals_hat = m_vals / (1 - beta1 ** t)
            v_vals_hat = v_vals / (1 - beta2 ** t)

            vals += learning_rate * m_vals_hat / (np.sqrt(v_vals_hat) + epsilon)

            l_v_ref = l_v(rd_file, vals)
            print(int(10**4*l_v_ref/n_r)/10**4, end = ', ')
            v_list.append(l_v_ref)
        print('')
    print('ok')
    v_list.append(l_v(df, vals))
    vals3 = np.where(vals == 0.0, np.nan, vals)
    for j in range(len(list_ac)):
        plt.scatter(np.array(list_dates), vals3[j, :], label=list_ac[j])
    plt.savefig('figures//estimators_figures//retir_propensions_'+title+'.png')
    plt.show()
    plt.plot(v_list)
    plt.show()
    return vals3, list_dates, list_ac

def save_prop(vals, list_ac, y_lims, title = 'propensions_test'):
    save = pd.DataFrame(vals, index=y_lims, columns=list_ac)
    excel_path = 'data//retirement_propensions//'+title+'.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        save.to_excel(writer, sheet_name='propensions', index=True)
    return None


def load_prop(title='propensions_test'):
    excel_path = 'data//retirement_propensions//' + title + '.xlsx'
    df = pd.read_excel(
        excel_path,
        sheet_name='propensions',
        index_col=0
    )
    list_dates = list(df.index)
    list_ac = list(df.columns)
    vals = df.values
    return vals, list_dates, list_ac

def setting_df_lin(ac_df):
    list_ranks = ac_df['rank'].unique()
    # Séparer les lignes où 'C' vaut 'a' et 'b'
    filter = ac_df['rank'] == list_ranks[-1]
    df_r = ac_df[~filter]
    df_a = ac_df[filter]

    df_a_filtered = df_a[df_a['Aircraft Type'].isin(df_r['Aircraft Type'].unique())]
    # df_b_filtered = df_b.copy() #on pose des problèmes sur la convergence.
    # Fusionner les DataFrames pour obtenir le résultat final
    file_f = pd.concat([df_r, df_a_filtered])
    file_f.reset_index(inplace= True, drop=True)
    return file_f

def random_list_lin(liste_r, params, beta):
    list_dates = sorted(liste_r['Delivery Date'].unique())
    propensions = ini_propensions_lin(params, beta, max(list_dates)-min(list_dates)+1)
    n_ac = params.shape[0]
    list_ac = np.arange(n_ac)
    classements = liste_r['rank'].unique()
    lignes = []
    l_e = liste_r[liste_r['rank'] <= classements[-1]][['Aircraft Type', 'Delivery Date', 'rank']]
    for c in classements[:-1]:
        extract = l_e[l_e['rank'] == c]
        n_r = extract.shape[0]
        if n_r > 1:
            pivot_table = (extract[['Aircraft Type', 'Delivery Date', 'rank']].pivot_table(values='rank',
                                                                                          index='Aircraft Type',
                                                                                          columns='Delivery Date',
                                                                                          aggfunc='count',
                                                                                          fill_value=0)
                           .reindex(index=list_ac, columns=np.arange(min(list_dates),max(list_dates)+1), fill_value=0))
            pivot_table.sort_index(axis=0, inplace=True)
            pivot_table.sort_index(axis=1, inplace=True)
            flotte = pivot_table.to_numpy()
            s = propensions.shape
            probas1 = propensions * flotte  ###on teste pour voir si ça bugue pas et si on fait mieux que 94s
            for i in range(n_r):  ### tirage simplifié ici, à améliorer et réfléchir
                probas2 = probas1 / probas1.sum()
                probas_cum = probas2.cumsum()
                tirage = rd.random()
                modele, date = np.unravel_index(np.argmax(probas_cum >= tirage), s)
                probas1[modele, date] += -propensions[modele, date]
                lignes.append([list_ac[modele], date+min(list_dates), c+i])
        else:
            lignes.append(extract.iloc[0].tolist())
    nouveau_df = pd.DataFrame(lignes, columns=['Aircraft Type', 'Delivery Date', 'rank'])
    nouveau_df['rank'] = nouveau_df.index + 1
    nouveau_df = pd.concat([nouveau_df, liste_r[liste_r['rank'] == classements[-1]]], sort=False)
    nouveau_df = nouveau_df.reset_index(drop=True)
    return nouveau_df

def ini_propensions_lin(params, beta, n_dates):
    propensions = np.exp(-(params[:,np.newaxis]+(beta*np.arange(n_dates))[np.newaxis,:]))
    return propensions

def grad_minibatch_lin(liste_r, params, beta):  # mode 0, classique
    n_ac = params.shape[0]
    list_ac = np.arange(n_ac)
    list_dates = sorted(liste_r['Delivery Date'].unique())
    classements = liste_r['rank'].unique()[:-1]
    propensions = ini_propensions_lin(params, beta,max(list_dates)+1-min(list_dates))

    pivot_table = (liste_r[['Aircraft Type', 'Delivery Date', 'rank']].pivot_table(values='rank',
                                                                                   index='Aircraft Type',
                                                                                   columns='Delivery Date',
                                                                                   aggfunc='count',
                                                                                   fill_value=0)
                   .reindex(index=list_ac, columns=np.arange(min(list_dates), max(list_dates) + 1), fill_value=0))
    pivot_table.sort_index(axis=0, inplace=True)
    pivot_table.sort_index(axis=1, inplace=True)

    flotte = pivot_table.to_numpy()
    grad_a = np.zeros(n_ac)
    grad_beta = 0
    probas = (propensions * flotte)
    j_list = np.arange(max(list_dates)-min(list_dates)+1)[np.newaxis, :]

    liste_r['Delivery Date'] = liste_r['Delivery Date']
    mapping = {value: idx for idx, value in enumerate(list_ac)}
    liste_r['Aircraft Type'] = liste_r['Aircraft Type'].map(mapping)
    data = liste_r[['Aircraft Type', 'Delivery Date']].values[0:classements[-1]]
    for row in data:
        mod = row[0]
        mill = row[1]
        probas_w = probas * (j_list-mill)
        u = grad_uni_lin(mod, probas,probas_w)
        grad_a += u[0]
        grad_beta += u[1]
        probas[mod, mill] += -propensions[mod, mill]
    return grad_a, grad_beta

def grad_uni_lin(mod, probas, probas_w):
    grad_a = np.zeros(probas.shape[0])
    quotient = probas.sum().sum()
    grad_a[mod] = -1
    grad_a += probas.sum(axis=1) / quotient
    grad_beta = probas_w.sum().sum()/quotient
    return grad_a, grad_beta

def l_v_lin(liste_r, params, beta):
    n_ac = params.shape[0]
    list_ac = np.arange(n_ac)
    list_dates = sorted(liste_r['Delivery Date'].unique())
    classements = liste_r['rank'].unique()[:-1]
    llike = 0
    propensions = ini_propensions_lin(params, beta,max(list_dates)+1-min(list_dates))

    pivot_table = (liste_r[['Aircraft Type', 'Delivery Date', 'rank']].pivot_table(values='rank',
                                                                                   index='Aircraft Type',
                                                                                   columns='Delivery Date',
                                                                                   aggfunc='count',
                                                                                   fill_value=0)
                   .reindex(index=list_ac, columns=np.arange(min(list_dates), max(list_dates) + 1), fill_value=0))
    pivot_table.sort_index(axis = 0, inplace =True)
    pivot_table.sort_index(axis=1, inplace =True)
    flotte = pivot_table.to_numpy()
    probs = propensions * flotte

    liste_r['Delivery Date'] = liste_r['Delivery Date']
    mapping = {value: idx for idx, value in enumerate(list_ac)}
    liste_r['Aircraft Type'] = liste_r['Aircraft Type'].map(mapping)
    data = liste_r[['Aircraft Type', 'Delivery Date']].values[0:classements[-1]]

    for row in data:
        mod = row[0]
        mill = row[1]
        llike += np.log(flotte[mod, mill]) - params[mod]-mill*beta - np.log(probs.sum().sum())
        probs[mod, mill] += -propensions[mod, mill]
    return llike

def fit_type_y_lin(df_o, epoch = 50, rep = 30, vals = None, beta = None,dico2_inv = None, n_ac = None ):
    df = df_o.copy()
    n_r = df['rank'].max()
    df['Delivery Date'] = df['Delivery Date'].astype(int)
    list_dates = sorted(df['Delivery Date'].unique())
    if n_ac is None :
        n_ac = len(sorted(df['Aircraft Type'].unique()))
    list_ac = np.arange(n_ac)
    n_dates = max(list_dates)-min(list_dates)+1
    if vals is None :
        vals = np.zeros(n_ac)
        beta = 0

    # --- Adam hyperparameters ---
    learning_rate = 0.02
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # --- Adam states ---
    m_vals = np.zeros_like(vals)
    v_vals = np.zeros_like(vals)
    m_beta = 0.0
    v_beta = 0.0
    t = 0

    if dico2_inv is None :
        list_types = sorted(df['Aircraft Type'].unique())
        dico2_inv = {value: idx for idx, value in enumerate(list_types)}
    v_list = []

    for i in range(epoch):
        print(str(i), end = ': ')
        rd_file = random_list_lin(df, vals, beta)
        v_list.append(l_v_lin(rd_file, vals, beta))
        print(int(10**4*v_list[-1]/n_r)/10**4, end=', ')

        # one Adam update per mini-batch
        for q in range(1, rep):
            grad_vals, grad_beta = grad_minibatch_lin(rd_file, vals, beta)

            t += 1
            m_vals = beta1 * m_vals + (1 - beta1) * grad_vals
            v_vals = beta2 * v_vals + (1 - beta2) * (grad_vals ** 2)
            m_beta = beta1 * m_beta + (1 - beta1) * grad_beta
            v_beta = beta2 * v_beta + (1 - beta2) * (grad_beta ** 2)

            m_vals_hat = m_vals / (1 - beta1 ** t)
            v_vals_hat = v_vals / (1 - beta2 ** t)
            m_beta_hat = m_beta / (1 - beta1 ** t)
            v_beta_hat = v_beta / (1 - beta2 ** t)

            vals += learning_rate * m_vals_hat / (np.sqrt(v_vals_hat) + epsilon)
            beta += learning_rate * m_beta_hat / (np.sqrt(v_beta_hat) + epsilon)

            l_v_ref = l_v_lin(rd_file, vals, beta)
            print(int(10**4*l_v_ref/n_r)/10**4, end = ', ')
            v_list.append(l_v_ref)
        print('')

    print('ok')
    v_list.append(l_v_lin(df, vals, beta))
    vals3 = np.where(vals == 0.0, np.nan, vals)
    plt.figure(figsize=(6, 8))
    plt.style.use('default')
    plt.grid(axis='both')
    if dico2_inv is None:
        plt.barh(list_ac, vals3)
    else:
        plt.barh(
            list_ac,
            vals3,
            tick_label=[dico2_inv[ac] for ac in list_ac])
    plt.xlabel("Retirement coefficient")
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()
    plt.plot(v_list)
    plt.show()
    return vals3, beta, list_ac

def save_prop_lin(vals, beta, list_ac, title = 'lin_propensions_test'):
    data = np.concatenate([np.array([beta]), vals])
    save = pd.DataFrame(data.reshape(1, -1), columns=['Beta'] + list(list_ac))
    excel_path = 'data//retirement_propensions//'+title+'.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        save.to_excel(writer, sheet_name='beta et propensions', index=True)
    return None


def load_prop_lin(title='lin_propensions_test'):
    excel_path = 'data//retirement_propensions//' + title + '.xlsx'
    df = pd.read_excel(
        excel_path,
        sheet_name='beta et propensions',
        index_col=0
    )
    list_ac = list(df.columns)[1:]
    vals = df.values[0,1:]
    beta = df.values[0,0]
    return vals, beta, list_ac