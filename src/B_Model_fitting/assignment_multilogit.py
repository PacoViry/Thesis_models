import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from scipy.optimize import minimize
from matplotlib.ticker import ScalarFormatter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# import time as tm
from src.Common_tools import vis_ref
importlib.reload(vis_ref)


#The entry dataset can contain the following columns :
# 'ADEP', 'ADEP Name',  'ADES', 'ADES Name', 'Period', 'Aircraft Type', 'Aircraft Type Name', 'Aircraft Operator','Aircraft Operator Name',
# 'N_flights', 'Seats', 'ASK', 'Activity', 'Distance_conn (km)', 'Flights_conn_p',
# 'Seats_conn_p', 'ASK_conn_p', 'Activity_conn_p', 'Weight'
#The following module allows :
#1) To compute the coefficients based on the method (4 methodes différentes), et la normalisation de la donnée (poids à intégrer)
#2) To evaluate prediction metrics for the model
#3) To connect aircraft caracteristics and assignment coefficients.


def data_formatting(df, seuils_f = 0.01, n_ac = 60, obs = 'ASK', weight = True, save_techn_carac = False, airports = False):
    if airports :
        add = ['ADES', 'ADEP']
    else :
        add = []
    if weight:
        df[obs+'_w'] = df[obs] * df['Weight'] #eventuellement le formuler en activité
    else :
        df[obs+'_w'] = df[obs]
    ac_types = df[['Aircraft Type','Aircraft Type Name', obs+'_w']].groupby(['Aircraft Type','Aircraft Type Name']).sum().reset_index().sort_values(by=obs+'_w',ascending=False)
    selec = ac_types[:n_ac]
    n_y = int((df['Period'].max() - df['Period'].min()))+1
    print(str(int(1000*selec[obs+'_w'].sum()/ac_types[obs+'_w'].sum())/10)+'% '+obs+'s selected.')
    df_f1 = df[df['Aircraft Type'].isin(selec['Aircraft Type'])]
    l_dist = []
    l_seats = []
    l_w = []
    for ac, ac_name, w_obs in zip(selec['Aircraft Type'],selec['Aircraft Type Name'], selec[obs+'_w']):
        mask = df_f1['Aircraft Type'] == ac
        d_0 = df_f1.loc[mask, 'Distance_conn (km)'].max()
        s = vis_ref.weighted_quantile(
            df_f1.loc[mask, 'Distance_conn (km)'].to_numpy(),
            df_f1.loc[mask, obs+'_w'].to_numpy(),
            1 - seuils_f)
        l_dist.append([ac, d_0, s, ac_name])
        df_f1 = df_f1[~(mask & (df_f1['Distance_conn (km)'] > s))]
        if save_techn_carac:
            l_seats.append((df_f1.loc[mask,obs+'_w']*df_f1.loc[mask,'Seats']/df_f1.loc[mask,'N_flights']).sum()/(df_f1.loc[mask,obs+'_w'].sum()))
            l_w.append(w_obs)
    #identification of years and types for the regression
    l_dist = np.array(l_dist)
    unique_ids = l_dist[:, 0]
    unique_names = l_dist[:, 3]
    max_dist = l_dist[:, 2].astype(float)
    period_min = df_f1['Period'].min()
    corr_table = pd.DataFrame({'Aircraft Type': unique_ids, 'Aircraft Type Name': unique_names, 'Id_mod': range(len(unique_ids))})
    df_f2 = df_f1.merge(corr_table, on=['Aircraft Type', 'Aircraft Type Name'], how='left').drop(columns=['Aircraft Type', 'Aircraft Type Name'])
    df_f2['Period'] = df_f2['Period'] - period_min
    if save_techn_carac:
        df_tech = pd.DataFrame({'Aircraft Type': unique_ids,'Aircraft Type Name': unique_names, obs+'_weight': l_w,'Est_avg_seats': l_seats,  'Est_max_range': max_dist})
        df_tech.to_excel('data//assignment_tech//BTS_'+obs+'_'+str(n_ac)+'_'+str(n_y)+'.xlsx')
    #volumes modeled per connection (used for validation)

    obs_conn_p = (
        df_f2.groupby(add+['Period','Distance_conn (km)', 'Seats_conn_p',obs+'_conn_p'])[obs+'_w']
        .sum()
        .reset_index()
        .rename(columns={obs+'_w': obs+'_w_mod'})
    )
    conn = df_f2.drop_duplicates(subset=add+['Period', 'Distance_conn (km)', 'Seats_conn_p', obs+'_conn_p','Weight'])[
        add+['Period', 'Distance_conn (km)', 'Seats_conn_p', obs+'_conn_p','Weight']]

    conn = pd.merge(conn,obs_conn_p,
    on=add+['Period','Distance_conn (km)', 'Seats_conn_p',obs+'_conn_p'],
    how='left')
    #aircraft presence cache for the regression
    contingency_table = pd.crosstab(df_f2['Id_mod'], df_f2['Period'])
    #deux lignes suivantes pour éviter des problèmes quand on saute le covid)
    all_periods = range(df_f2['Period'].min(),
                        df_f2['Period'].max() + 1)

    contingency_table = contingency_table.reindex(
        columns=all_periods,
        fill_value=0
    )
    binary_table = (contingency_table > 0).astype(int)
    aircraft_existence_cache = np.array(binary_table)
    return df_f2[add+['Period', 'Distance_conn (km)', 'Seats_conn_p', obs+'_conn_p', 'Id_mod', obs + '_w']], corr_table, aircraft_existence_cache, max_dist, conn

def log_likelihood_1_0(data_batch, cache, alphas, betas, omegas, ranges):
    us = (alphas[:, None] * data_batch[:, 1] + betas[:, None] * data_batch[:, 2]) + omegas[
        :, data_batch[:, 0].astype(int)]
    pot = (ranges[:, None] >= data_batch[:, 2]) * cache[:, data_batch[:, 0].astype(int)] * np.exp(
        us)
    log_lik = np.sum(
        data_batch[:, 4] * (us[data_batch[:, 3].astype(int),np.arange(len(data_batch))] - np.log(pot.sum(axis=0))))
    return log_lik

def gradient_1_0_fast(data_batch, cache, alphas, betas, omegas, ranges):
    # unpack
    y = data_batch[:, 0].astype(int)
    x1 = data_batch[:, 1]
    x2 = data_batch[:, 2]
    choice = data_batch[:, 3].astype(int)
    w = data_batch[:, 4]
    # utilities
    us = (
        alphas[None, :] * x1[:, None]
        + betas[None, :] * x2[:, None]
        + omegas[:, y].T
    )
    pot = (
        (ranges[None, :] >= x2[:, None])
        * cache[:, y].T
        * np.exp(us)
    )
    pot_sum = pot.sum(axis=1, keepdims=True)
    pot_norm = pot / pot_sum
    # ===== gradients alphas / betas =====
    grad_alphas = -np.sum(w[:, None] * pot_norm * x1[:, None], axis=0)
    grad_betas  = -np.sum(w[:, None] * pot_norm * x2[:, None], axis=0)
    np.add.at(grad_alphas, choice, w * x1)
    np.add.at(grad_betas,  choice, w * x2)
    # ===== gradient omegas =====
    num_indices, num_years = omegas.shape
    grad_omegas = np.zeros_like(omegas)
    # positive term
    np.add.at(grad_omegas, (choice, y), w)
    # negative term
    np.add.at(
        grad_omegas,
        (np.repeat(np.arange(num_indices), len(y)),
         np.tile(y, num_indices)),
        - (pot_norm.T * w).ravel()
    )
    return grad_alphas, grad_betas, grad_omegas

def fit_function_adam(training_data, existence_cache, ranges, num_epochs = 100, n_batches = 30, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
                 learning_rate = 0.02, alphas = None, betas = None, omegas = None, n_ac = 60, n_y = 35, names_ac = None, title = 'test' ,obs_norm2 = 1, y_min= 1990):
    batch_size = training_data.shape[0]//n_batches
    if alphas is None :
        alphas = np.zeros(n_ac)
        betas = np.zeros(n_ac)
        omegas = np.zeros((n_ac, n_y)) #normal version
    m_alphas, v_alphas = np.zeros_like(alphas), np.zeros_like(alphas)
    m_betas, v_betas = np.zeros_like(betas), np.zeros_like(betas)
    m_omegas, v_omegas = np.zeros_like(omegas), np.zeros_like(omegas)
    likely = []
    likely.append(int(100000 *
                      log_likelihood_1_0(training_data, existence_cache, alphas, betas, omegas,
                                                               ranges)/obs_norm2) / 100000)
    print(f"Without training {likely[-1]}")
    # Boucle d'entraînement
    for epoch in range(num_epochs):
        perm = np.random.permutation(len(training_data))  # Permuter les données pour chaque epoch
        print('epoch ' + str(epoch) + ';', end=' ')
        for i in range(n_batches):
            batch_idx = perm[i * batch_size: (i + 1) * batch_size]
            batch = training_data[batch_idx]
            grad_alphas, grad_betas, grad_omegas = gradient_1_0_fast(batch, existence_cache,
                                                                                           alphas, betas, omegas,
                                                                                           ranges)  # (version pimpée chatgpt)

            m_alphas = beta1 * m_alphas + (1 - beta1) * grad_alphas
            v_alphas = beta2 * v_alphas + (1 - beta2) * (grad_alphas ** 2)
            m_alphas_hat = m_alphas / (1 - beta1 ** (epoch * n_batches + i + 1))
            v_alphas_hat = v_alphas / (1 - beta2 ** (epoch * n_batches + i + 1))

            alphas += learning_rate * m_alphas_hat / (np.sqrt(v_alphas_hat) + epsilon)
            m_betas = beta1 * m_betas + (1 - beta1) * grad_betas
            v_betas = beta2 * v_betas + (1 - beta2) * (grad_betas ** 2)
            m_betas_hat = m_betas / (1 - beta1 ** (epoch * n_batches + i + 1))
            v_betas_hat = v_betas / (1 - beta2 ** (epoch * n_batches + i + 1))
            betas += learning_rate * m_betas_hat / (np.sqrt(v_betas_hat) + epsilon)
            m_omegas = beta1 * m_omegas + (1 - beta1) * grad_omegas
            v_omegas = beta2 * v_omegas + (1 - beta2) * (grad_omegas ** 2)
            m_omegas_hat = m_omegas / (1 - beta1 ** (epoch * n_batches + i + 1))
            v_omegas_hat = v_omegas / (1 - beta2 ** (epoch * n_batches + i + 1))
            omegas += learning_rate * m_omegas_hat / (np.sqrt(v_omegas_hat) + epsilon)
        likely.append(int(100000 *
                          log_likelihood_1_0(training_data, existence_cache, alphas, betas,
                                                                   omegas, ranges)/obs_norm2) / 100000)
        if epoch % 5 == 0:
            print(f"Log-likelihood = {likely[-1]}")
        if epoch % 50 == 49:
            save_estim(alphas, betas, omegas, names_ac,
                        name=title +'_' + str(epoch // 50), y_min= y_min)
    return alphas, betas, omegas, likely



def fit_function_bfgs(training_data, existence_cache, ranges,
                 num_epochs=100, alphas=None, betas=None, omegas=None,
                 n_ac=60, n_y=35, names_ac=None, title='test', obs_norm2=1, y_min= 1990):

    if alphas is None:
        alphas = np.zeros(n_ac)
        betas = np.zeros(n_ac)
        omegas = np.zeros((n_ac, n_y))

    # --- packing / unpacking ---
    def pack(a, b, o):
        return np.concatenate([a, b, o.ravel()])

    def unpack(theta):
        a = theta[:n_ac]
        b = theta[n_ac:2*n_ac]
        o = theta[2*n_ac:].reshape((n_ac, n_y))
        return a, b, o

    theta0 = pack(alphas, betas, omegas)
    likely = []
    ll0 = log_likelihood_1_0(training_data, existence_cache,
                             alphas, betas, omegas, ranges) / obs_norm2
    likely.append(round(ll0,5))
    print(f"Without training {likely[-1]}")
    def objective(theta):

        a,b,o = unpack(theta)

        ll = log_likelihood_1_0(
            training_data,
            existence_cache,
            a,b,o,
            ranges
        )

        return -ll/obs_norm2

    def gradient(theta):

        a,b,o = unpack(theta)

        grad_a, grad_b, grad_o = gradient_1_0_fast(
            training_data,
            existence_cache,
            a,b,o,
            ranges
        )

        g = pack(grad_a, grad_b, grad_o)

        return -g/obs_norm2

    cb_count = 0
    def callback(theta):
        nonlocal cb_count
        cb_count+=1
        if cb_count % 5 == 0:
            a, b, o = unpack(theta)

            ll = log_likelihood_1_0(
                training_data,
                existence_cache,
                a, b, o,
                ranges
            ) / obs_norm2
            likely.append(round(ll, 5))
            print(f"Log-likelihood = {likely[-1]}")
        if cb_count % 50 == 0:
            save_estim(alphas, betas, omegas, names_ac,
                       name=title + '_BFGS_'+str(cb_count//50))

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=gradient,
        callback=callback,
        options={
            "maxiter": num_epochs,
            "disp": True,
            "gtol": 1e-8,
        }
    )

    alphas, betas, omegas = unpack(result.x)
    llf = log_likelihood_1_0(training_data, existence_cache,
                             alphas, betas, omegas, ranges) / obs_norm2
    print(f"After training {llf}")
    save_estim(alphas, betas, omegas, names_ac,
               name=title + '_BFGS', y_min=y_min)

    return alphas, betas, omegas, likely


def save_estim(alphas, betas, omegas, name_c,y_min = 1990, name ='logit_model_test'):
    save1 = pd.DataFrame(alphas.reshape((1, omegas.shape[0])), columns=name_c)
    save2 = pd.DataFrame(betas.reshape((1, omegas.shape[0])), columns=name_c)
    save3 = pd.DataFrame(omegas.T, columns=name_c, index=np.arange(y_min, y_min +omegas.shape[1]))
    excel_path = 'data//assignment_coeff//' + name + '.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        save1.to_excel(writer, sheet_name='Alphas', index=True)
        save2.to_excel(writer, sheet_name='Betas', index=True)
        save3.to_excel(writer, sheet_name='Omegas', index=True)

def load_estim(name ='logit_model_test', columns = False):
    excel_path = 'data//assignment_coeff//' + name + '.xlsx'

    save1 = pd.read_excel(excel_path, sheet_name='Alphas', index_col=0)
    save2 = pd.read_excel(excel_path, sheet_name='Betas', index_col=0)
    save3 = pd.read_excel(excel_path, sheet_name='Omegas', index_col=0)

    alphas = save1.to_numpy().reshape(-1)
    betas = save2.to_numpy().reshape(-1)
    omegas = save3.to_numpy().T

    if columns :
        name_c = save1.columns.to_list()
        return alphas, betas, omegas, name_c

    else :
        return alphas, betas, omegas

def correspondance_table(title_coeffs = 'logit_model_bts_35_5_1_9',title_tech = 'BTS_35_4', obs_choice = 'ASK', d_norm = 15000, cap_norm = 15):
    excel_path_coeffs = 'data//assignment_coeff//' + title_coeffs + '.xlsx'
    excel_path_tech = 'data//assignment_tech//' + title_tech + '.xlsx'

    save1 = pd.read_excel(excel_path_coeffs, sheet_name='Alphas', index_col=0).transpose()
    save2 = pd.read_excel(excel_path_coeffs, sheet_name='Betas', index_col=0).transpose()
    coeffs = pd.merge(save1, save2, left_index=True, right_index=True).rename(columns={'0_x': 'alphas', '0_y': 'betas'})
    coeffs['alphas']=coeffs['alphas']/d_norm
    coeffs['betas']=coeffs['betas']/cap_norm
    tech_table = pd.read_excel(excel_path_tech, index_col=0)[
        ['Aircraft Type Name', obs_choice + '_weight', 'Est_avg_seats', 'Est_max_range']].set_index(
        'Aircraft Type Name')
    tech_table['Aircraft Type Name'] = tech_table.index
    data_table = pd.merge(tech_table, coeffs, left_index=True, right_index=True)
    # p_min = int(np.log10(data_table[obs_choice + '_weight'].min())) #Pour normaliser, ici on l'enlève pour avoir une estimation réaliste pour le nombre d'avion neufs équivalents.
    # data_table[obs_choice + '_weight'] = data_table[obs_choice + '_weight'] / 10 ** p_min
    return data_table

def predict_assignt(conn_data, existence_cache, alphas, betas, omegas, ranges, d_norm = 15000, cap_norm = 15, single_p = False):
    if (d_norm == 1) and (cap_norm == 1):
        distance = conn_data[:, 1]
        capacity = np.log(conn_data[:, 2])
        ranges_b = ranges
    else :
        distance = conn_data[:, 1] / d_norm
        capacity = np.log(conn_data[:, 2]) / cap_norm
        ranges_b = ranges/d_norm

    if single_p :
        us = alphas[:, None] * distance + betas[:, None] * capacity + omegas[
            :, 0][:, None]
        valid = (ranges_b[:, None] >= distance[None, :]) & (
            existence_cache[:, 0][:, None])
        us = np.where(valid, us, -np.inf)
    else :
        ids = conn_data[:, 0]
        if ids.dtype != int:
            ids = ids.astype(int)
        us = alphas[:, None] * distance + betas[:, None] * capacity + omegas[
            :, ids]
        valid = (ranges_b[:, None] >= distance[None, :]) & (
        existence_cache[:, ids])
        us = np.where(valid, us, -np.inf)
    us_max = np.max(us, axis=0, keepdims=True)
    exp_us = np.exp(us - us_max)
    denom = exp_us.sum(axis=0, keepdims=True)
    pots = exp_us / np.maximum(denom, 1e-12)
    return pots

def obs_comp(training_data, conn_data, pots, obs_norm,obs_norm2, save = False, title = None):
    e = 0
    e_p = 0
    c = 0
    m = 0
    plt.figure(figsize=(6, 6))
    sns.set_theme(style='whitegrid')
    y_min = conn_data[:, 0].min().astype(int)
    y_max = conn_data[:, 0].max().astype(int)
    n_ac = pots.shape[0]
    for t in range(y_max - y_min + 1):
        for ac in range(n_ac):
            obs_cum_mod = (pots[ac, (conn_data[:, 0] == t)] * conn_data[(conn_data[:, 0] == t)][:, 3]).sum()
            obs_real = obs_norm* training_data[(training_data[:, 3] == ac) & (training_data[:, 0] == t), 4].sum()
            if obs_cum_mod != 0:
                e += np.abs(obs_real - obs_cum_mod) / obs_real
                e_p += np.abs(obs_real - obs_cum_mod)
                if np.abs(obs_real - obs_cum_mod) / obs_real > 0.25:
                    print(t, ac,obs_real, obs_cum_mod)
                c += 1
                m = max(m, obs_real, obs_cum_mod)
                if t == 0 and ac == 1:
                    plt.scatter(obs_real, obs_cum_mod, color='black', s=15, marker='+', linewidths=0.5,
                                label='Total obs for each aircraft \n type, each year', zorder=2)
                else:
                    plt.scatter(obs_real, obs_cum_mod, color='black', s=15, marker='+', linewidths=0.5, zorder=2)
    print('avg relative error : ' + str(e / c))
    print('obs_weighted avg relative error : ' + str(e_p /obs_norm/obs_norm2))
    plt.plot([0, m], [0, m], linestyle='--', linewidth=2, label='y = x', zorder=1)
    if save:
        if title is None:
            title = 'obs_comp'
        plt.savefig('figures//assignment_figure//'+title + '.pdf', dpi=300)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xmin=m/1000, xmax=m)
    plt.ylim(ymin=m/1000, ymax=m)
    plt.show()
    return None

def nwsd(aircraft_existence, training_data, conn_data, pots, d_norm, cap_norm, obs_norm, save = False, title = None, year_ref = 1990, freq = 1):
    y_min = conn_data[:, 0].min().astype(int)
    y_max = conn_data[:, 0].max().astype(int)
    n_ac = pots.shape[0]

    dist_moy_d = np.zeros((y_max - y_min + 1))
    dist_moy_i = np.zeros((y_max - y_min + 1))
    obs = np.zeros((y_max - y_min + 1))

    for ac in range(n_ac):
        extract_ac = training_data[(training_data[:, 3] == ac)]
        for t in range(y_max - y_min + 1):
            if aircraft_existence[ac, t] != 0:
                mask_conn = conn_data[:, 0] == t
                obs_cum_mod = (pots[ac, mask_conn] * conn_data[mask_conn][:, 3])
                result_mod = np.array((conn_data[mask_conn][:, 1], conn_data[mask_conn][:, 2], obs_cum_mod)).T
                result_real = extract_ac[(extract_ac[:, 0] == t)][:, [1, 2, 4]]
                result_real[:,0]= d_norm * result_real[:,0]
                result_real[:,1] = np.exp(result_real[:,1]*cap_norm)
                result_real[:,2] = obs_norm* result_real[:,2]

                distance_moy = (result_real[:, 2] * result_real[:, 0]).sum() / (result_real[:, 2].sum())
                int_moy = (result_real[:, 2] * result_real[:, 1]).sum() / (result_real[:, 2].sum())
                obs_ac_t = result_real[:, 2].sum()
                result_real[:, 2] = result_real[:, 2] / result_real[:, 2].sum()

                # Extraction des valeurs pour chaque dimension et les probabilités
                A_dist, A_int, A_prob = result_real[:, 0], result_real[:, 1], result_real[:, 2]
                B_dist, B_int, B_prob = result_mod[:, 0], result_mod[:, 1], result_mod[:, 2]
                # Calcul de la distance de Wasserstein pour chaque dimension
                wasserstein_x = wasserstein_distance(A_int, B_int, u_weights=A_prob, v_weights=B_prob)
                wasserstein_y = wasserstein_distance(A_dist, B_dist, u_weights=A_prob, v_weights=B_prob)
                dist_moy_i[t] += obs_ac_t * wasserstein_x / int_moy
                dist_moy_d[t] += obs_ac_t * wasserstein_y / distance_moy
                obs[t] += obs_ac_t
    wass_tot_D = dist_moy_d.sum() / obs.sum()
    wass_tot_I = dist_moy_i.sum() / obs.sum()

    print(wass_tot_D, wass_tot_I)
    dist_moy_D = dist_moy_d / obs
    dist_moy_I = dist_moy_i / obs

    plt.plot(np.arange(year_ref+y_min/freq, year_ref+(y_max + 1)/freq, 1/freq), dist_moy_D, label='Distance', color='orange', marker='o')
    plt.plot(np.arange(year_ref+y_min/freq, year_ref+(y_max + 1)/freq, 1/freq), dist_moy_I, label='Capacity of the route (seats/year) ', color='green',
             marker='o')
    plt.xlabel('Year')
    plt.ylabel('NWSD (Data, Model)')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
    plt.ylim((0, 0.2))
    plt.grid(linestyle='--')
    # plt.legend(bbox_to_anchor = (1,1))
    plt.legend(framealpha=1)
    if save:
        if title is None:
            title = 'obs_NWSD'
        plt.savefig('figures//assignment_figure//'+title + '.pdf', dpi=300)
    plt.show()
    return(None)

def visu_coeffs(alphas, betas, names_ac,ds_name, d_norm = 1, cap_norm = 1):
    plt.figure(figsize=(5.5, 5.5))
    plt.grid(True, linestyle='--', linewidth=0.3, color='gray')
    n_ac = len(names_ac)
    n_colors = min(10,-int(np.floor(-n_ac/6)))
    n_col = (n_ac-1)//19+1
    colors = vis_ref.colors_10[:n_colors]
    df_visu = pd.DataFrame({'alphas': alphas, 'betas': betas}, index=names_ac).sort_index()
    alphas = np.array(df_visu['alphas'])
    betas = np.array(df_visu['betas'])
    names_ac = df_visu.index.to_list()
    for i in range(min(len(names_ac),59)):
        plt.scatter(alphas[i]/d_norm,betas[i]/cap_norm, s=1.5*vis_ref.sizes[i // n_colors], color=colors[i % n_colors], marker= vis_ref.marker_type[i // n_colors],
                    label= names_ac[i][:30], edgecolors='black', linewidth=0.5)
    plt.legend(markerscale=1.3, scatterpoints=1, ncol = n_col, bbox_to_anchor=(1, 1))
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.gca().xaxis.get_major_formatter().set_scientific(True)
    plt.gca().xaxis.get_major_formatter().set_powerlimits((-1, 1))
    plt.xlabel('Impact of distance', fontsize=14)
    plt.ylabel('Impact of log(capacity)', fontsize=14)
    plt.savefig('figures//estimators_figures//estim_model_' +ds_name+'.pdf', bbox_inches="tight", format='pdf')
    plt.show()

def regressor_assign_coeffs(c_table, weight = False, weight_log = False, obs_choice = 'ASK', visu = True):
    x_0 = c_table[['Est_max_range','Est_avg_seats']]
    x = np.concatenate((x_0, np.log(x_0), 1 / x_0), axis=1)


    y1 = c_table['alphas'] #variable with heteroscedasticity. Problématic in our case. We try here to weight with the log (optionnal)
    y2 = c_table['betas']
    if weight :
        p_max = int(np.log10(c_table[obs_choice + '_weight'].max()))
        weight_c2 = c_table[obs_choice + '_weight']/10**(p_max-2)
        if weight_log :
            d_min=c_table['Est_max_range'].min()
            weight_c1 = weight_c2*np.log1p(c_table['Est_max_range'].values/d_min)
        else :
            weight_c1 = weight_c2
    else :
        weight_c2 = 25*np.ones(len(y1))
        if weight_log :
            d_min=c_table['Est_max_range'].min()
            weight_c1 = weight_c2*np.log1p(c_table['Est_max_range'].values/d_min)
        else :
            weight_c1 = weight_c2

    model1 = LinearRegression()
    model1.fit(x, y1, sample_weight=weight_c1)

    if visu :
        y_pred = model1.predict(x)
        mse = mean_squared_error(y1, y_pred, sample_weight=weight_c1)
        r2 = r2_score(y1, y_pred, sample_weight=weight_c1)
        print(f"MSE : {10 ** 8 * mse:.3f}e-8")
        print(f"R^2 : {r2:.3f}")
        plt.scatter(y_pred, y1, color='blue', alpha=0.8, label='Aircraft type', s=weight_c1)
        plt.plot([min(y1), max(y1)], [min(y1), max(y1)], color='red', linestyle='--', label='y = x')
        plt.ylabel(r'$\alpha_j$: Real assignment parameter', fontsize=16)
        plt.xlabel(r'$\hat{\alpha_j}$: Predicted assignment parameter', fontsize=16)
        plt.legend(fontsize=18, framealpha=1)
        plt.grid(True)
        plt.show()

    model2 = LinearRegression()
    model2.fit(x, y2, sample_weight=weight_c2)

    if visu :
        y_pred = model2.predict(x)
        mse = mean_squared_error(y2, y_pred, sample_weight=weight_c2)
        r2 = r2_score(y2, y_pred, sample_weight=weight_c2)
        print(f"MSE : {10 * mse:.3f}e-1")
        print(f"R^2 : {r2:.3f}")
        plt.scatter(y_pred, y2, color='blue', alpha=0.8, label='Aircraft type', s=weight_c2)
        plt.plot([min(y2), max(y2)], [min(y2), max(y2)], color='red', linestyle='--', label='y = x')
        plt.ylabel(r'$\beta_j$: Real assignment parameter', fontsize=16)
        plt.xlabel(r'$\hat{\beta_j}$: Predicted assignment parameter', fontsize=16)
        plt.legend(fontsize=18, framealpha=1)
        plt.grid(True)
        plt.show()
    return np.concatenate([np.array([model1.intercept_]),model1.coef_]), np.concatenate([np.array([model2.intercept_]),model2.coef_])

def regressor_visu_model(c_table, alphas_coeff, betas_coeff, reg_name='test'):
    y1 = c_table['alphas']
    y2 = c_table['betas']
    names_ac = c_table['Aircraft Type Name'].to_list()
    plt.figure(figsize=(7.5, 5.5))
    plt.grid(True, linestyle='--', linewidth=0.3, color='gray')
    p_d = 20
    p_s = 3
    u = 0
    for sqrt_range in np.linspace(1000.05 ** (1 / p_d), 18000.5 ** (1 / p_d), 19):
        ranges = 50 * int(sqrt_range ** p_d / 50)
        inputs = np.array([ranges * np.ones(700), np.linspace(40.5 ** (1 / p_s), 700.5 ** (1 / p_s), 700) ** p_s]).T
        sample_x = np.concatenate((inputs, np.log(inputs), 1 / inputs), axis=1)
        alphas_sample = alphas_coeff[0]+ (alphas_coeff[1:][:, None] * sample_x.T).sum(axis =0)
        betas_sample = betas_coeff[0]+ (betas_coeff[1:][:, None] * sample_x.T).sum(axis =0)
        if u % 2 == 0:
            plt.plot(alphas_sample, betas_sample, color='b', linewidth=0.5, linestyle='--', alpha=1)
        # else :
        #   plt.plot(alphas_sample, betas_sample, color = 'b', linewidth = 0.5, linestyle = '--', alpha = 0.2)
        u += 1
    u = 0
    for cap in np.linspace(40.5 ** (1 / p_s), 700.5 ** (1 / p_s), 19):
        cap = 5 * int(cap ** p_s / 5)
        inputs = np.array([np.linspace(1000.05**(1/p_d),18000.5**(1/p_d),700)**p_d,cap*np.ones(700)]).T
        sample_x = np.concatenate((inputs, np.log(inputs), 1 / inputs), axis=1)
        alphas_sample = alphas_coeff[0] + alphas_coeff[1:] @ sample_x.T
        betas_sample = betas_coeff[0] + betas_coeff[1:] @ sample_x.T
        if u % 2 == 0:
            plt.plot(alphas_sample, betas_sample, color='r', linewidth=0.5, linestyle='--', alpha=1)
        # else :
        #   plt.plot(alphas_sample, betas_sample, color = 'r', linewidth = 0.5, linestyle = '--', alpha = 0.2)
        u += 1
    ann1 = np.array(
        [50 * np.trunc(np.linspace(1000.05 ** (1 / p_d), 18000.5 ** (1 / p_d), 10) ** p_d / 50), 40 * np.ones(10)]).T
    ann2 = np.concatenate((ann1, np.log(ann1), 1 / ann1), axis=1)
    alphas_sample = alphas_coeff[0] + alphas_coeff[1:] @ ann2.T
    betas_sample = betas_coeff[0] + betas_coeff[1:] @ ann2.T
    for i, label in enumerate(np.linspace(1000.05 ** (1 / p_d), 18000.5 ** (1 / p_d), 10) ** p_d):
        plt.scatter(alphas_sample[i], betas_sample[i], color="darkblue", zorder=3, s=5, marker='s')
        plt.annotate(
            str(50 * int(label / 50+0.5)),  # Le label à afficher
            (alphas_sample[i], betas_sample[i]),  # Position (valeur réelle, prédiction)
            textcoords="offset points",  # Décalage par rapport à la position
            xytext=(-10, -12),  # Décalage en pixels (x, y)
            fontsize=8,  # Taille de la police
            color="darkblue"  # Couleur des annotations
        )
    ann3 = np.array([1000*np.ones(10),5*np.trunc(np.linspace(40.5**(1/p_s),700.5**(1/p_s),10)**p_s/5)]).T
    ann4 = np.concatenate((ann3, np.log(ann3), 1 / ann3), axis=1)
    alphas_sample = alphas_coeff[0] + alphas_coeff[1:] @ ann4.T
    betas_sample = betas_coeff[0] + betas_coeff[1:] @ ann4.T
    for i, label in enumerate(np.linspace(40.5 ** (1 / p_s), 700.5 ** (1 / p_s), 10) ** p_s):
        plt.scatter(alphas_sample[i], betas_sample[i], color="darkred", zorder=3, s=5, marker='s')
        plt.annotate(
            str(5 * int(label / 5+0.5)),  # Le label à afficher
            (alphas_sample[i], betas_sample[i]),  # Position (valeur réelle, prédiction)
            textcoords="offset points",  # Décalage par rapport à la position
            xytext=(-18, -3),  # Décalage en pixels (x, y)
            fontsize=8,  # Taille de la police
            color="darkred"  # Couleur des annotations
        )
    plt.annotate(
        'MAX RANGE (km)',  # Le label à afficher
        (-0.01, -5.5),  # Position (valeur réelle, prédiction)
        textcoords="offset points",  # Décalage par rapport à la position
        xytext=(0, 0),  # Décalage en pixels (x, y)
        fontsize=12,  # Taille de la police
        color="darkblue"  # Couleur des annotations
    )
    plt.annotate(
        'AVG SEATS',  # Le label à afficher
        (-0.015, -3.5),  # Position (valeur réelle, prédiction)
        textcoords="offset points",  # Décalage par rapport à la position
        xytext=(0, 0),  # Décalage en pixels (x, y)
        fontsize=12,  # Taille de la police
        color="darkred"  # Couleur des annotations
    )

    n_ac = len(names_ac)
    n_colors = min(10,-int(np.floor(-n_ac/6)))
    n_col = (n_ac - 1) // 19 + 1
    colors = vis_ref.colors_10[:n_colors]
    for i in range(min(len(names_ac),59)):
        plt.scatter(y1[i], y2[i], s=1.5 * vis_ref.sizes[i // n_colors],
                    color=colors[i % n_colors], marker=vis_ref.marker_type[i // n_colors],
                    label=names_ac[i][:30], edgecolors='black', linewidth=0.5)
    plt.legend(markerscale=1.3, scatterpoints=1, ncol=n_col, bbox_to_anchor=(1, 1))
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.gca().xaxis.get_major_formatter().set_scientific(True)
    plt.gca().xaxis.get_major_formatter().set_powerlimits((-1, 1))
    plt.xlabel('Impact of distance', fontsize=14)
    plt.ylabel('Impact of log(capacity)', fontsize=14)
    plt.savefig('figures//estimators_figures//visu_regressor_' + reg_name + '.pdf', bbox_inches="tight", format='pdf')
    plt.show()

def assign_coeffs_pred(alphas_coeff, betas_coeff,seats_ac, range_ac):
    x = np.array([range_ac, seats_ac,np.log(range_ac), np.log(seats_ac), 1 / range_ac, 1 / seats_ac])
    return alphas_coeff[0]+ (x*alphas_coeff[1:]).sum(), betas_coeff[0]+ (x*betas_coeff[1:]).sum(),seats_ac, range_ac