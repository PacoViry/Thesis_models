import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# import time as tm
import os
from PIL import Image
import glob
import imageio.v2 as imageio
import importlib
from src.C_Modelling import fleet_content_modelling
from src.C_Modelling import assignment_conv
from src.C_Modelling import productivity_modelling
from src.B_Model_fitting import assignment_multilogit as am
from src.A_Data_analysis import assignment_analysis
from src.A_Data_analysis import fleet_analysis
from src.Common_tools import vis_ref


importlib.reload(fleet_content_modelling)
# importlib.reload(assignment_analysis)
# importlib.reload(productivity_modelling)
importlib.reload(assignment_conv)


def fuel_calculations(df, seats_c, dic_fuel, fb_vals, n_flights = True): #gives Seymmour estimation of the fuel consumption in kg (maybe a corrective factor to include for gcd vs real distance)
    indices_s = df['Aircraft Type Name'].map(dic_fuel)
    if n_flights :
        df['N_flights'] = df['Seats']/ np.array(seats_c)[df['Aircraft Type']]
    distances = df['Distance_conn (km)'].values
    coeffs = fb_vals[indices_s]
    fb = coeffs[:, 0] * distances ** 2 + coeffs[:, 1] * distances + coeffs[:, 2]
    df['FB'] = fb*df['N_flights']
    return None


def compute_scenario(initial_fleet, deliveries, obs_sizes, traffic_structures, retirement_propensions, alphas, betas, omegas_0,
             ranges, aging_activity_coeff=0, taux_utilisation_usuel=1, period_duration=1, precision = 1e-6):
    n_y = deliveries.shape[0]
    tot_obs = traffic_structures[:, :, 5].sum(axis=1)
    y_s = initial_fleet.shape[0] - deliveries.shape[0]

    fleet_pot_t = initial_fleet.copy()
    x_eq = 1
    omegas_t = omegas_0.copy()

    vol_obs = [] #on commence à vide, car la première étape consiste à supposer aucune livraison
    vol_act = []
    age_array = np.concatenate([np.arange(y_s), y_s + period_duration * np.arange(n_y)])

    print('Computing fleet composition, '+str(n_y)+' periods:', end =' ')
    for t in range(n_y):
        print('' + str(int(1000*t/n_y+0.5)/10), end='% ')
        fleet_pot_t[-(n_y - t), :] += deliveries[t, 0, :]
        fleet_obs_t = fleet_pot_t * obs_sizes[None, :] * taux_utilisation_usuel * \
                      np.exp(aging_activity_coeff * (-y_s + 1 - t * period_duration + age_array))[:, None]
        # print('total fleet: ' + str(fleet_obs_t.sum()))
        # print('total constraint: ' + str(tot_obs[t]))
        fleet_obs_act_t, x_eq = fleet_content_modelling.fleet_content(0, x_eq, fleet_obs_t, retirement_propensions,
                                                                      tot_obs[t], epsilon=precision, first=True)
        if x_eq < 1e-30: #numerical stability
            x_eq = x_eq ** (1 / 8)
            retirement_propensions = retirement_propensions - 3 * np.log(2)
        fleet_real_seats_t = fleet_obs_act_t / taux_utilisation_usuel * \
                             np.exp(-aging_activity_coeff * (-y_s + 1 - t * period_duration + age_array))[:, None]
        vol_obs.append(fleet_real_seats_t)
        vol_act.append(fleet_obs_act_t)

    omegas_l = []
    print('')
    print('Computing fleet assignment, ' + str(n_y) + ' periods:', end=' ')
    for t in range(n_y):
        print('' + str(int(1000 * t / n_y + 0.5) / 10), end='% ')
        # années ne sont pas importantes, eventuellement glisser en amont l'impact du vieillissement, en diminuant les quantités d'années en années.
        fleet_obs_const_t = vol_act[t].sum(axis=0)
        # lightening the datafile
        aircraft_quantity = fleet_obs_const_t.sum()
        u = -6
        fleet_obs_const_test = np.where(fleet_obs_const_t < aircraft_quantity *precision* (10 ** u), 0,
                                        fleet_obs_const_t)  # éviter effets de seuils chelous, a REVOIR
        existence_cache = (fleet_obs_const_test != 0)[:, None]
        while existence_cache.sum() < 16:
            u -= 1
            fleet_obs_const_test = np.where(fleet_obs_const_t < aircraft_quantity *precision* (10 ** u), 0,
                                            fleet_obs_const_t)  # éviter effets de seuils chelous, a REVOIR
            existence_cache = (fleet_obs_const_test != 0)[:, None]
            if u == -200:
                print('Too many aircraft were too much retired...')
                break
        fleet_obs_const_t = fleet_obs_const_test.copy()
        non_zero_indices = np.nonzero(existence_cache)[0]
        omegas_t = assignment_conv.conv_assignt(traffic_structures[t, :, 2:], existence_cache, alphas, betas, omegas_t,
                                                ranges, fleet_obs_const_t[non_zero_indices], non_zero_indices,
                                                epsilon=precision)
        omegas_l.append(omegas_t)
    print('Ok.')
    vol_obs = np.array(vol_obs)
    vol_act = np.array(vol_act)
    return vol_obs, vol_act, np.array(omegas_l)

def observe_scenario(alphas, betas,ranges, omegas, types_names, traffic_structures, period_duration=1, observation = 'Av_ac_seats',
                     obs_sizes = None, dic_fuel = None, vals_fuel = None,period_ref = None, save_name = 'scenario_test', reso=400):
    if period_ref is None :
        period_ref = omegas.shape[1] - 1
    if observation is None :
        print('Av_ac_seats')
        dynamic_pred_vis(traffic_structures, alphas, betas, ranges, omegas, list(range(omegas.shape[0])),
                         name_periods=None, types_names=types_names,
                         video_name=save_name+'_Av_ac_seats', format_v='video', color_mix=vis_ref.colors_26,
                         market='Aircraft Type', observation='Av_ac_seats', dist_limits=(4e2, 1.9e4),
                         capac_limits=(9e3, 4e6),period_duration=period_duration,
                         frame_s=2, period_ref=period_ref, reso=reso)
        print('ASK')
        dynamic_pred_vis(traffic_structures, alphas, betas, ranges, omegas, list(range(omegas.shape[0])),
                         name_periods=None, types_names=types_names, obs_sizes=obs_sizes,
                         video_name='test_sc_obs_market'+'_ASK', format_v='video', color_mix=vis_ref.colors_26,
                         market='Aircraft Type', observation='ASK', dist_limits=(4e2, 1.9e4),
                         capac_limits=(9e3, 4e6),period_duration=period_duration,
                         frame_s=2, period_ref=period_ref, reso=reso)
        print('FB')
        dynamic_pred_vis(traffic_structures, alphas, betas, ranges, omegas, list(range(omegas.shape[0])),
                         name_periods=None, types_names=types_names,obs_sizes=obs_sizes,
                         video_name='test_sc_obs_market'+'_FB', format_v='video', color_mix=vis_ref.colors_26,
                         market='Aircraft Type', observation='FB', dist_limits=(4e2, 1.9e4),
                         capac_limits=(9e3, 4e6), frame_s=2, period_ref=0,period_duration=period_duration,
                         dic_fuel = dic_fuel, vals_fuel = vals_fuel, reso=reso)
        return None
    elif observation not in ['Av_ac_seats', 'ASK', 'FB']:
        print('observation not recognized.')
        print('observation must be in [Av_ac_seats, ASK, FB].or N_flights or Seats if the code is updated.')
        return None
    dynamic_pred_vis(traffic_structures, alphas, betas, ranges, omegas, list(range(omegas.shape[0])),
                        name_periods=None, types_names=types_names,obs_sizes=obs_sizes,
                         video_name=save_name+'_'+observation, format_v='video', color_mix=vis_ref.colors_26,
                         market='Aircraft Type', observation=observation, dist_limits=(4e2, 1.9e4),
                         capac_limits=(9e3, 4e6), frame_s=2, period_ref=period_ref,period_duration=period_duration,
                         dic_fuel = dic_fuel, vals_fuel = vals_fuel, reso=reso)
    return None

def create_df(traff, alphas,betas, omegas_p, ranges,types_names):
    existence_cache = (omegas_p != 0)
    pots = am.predict_assignt(traff[:, 2:], existence_cache, alphas, betas, omegas_p, ranges, d_norm=1,
                              cap_norm=1, single_p=True)
    ass_mod = pots * traff[:, 2:][:, 3][None, :]
    non_zero_indices = np.nonzero(existence_cache[:,0])[0]
    assign_rep = np.tile(traff, (non_zero_indices.shape[0], 1))
    col7 = ass_mod[non_zero_indices, :].reshape(-1, 1)
    col8 = np.repeat(non_zero_indices, traff.shape[0]).reshape(-1, 1)
    share_filters = pots[non_zero_indices, :].reshape(-1, 1)
    assign_final = np.hstack([assign_rep, col7, col8])
    assign_final = assign_final[share_filters[:, 0] > 0, :]

    df_visu = pd.DataFrame(assign_final, columns=['ADES', 'ADEP', 'Period', 'Distance_conn (km)', 'Seats_conn_p',
                                                  'Av_ac_seats'+'_conn_p', 'Av_ac_seats', 'Aircraft Type'])
    df_visu = df_visu.astype({
        'ADES': 'int32',
        'ADEP': 'int32',
        'Period': 'int16',
        'Aircraft Type': 'int16',
        'Distance_conn (km)': 'float32',
        'Seats_conn_p': 'float32',
        'Av_ac_seats'+'_conn_p': 'float32',
        'Av_ac_seats': 'float32'
    })
    # l'attribution du modélisé à la connection est discutable, mais simplifie la vie, et ok si on modélise la grande majorité des modèles
    dico_names = dict(zip(range(len(types_names)), types_names))
    df_visu['Aircraft Type Name'] = df_visu['Aircraft Type'].map(dico_names)
    return df_visu

def dynamic_pred_vis(traff_structures, alphas, betas, ranges, omegas, periods, obs_sizes = None, types_names = None, name_periods = None,
                     video_name = 'test_video_market', format_v = 'gif', color_mix = vis_ref.colors_26,
                     market = 'Aircraft Type',observation = 'Av_ac_seats', dist_limits = (4e2, 1.9e4), capac_limits = (9e3, 5e6),
                     frame_s = 2, period_ref = None, n_market = 18,reso = 400, smooth_param = 0.04,
                     period_duration = 1,label_size_param = 1,dic_fuel = None, vals_fuel = None):
    if name_periods is None:
       name_periods = periods
    if types_names is None:
       types_names = list(range(omegas.shape[2]))
    if period_ref is None:
        period_ref = periods[-1]
        return None
    print('Color choice...')
    dimensions = (14, 10)
    width_grid, height_grid = 21, 8
    width_main, height_main = 17, 5
    smooth_x = (np.log(dist_limits[1] - np.log(dist_limits[0]))) * smooth_param / (
            dimensions[0] * width_main / width_grid)
    smooth_y = (np.log(capac_limits[1] - np.log(capac_limits[0]))) * smooth_param / (
            dimensions[
                1] * height_main / height_grid)  # seulement pour combler le lissage de la figure précédent, peut être un sujet à adapter

    # Ordonnancement des couleurs
    type_obs = []
    lists_c = []
    lists_r = []
    for period in periods:
        df_p =  create_df(traff_structures[period], alphas, betas, omegas[period], ranges, types_names)
        if observation == 'ASK':
            productivity_modelling.ASK_assignment(df_p, 'EC_preCOVID')
            df_p = assignment_analysis.agg_ind_s(df_p, observation='ASK')
        elif observation == 'FB':
            productivity_modelling.ASK_assignment(df_p, 'EC_preCOVID')
            fuel_calculations(df_p, obs_sizes, dic_fuel, vals_fuel)
            df_p = assignment_analysis.agg_ind_s(df_p, observation='FB')
        volumes = (
            df_p[['Aircraft Type', observation]]
            .groupby('Aircraft Type')
            .sum()
            .reindex(range(omegas.shape[1]), fill_value=0)
            .rename_axis('Aircraft Type')  # nom de l'index
            .reset_index()
            .sort_values(by='Aircraft Type').to_numpy()
        )
        type_obs.append(volumes[:, 1])
        group = df_p[[market,  observation]].groupby([market]).sum().sort_values(by=market, ascending=False)
        market_types = group.sort_values(by=observation, ascending=False).reset_index()
        selec = market_types[:n_market][market]
        selec_r = market_types[market]
        lists_r.append(list(selec_r))
        lists_c.append(list(selec))
    sol = vis_ref.color_lists(lists_c)
    if sol is not None:
        print("Coloriage trouvé :", sol)
    else:
        print("Pas de solution simple trouvée, attribution random")
        sol = dict(zip(lists_c, range(2,len(lists_c)+2)))
    #Ordonnancement des markets pour des transitions smooth entre avions ou opérateurs
    type_obs = np.array(type_obs)
    totals = type_obs.sum(axis=0)
    top_k = min(48, type_obs.shape[1])
    top_indices = np.argsort(-totals)[:top_k]

    # vecteur temps
    t = np.arange(type_obs.shape[0])

    # calcul de la date moyenne d'utilisation
    mean_date = {}
    for m in top_indices:
        if totals[m] > 0:
            mean_date[m] = (t * type_obs[:, m]).sum() / totals[m]
        else:
            mean_date[m] = -np.inf

    # tri décroissant
    sorted_top = sorted(
        top_indices,
        key=lambda m: -mean_date[m]
    )

    # ajouter les autres colonnes dans leur ordre
    others = [m for m in range(type_obs.shape[1]) if m not in sorted_top]
    types_sorted = list(sorted_top) + others

    # ranking final
    rank = {types_names[m]: k for k, m in enumerate(types_sorted)}
    rank2 = {m: k for k, m in enumerate(types_sorted)}

    # --- TRI SELON RANK ---
    ordered_indices = sorted(
        rank2.keys(),
        key=lambda i: rank2[i]
    )
    type_obs = type_obs[:, ordered_indices]
    obs_names = [types_names[i] for i in ordered_indices]

    n_bars = type_obs.shape[0]
    p_categories = type_obs.shape[1]

    x = np.arange(n_bars) * period_duration + 2024
    bottom = np.zeros(n_bars)
    top = np.zeros(n_bars)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.grid(axis='y', color='grey', linestyle='--')

    for i in range(p_categories):
        top = top + type_obs[:, i]
        if i < 24:
            ax.fill_between(x, bottom, top, label=obs_names[i],
                            color=color_mix[i], linewidth=0, alpha=0.9)
            bottom = top
        elif i < 48:
            ax.fill_between(x, bottom, top, label=obs_names[i],
                            color=color_mix[i - 24], linewidth=0, alpha=0.9, hatch='..', edgecolor='0.2')
            bottom = top
    mpl.rcParams['hatch.color'] = 'white'
    ax.fill_between(x, bottom, top, alpha=0.9, color='grey', linewidth=0, edgecolor='white', label='Others', hatch='//')
    ax.legend(framealpha=1, bbox_to_anchor=(1.05, 1),
              loc='upper left', borderaxespad=0., ncol=2)

    plt.xlim(x.min(), x.max())
    plt.ylim(0, 1.1 * top.max())
    plt.xlabel('Years')
    plt.ylabel('Quantity of ' + observation)
    plt.tight_layout()
    plt.savefig('figures/integrated_observation/scenario/' + video_name + '.pdf')
    plt.show()

    #Hauteurs de références
    df_p = create_df(traff_structures[period_ref], alphas, betas, omegas[period_ref], ranges, types_names)
    if observation == 'ASK':
        productivity_modelling.ASK_assignment(df_p, 'EC_preCOVID')
        df_p = assignment_analysis.agg_ind_s(df_p, observation='ASK')
    elif observation == 'FB':
        productivity_modelling.ASK_assignment(df_p, 'EC_preCOVID')
        fuel_calculations(df_p, obs_sizes, dic_fuel, vals_fuel)
        df_p = assignment_analysis.agg_ind_s(df_p, observation='FB')
    if observation!='Av_ac_seats':
        print('a remplir')
    #pas de conditions sur les weight qui sont deja intégrés
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
    #nettoyage des images existantes
    for filename in os.listdir('figures/video_storage/'):
        if filename.endswith(".png"):
            os.remove(os.path.join('figures/video_storage/', filename))


    #Créations des images
    for i in range(len(periods)):
        if i%3 == 0 :
            print(str(int(1000*i/len(periods)+0.5)/10)+'%', end =' ')
        df_p = create_df(traff_structures[periods[i]], alphas, betas, omegas[periods[i]], ranges, types_names)
        if observation == 'ASK':
            productivity_modelling.ASK_assignment(df_p, 'EC_preCOVID')
            df_p = assignment_analysis.agg_ind_s(df_p, observation='ASK')
        elif observation == 'FB':
            productivity_modelling.ASK_assignment(df_p, 'EC_preCOVID')
            fuel_calculations(df_p, obs_sizes, dic_fuel, vals_fuel)
            df_p = assignment_analysis.agg_ind_s(df_p, observation='FB')
        assignment_analysis.market_vis(df_p,name_fig ='video_'+str(100+i), color_mix = color_mix, rank = rank, color_rank = sol,
                   market = market, observation = observation, dist_limits = dist_limits, capac_limits = capac_limits, n_market = n_market,
                   m_x_ref= traff_x_max, m_y_ref = traff_y_max, ask_d_ref = ask_max, video = True, title_fig = name_periods[i],
                    smooth_param = smooth_param, weight = None, reso = reso,label_size_param=label_size_param)

    #Animation des images
    image_files = sorted(glob.glob("figures/video_storage/*.png"))
    if format_v == 'video':
        with imageio.get_writer('figures/integrated_observation/scenario/'+video_name + '.mp4', fps=frame_s) as writer:
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
        frames[0].save('figures/integrated_observation/scenario/'+ video_name + '.gif', save_all=True,
                       append_images=[frames[0] for i in range(3)] + frames[:] + [frames[-1] for i in range(3)],
                       duration=1000/frame_s, loop=0)

    return None