import pandas as pd
import numpy as np
import importlib
from src.C_Modelling import fleet_content_modelling
from src.C_Modelling import assignment_conv
from src.C_Modelling import productivity_modelling
from src.B_Model_fitting import assignment_multilogit as am
from src.A_Data_analysis import assignment_analysis
from src.A_Data_analysis import fleet_analysis

importlib.reload(fleet_analysis)
importlib.reload(assignment_analysis)
# importlib.reload(productivity_modelling)
importlib.reload(assignment_conv)


def fuel_calculations(df, seats_c, dic_fuel, fb_vals, n_flights = True): #gives Seymmour estimation of the fuel consumption in kg (maybe a corrective factor to include for gcd vs real distance)
    if n_flights :
        df['N_flights'] = df['Seats']/ np.array(seats_c)[df['Aircraft Type'].values.astype(int)]
    indices_s = df['Aircraft Type Name'].map(dic_fuel)
    distances = df['Distance_conn (km)'].values
    coeffs = fb_vals[indices_s]
    fb = coeffs[:, 0] * distances ** 2 + coeffs[:, 1] * distances + coeffs[:, 2]
    df['FB'] = fb*df['N_flights']
    return None

def scenario_ini_params(new_ac_range, new_ac_seats, measured_prop, prosp_prop):
    print('include actual generic ac, future ac')
    return None

def scenario(initial_fleet, deliveries, obs_sizes, traffic_structures, retirement_propensions, alphas, betas, omegas_0, ranges,
             types_names = None, aging_activity_coeff = 0, taux_utilisation_usuel = 1, name_video = 'sc_test', period_duration = 1,
             dic_fuel= None, vals_fuel = None):
    if types_names is None :
        types_names = [str(i) for i in range(alphas.shape[0])]
    n_y = deliveries.shape[0]
    tot_obs = traffic_structures[:,:,5].sum(axis=1)
    y_s = initial_fleet.shape[0]-deliveries.shape[0]

    fleet_pot_t = initial_fleet.copy()
    x_eq = 1 #on suppose ici des retraits continuels, et pas de scénarios de reprise
    # c'est possible sans stockage de variable, mais ça risquerait d'ajouter un peu de temps
    omegas_t = omegas_0.copy()

    type_obs = []
    vol_obs = [(fleet_pot_t*obs_sizes[np.newaxis,:])]
    omegas_l =[]
    assignments = []
    df_list = []
    age_array = np.concatenate([np.arange(y_s),y_s+period_duration*np.arange(n_y)])
    for t in range(n_y):
        print('Period '+str(t), end=',')
        fleet_pot_t[-(n_y-t),:] += deliveries[t,0,:]
        fleet_obs_t = fleet_pot_t*obs_sizes[np.newaxis,:] * taux_utilisation_usuel* np.exp(aging_activity_coeff*(-y_s+1-t*period_duration+age_array))[:,np.newaxis]

        print('total fleet: ' + str(fleet_obs_t.sum()))
        print('total constraint: '+str(tot_obs[t]))
        fleet_obs_act_t, x_eq = fleet_content_modelling.fleet_content(0,x_eq,fleet_obs_t, retirement_propensions, tot_obs[t], epsilon = 1e-6, first = True)
        if x_eq <1e-30 :
            x_eq =x_eq**(1/8)
            retirement_propensions = retirement_propensions-3*np.log(2)
        fleet_real_seats_t = fleet_obs_act_t/ taux_utilisation_usuel* np.exp(-aging_activity_coeff*(-y_s+1-t*period_duration+age_array))[:,np.newaxis]
        vol_obs.append(fleet_real_seats_t)
        # années ne sont pas importantes, eventuellement glisser en amont l'impact du vieillissement, en diminuant les quantités d'années en années.
        fleet_obs_const_t = fleet_obs_act_t.sum(axis = 0)
        # lightening the datafile
        aircraft_quantity = fleet_obs_const_t.sum()
        u = -6
        fleet_obs_const_test = np.where(fleet_obs_const_t<aircraft_quantity*10**u,0,fleet_obs_const_t) #éviter effets de seuils chelous, a REVOIR
        existence_cache_test = (fleet_obs_const_test!=0)[:, np.newaxis]
        while existence_cache_test.sum()<16:
            u-=1
            fleet_obs_const_test = np.where(fleet_obs_const_t <aircraft_quantity*10**u, 0,
                                            fleet_obs_const_t)  # éviter effets de seuils chelous, a REVOIR
            existence_cache_test = (fleet_obs_const_test != 0)[:, np.newaxis]
            if u == -200:
                print('too many zeroes update the source code')
                break
        existence_cache = existence_cache_test.copy()
        fleet_obs_const_t = fleet_obs_const_test.copy()
        non_zero_indices = np.nonzero(fleet_obs_const_t)[0]
        # print('type constraint: '+str(fleet_obs_const_t)+str(fleet_obs_const_t/(fleet_obs_t.sum(axis=0))))
        type_obs.append(fleet_obs_const_t)
        # print('type age constraint:'+str(fleet_obs_act_t))
        print('Fleet content, ok.')

        omegas_t = assignment_conv.conv_assignt(traffic_structures[t,:,2:], existence_cache, alphas, betas, omegas_t, ranges, fleet_obs_const_t[non_zero_indices],non_zero_indices,epsilon = 1e-5)
        omegas_l.append(omegas_t)
        pots = am.predict_assignt(traffic_structures[t,:,2:], existence_cache, alphas, betas, omegas_t, ranges, d_norm =1, cap_norm = 1, single_p = True)
        ass_mod = pots * traffic_structures[t,:,2:][:, 3][np.newaxis, :]
        print('Fleet assignment, ok.')

        assign = traffic_structures[t,:,:].copy()
        # 1. Répéter A pour chaque ligne de B
        assign_rep = np.tile(assign, (non_zero_indices.shape[0], 1))
        # 2. Construire la 7e colonne (obs) et 8e colonne (id)
        col7 = ass_mod[non_zero_indices,:].reshape(-1, 1)
        col8 = np.repeat(non_zero_indices, assign.shape[0]).reshape(-1, 1)
        share_filters = pots[non_zero_indices,:].reshape(-1, 1)
        assign_final = np.hstack([assign_rep, col7, col8])
        # lightening the datafile (ratio tout petit
        assign_final = assign_final[share_filters[:,0]>10**u,:]
        assignments.append(assign_final)
        df_visu = pd.DataFrame(assign_final, columns=['ADES','ADEP','Period', 'Distance_conn (km)', 'Seats_conn_p', 'Av ac seats_conn_p', 'Av ac seats', 'Aircraft Type'])
        df_visu = df_visu.astype({
            'ADES': 'int32',
            'ADEP': 'int32',
            'Period': 'int16',
            'Aircraft Type': 'int16',
            'Distance_conn (km)': 'float32',
            'Seats_conn_p': 'float32',
            'Av ac seats_conn_p': 'float32',
            'Av ac seats': 'float32'
        })
        #l'attribution du modélisé à la connection est discutable, mais simplifie la vie, et ok si on modélise la grande majorité des modèles
        dico_names =  dict(zip(range(len(types_names)), types_names))
        df_visu['Aircraft Type Name'] = df_visu['Aircraft Type'].map(dico_names)
        df_list.append(df_visu)
        # assignment_analysis.market_vis(df_visu, name_fig = 'test_scenario_y'+str(t), observation = 'Av ac seats', n_market = min(16, non_zero_indices.shape[0]))
        print('Period '+str(t)+' done.')
    vol_obs = np.array(vol_obs)
    retirement_seats_volumes = -np.diff(vol_obs, axis=0)
    retirement_seats_volumes_pos = np.where(retirement_seats_volumes<0,0,retirement_seats_volumes)
    retirement_seats_volumes_agg = retirement_seats_volumes_pos.sum(axis=1)
    fleet_analysis.visu_retirements_array(retirement_seats_volumes_agg, types_names,period_duration,min(16, len(types_names)),graph_name = name_video)
    df_props = pd.concat(df_list, ignore_index=True)
    assignment_analysis.dynamic_market_vis(df_props, [i for i in range(n_y)],name_periods = [2024+int(i*period_duration) for i in range(n_y)],
                                           video_name=name_video+'_available_seats', reso=400,smooth_param=0.04,market='Aircraft Type',
                                           observation='Av ac seats', dist_limits=[1.5e2, 2.1e4], period_ref=n_y-1,period_duration=period_duration,
                                        capac_limits=[5e3, 7e6], n_market=min(16, len(types_names)), format_v='video', weight = False, frame_s = 1/period_duration,
                                        type_obs= type_obs, obs_names=types_names) # pas de weight car c'est déjà intégré

    #déclinaison en ASKs
    productivity_modelling.ASK_assignment(df_props, 'EC_preCOVID')
    df_props = assignment_analysis.agg_ind_s(df_props, observation='ASK')
    sums_a = (
        df_props[['Aircraft Type', 'Period', 'ASK']]
        .groupby(['Aircraft Type', 'Period'])['ASK']
        .sum()
        .unstack(fill_value=0)
    )
    sums_a = sums_a.reindex(
        index=range(len(types_names)),
        columns=range(n_y),
        fill_value=0
    ).to_numpy().T
    assignment_analysis.dynamic_market_vis(df_props, [i for i in range(n_y)],name_periods=[2024 + int(i * period_duration) for i in range(n_y)],
                                           video_name=name_video+'_ask', reso=400, smooth_param=0.04, market='Aircraft Type',
                                           observation='ASK', dist_limits=[1.5e2, 2.1e4], period_ref=n_y-1,period_duration=period_duration,
                                        capac_limits=[5e3, 7e6], n_market=min(16, len(types_names)), format_v='video', weight=False,
                                           frame_s = 1/period_duration, type_obs=sums_a, obs_names=types_names)

    fuel_calculations(df_props, obs_sizes, dic_fuel, vals_fuel)
    df_props = assignment_analysis.agg_ind_s(df_props, observation='FB')
    sums_a = (
        df_props[['Aircraft Type', 'Period', 'FB']]
        .groupby(['Aircraft Type', 'Period'])['FB']
        .sum()
        .unstack(fill_value=0)
    )
    sums_a = sums_a.reindex(
        index=range(len(types_names)),
        columns=range(n_y),
        fill_value=0).to_numpy().T
    assignment_analysis.dynamic_market_vis(df_props, [i for i in range(n_y)],
                                           name_periods=[2024 + int(i * period_duration) for i in range(n_y)],
                                           video_name=name_video + '_fb', reso=400, smooth_param=0.04,
                                           market='Aircraft Type',
                                           observation='FB', dist_limits=[1.5e2, 2.1e4], period_ref=5,period_duration=period_duration,
                                        capac_limits=[5e3, 7e6], n_market=min(16, len(types_names)),
                                           format_v='video', weight=False,
                                           frame_s=1/ period_duration, type_obs=sums_a, obs_names=types_names)

    return type_obs,np.array(omegas_l),assignments