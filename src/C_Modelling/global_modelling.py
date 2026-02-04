import pandas as pd
import numpy as np
import importlib
import matplotlib.pyplot as plt
from src.C_Modelling import fleet_content_modelling
from src.C_Modelling import assignment_conv
from src.B_Model_fitting import assignment_multilogit as am
from src.A_Data_analysis import assignment_analysis
importlib.reload(assignment_conv)
importlib.reload(assignment_analysis)


def scenario_ini_params(new_ac_range, new_ac_seats, measured_prop, prosp_prop):
    print('include actual generic ac, future ac')
    return None

def scenario(initial_fleet, deliveries, obs_sizes, traffic_structures, retirement_propensions, alphas, betas, omegas_0, ranges, types_names = None, aging_activity_coeff = 0):
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
    omegas_l =[]
    assignments = []

    for t in range(n_y):
        print('year '+str(t), end=',')
        fleet_pot_t += deliveries[t,:,:]
        fleet_obs_t = fleet_pot_t*obs_sizes[np.newaxis,:] * np.exp(-aging_activity_coeff*(t-np.arange(n_y+y_s)))[:,np.newaxis]
        print('total constraint: '+str(tot_obs[t]))
        fleet_obs_act_t, x_eq = fleet_content_modelling.fleet_content(0,x_eq,fleet_obs_t, retirement_propensions, tot_obs[t], epsilon = 0.0001, first = True)
        # années ne sont pas importantes, eventuellement glisser en amont l'impact du vieillissement, en diminuant les quantités d'années en années.
        fleet_obs_const_t = fleet_obs_act_t.sum(axis = 0)
        existence_cache = (fleet_obs_const_t!=0)[:, np.newaxis]
        non_zero_indices = np.nonzero(fleet_obs_const_t)[0]
        print('type constraint: '+str(fleet_obs_const_t))
        type_obs.append(fleet_obs_const_t)
        # print('type age constraint:'+str(fleet_obs_act_t))
        print('Fleet content, ok.')

        omegas_t = assignment_conv.conv_assignt(traffic_structures[t,:,2:], existence_cache, alphas, betas, omegas_t, ranges, fleet_obs_const_t[non_zero_indices],non_zero_indices)
        omegas_l.append(omegas_t)
        pots = am.predict_assignt(traffic_structures[t,:,2:], existence_cache, alphas, betas, omegas_t, ranges, d_norm =1, cap_norm = 1)
        ass_mod = pots * traffic_structures[t,:,2:][:, 3][np.newaxis, :]
        print(ass_mod)
        print('Fleet assignment, ok.')

        assign = traffic_structures[t,:,:].copy()
        # 1. Répéter A pour chaque ligne de B
        assign_rep = np.tile(assign, (non_zero_indices.shape[0], 1))
        # 2. Construire la 7e colonne (obs) et 8e colonne (id)
        col7 = ass_mod[non_zero_indices,:].reshape(-1, 1)
        col8 = np.repeat(non_zero_indices, assign.shape[0]).reshape(-1, 1)
        assign_final = np.hstack([assign_rep, col7, col8])
        assignments.append(assign_final)
        df_visu = pd.DataFrame(assign_final, columns=['ADES','ADEP','Period', 'Distance_conn (km)', 'Seats_conn_p', 'Activity Seats_conn_p', 'Activity Seats', 'Aircraft Type'])
        #l'attribution du modélisé à la connection est discutable, mais simplifie la vie, et ok si on modélise la grande majorité des modèles
        df_visu['Aircraft Type Name'] = df_visu['Aircraft Type']
        assignment_analysis.market_vis(df_visu, name_fig = 'test_scenario_y'+str(t), observation = 'Activity Seats', n_market = min(12, non_zero_indices.shape[0]))
        print('Fleet visu, ok.')

    type_obs = np.array(type_obs)
    n_bars = type_obs.shape[0]  # Nombre de barres (n)
    p_categories = type_obs.shape[1]  # Nombre de catégories (p)
    fig, ax = plt.subplots()
    bottom = np.zeros(n_bars)  # Pour gérer l'empilement
    plt.grid(axis='y', color='black', linestyle='--')

    for i in range(p_categories):
        ax.bar(range(n_bars), type_obs[:, p_categories - i - 1], bottom=bottom, label=types_names[p_categories - i - 1],
               alpha=1, zorder=1)
        bottom += type_obs[:, p_categories - i - 1]

    ax.set_xticks(range(n_bars))
    ax.set_xticklabels([f'Year {i}' for i in range(n_bars)])
    ax.legend(framealpha=1)
    plt.title("Activity graph")
    plt.show()
    return type_obs,np.array(omegas_l),assignments