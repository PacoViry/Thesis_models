import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import jetfuelburn as jfb #si on souhaite utiliser les modèles pour la consommation de Seymmour
# from jetfuelburn.reducedorder import seymour_etal
# from jetfuelburn import ureg

### This section uses the regressed producivity coefficients to compute activity metric that can be included in the assignment scenario.
# Adds the possibility to load the chosen productivity measures, and include a linear correction coefficients to account for variations between databases.
# Individual flights dataset :
#     'ADEP', 'ADES','ADEP -1', 'ADES -1', 'ADEP +1', 'ADES +1', 'FILED OFF BLOCK TIME', 'FILED ARRIVAL TIME',
#     'ACTUAL OFF BLOCK TIME', 'ACTUAL ARRIVAL TIME', 'Aircraft Type',
#     'Aircraft Operator', 'Aircraft Registration', 'Period',
#     'Real Flight Duration', 'Planned Flight Duration', 'Ground Time',
#     'Ground Time -1', 'Planned Flight Duration +1',
#     'Real Flight Duration +1', 'Planned Flight Duration -1',
#     'Real Flight Duration -1'

def activity_assignment(df, excel_title, corrective_factor = 1.0, d_seuil=5000, v_1=830, v_2=900,v_c = 500, supp=0.5, act =  True):
    stat_est = pd.read_excel('data/productivity_measures/GT_contr_estimates_'+excel_title+'.xlsx')
    stat_est['abscisse'] = np.log(stat_est['abscisse'])

    FT_appro_lin = np.array(stat_est['abscisse'])
    TAT_appro_lin = np.array(stat_est['GT_contr'])
    # Créer la fonction d'interpolation linéaire par segments avec personnalisation
    linear_interp = interp1d(
        FT_appro_lin,
        TAT_appro_lin,
        kind='linear',
        bounds_error=False,
        fill_value=(TAT_appro_lin[0], TAT_appro_lin[-1])
    )
    df['Estimated FT'] = (((df['Distance_conn (km)']<d_seuil*corrective_factor)*corrective_factor*df['Distance_conn (km)']/v_1 +
                          (df['Distance_conn (km)'] >= d_seuil*corrective_factor) * corrective_factor *df['Distance_conn (km)'] / v_2)
                          + supp*(1-np.exp(-df['Distance_conn (km)']*corrective_factor/(v_c*v_1)*(v_1-v_c)/supp))) #terme à la schlague pour intégrer une grimpe progressive et les 2 asymptotes
    gt_cont = linear_interp(np.log(df['Estimated FT'].values+0.3)) #+0.3 correspond aux effets du taxi in/out
    if act :
        df['Activity'] = df['N_flights']*(df['Estimated FT']+gt_cont)/(24*365.25)
    df['Av_ac_seats']= df['Seats']*(df['Estimated FT']+gt_cont)/(24*365.25)

    x_l = np.linspace(np.log(0.1), np.log(15), 500)
    y = linear_interp(x_l)
    x = np.exp(x_l)
    # plt.scatter(np.exp(stat_est['abscisse']),np.array(stat_est['GT_contr']))
    # plt.plot(x, y, label ='GT_cont')
    # plt.yscale('log')

    plt.plot(x, 24*x / (x + y), label = 'ut_rate')
    # plt.yscale('log')
    plt.grid(True)
    plt.show()
    return(None)


def ASK_assignment(df, excel_title, corrective_factor = 1.0, d_seuil=5000, v_1=830, v_2=900,v_c = 500, supp=0.5):
    stat_est = pd.read_excel('data/productivity_measures/GT_contr_estimates_' + excel_title + '.xlsx')
    stat_est['abscisse'] = np.log(stat_est['abscisse'])

    FT_appro_lin = np.array(stat_est['abscisse'])
    TAT_appro_lin = np.array(stat_est['GT_contr'])
    # Créer la fonction d'interpolation linéaire par segments avec personnalisation
    linear_interp = interp1d(
        FT_appro_lin,
        TAT_appro_lin,
        kind='linear',
        bounds_error=False,
        fill_value=(TAT_appro_lin[0], TAT_appro_lin[-1])
    )
    df['Estimated FT'] = (((df['Distance_conn (km)'] < d_seuil * corrective_factor) * corrective_factor * df[
        'Distance_conn (km)'] / v_1 +
                           (df['Distance_conn (km)'] >= d_seuil * corrective_factor) * corrective_factor * df[
                               'Distance_conn (km)'] / v_2)
                          + supp * (1 - np.exp(-df['Distance_conn (km)'] * corrective_factor / v_c * (
                        v_1 - v_c) / supp)))  # terme à la schlague pour intégrer une grimpe progressive et les 2 asymptotes
    gt_cont = linear_interp(np.log(df['Estimated FT'].values + 0.3))  # +0.3 correspond aux effets du taxi in/out
    df['Seats'] = df['Av_ac_seats']/ ((df['Estimated FT'] + gt_cont) / (24 * 365.25))
    df['ASK'] = df['Seats']*df['Distance_conn (km)']
    return None

