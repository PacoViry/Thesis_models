import numpy as np
import importlib
import matplotlib.pyplot as plt

from src.B_Model_fitting import assignment_multilogit as am
importlib.reload(am)

# def conv_assignt(conn_data, existence_cache, alphas, betas, omegas_p, ranges, constraints, indices, epsilon = 1e-4):
#     omegas = omegas_p.copy()
#     diffs =np.array(1)*np.ones(len(indices))
#     weights = conn_data[:, 3][np.newaxis, :]
#         # L_diffs = []
#     while np.abs(diffs).max() > epsilon:
#         speed_adapt = 1.4
#         if np.abs(diffs).max() > 0.2 :
#             speed_adapt = 1
#         elif np.abs(diffs).max() > 0.1 :
#             speed_adapt = 1.2
#         pots = am.predict_assignt(conn_data, existence_cache, alphas, betas, omegas, ranges, d_norm =1, cap_norm = 1, single_p = True)
#         ass_mod = pots[indices, :] * weights
#         ass_const = ass_mod.sum(axis=1)
#         diffs = (np.log(constraints) - np.log(ass_const))
#         omegas[indices]+= speed_adapt*diffs[:, np.newaxis]
#         # L_diffs.append(diffs)
#     return omegas

def conv_assignt(conn_data, existence_cache, alphas, betas, omegas_p, ranges, constraints, indices, epsilon=1e-4):
    omegas = omegas_p.copy()  # Éviter la copie si possible, sinon garder .copy()
    diffs = np.ones(len(indices))
    weights = conn_data[:, 3][np.newaxis, :]

    # Pré-calculer log(constraints) une seule fois
    log_constraints = np.log(constraints)
    N_r = 0
    while True:
        N_r+=1
        max_diff = np.abs(diffs).max()
        if max_diff <= epsilon:
            break
        if max_diff > 0.2:
            speed_adapt = 1
        elif max_diff > 0.1:
            speed_adapt = 1.4
        else:
            speed_adapt = 1.7

        pots = am.predict_assignt(conn_data, existence_cache, alphas, betas, omegas, ranges,
                                  d_norm=1, cap_norm=1, single_p=True)

        ass_mod = pots[indices, :] * weights
        ass_const = ass_mod.sum(axis=1)

        # Éviter log(0) avec un epsilon
        diffs = log_constraints - np.log(ass_const)
        # if N_r>1:
        #     plt.plot(diffs)        # Adapter la vitesse en une seule évaluation
        omegas[indices] += speed_adapt * diffs[:, np.newaxis]
    # plt.show()
    print(N_r)
    return omegas


def range_distance(pots, conn_data, ranges):
    ass_mod = pots * conn_data[:, 3][np.newaxis, :]
    avg_rg = ((ass_mod*conn_data[:, 1][np.newaxis, :]).sum(axis=1))/ass_mod.sum(axis=1)
    return avg_rg


