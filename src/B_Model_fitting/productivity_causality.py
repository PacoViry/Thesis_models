import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from scipy.sparse import issparse
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

# this file calculates and saves the GT invariance based on previous and following flight time
# it also allows to plot the prevision of the model and compare them to the litterature.
# Finally, it allows to attribute to assignment dataset a measure of type-specific aircraft utilisation.

def tat_reg(df, title='ref', reso=40):
    # =========================
    # 0. Préparation des données
    # =========================
    flight_reg = df[(~df['Ground Time'].isna()) & (df['Ground Time'] < 24)]
    flight_reg = flight_reg[
        (flight_reg['Planned Flight Duration'] > 0) &
        (flight_reg['Planned Flight Duration'] <= 20) &
        (flight_reg['Planned Flight Duration +1'] > 0) &
        (flight_reg['Planned Flight Duration +1'] <= 20)
    ]
    resolution = reso + 2
    bornes = (
        [np.log(0.01)]
        + list(np.linspace(np.log(0.49), np.log(14), resolution - 1))
        + [np.log(20)]
    )
    flight_reg['cat_ft'] = pd.cut(
        np.log(flight_reg['Planned Flight Duration']),
        bins=bornes,
        labels=range(len(bornes) - 1)
    )
    flight_reg['cat_ft_nf'] = pd.cut(
        np.log(flight_reg['Planned Flight Duration +1']),
        bins=bornes,
        labels=range(len(bornes) - 1)
    )
    flight_reg = flight_reg[['Ground Time', 'cat_ft', 'cat_ft_nf']]
    # =========================
    # 1. Encodage sparse
    # =========================
    categorical_features = ['cat_ft', 'cat_ft_nf']
    encoder = OneHotEncoder(sparse_output=True, drop='first')
    X_sparse = encoder.fit_transform(flight_reg[categorical_features])
    y = flight_reg['Ground Time'].values
    assert issparse(X_sparse)
    # =========================
    # 2. Régression Ridge (OLS)
    # =========================
    model = Ridge(alpha=0.0, fit_intercept=True)
    model.fit(X_sparse, y)
    y_pred = model.predict(X_sparse)
    print(f"Mean Squared Error: {mean_squared_error(y, y_pred)}")
    print(f"R-squared: {r2_score(y, y_pred)}")
    # =========================
    # 3. Accumulation OLS par batches
    # =========================
    batch_size = 10**5
    n, p = X_sparse.shape
    XtX = np.zeros((p + 1, p + 1), dtype=np.float64)
    Xty = np.zeros(p + 1, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        X_b = X_sparse[start:end]
        y_b = y[start:end]
        ones = np.ones(end - start)
        # XtX
        XtX[0, 0] += ones @ ones
        XtX[0, 1:] += ones @ X_b
        XtX[1:, 0] = XtX[0, 1:]
        XtX[1:, 1:] += (X_b.T @ X_b).toarray()
        # Xty
        Xty[0] += ones @ y_b
        Xty[1:] += X_b.T @ y_b
    # =========================
    # 4. Estimation exacte OLS
    # =========================
    L = np.linalg.cholesky(XtX)
    beta_full = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
    intercept_hat = beta_full[0]
    betas_hat = beta_full[1:]
    # =========================
    # 5. Variance résiduelle
    # =========================
    rss = 0.0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_b = X_sparse[start:end]
        y_b = y[start:end]

        r = y_b - intercept_hat - X_b @ betas_hat
        rss += np.dot(r, r)

    sigma_squared = rss / (n - (p + 1))
    # =========================
    # 6. Covariances utiles
    # =========================

    XtX_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(p + 1)))

    diag_cov = sigma_squared * np.diag(XtX_inv)
    cov_0 = sigma_squared * XtX_inv[0, :]
    var_0 = diag_cov[0]

    # =========================
    # 7. Effets totaux
    # =========================

    idx = np.arange(1, p + 1)
    a = np.where(idx < resolution, 2.2 / 3, 0.8 / 3)

    var_effet = (
        var_0
        + a**2 * diag_cov[idx]
        + 2 * a * cov_0[idx]
    )

    std_errors_bis = np.sqrt(var_effet)

    # =========================
    # 8. sigma_sum
    # =========================

    i = np.arange(1, resolution - 1)
    j = resolution - 1 + i

    cov_ij = sigma_squared * XtX_inv[j, i]

    var_effet_sum = (
        var_0
        + diag_cov[i]
        + diag_cov[j]
        + 2 * (cov_ij + cov_0[i] + cov_0[j])
    )

    sigma_sum = np.sqrt(var_effet_sum)

    coefs = model.coef_
    moy1 = flight_reg.groupby(['cat_ft_nf']).agg({'Ground Time': 'mean'}).reset_index()
    moy1_count = flight_reg.groupby(['cat_ft_nf']).agg({'Ground Time': 'count'}).reset_index()
    moy1_std = flight_reg.groupby(['cat_ft_nf']).agg({'Ground Time': 'std'}).reset_index()
    moy2 = flight_reg.groupby(['cat_ft']).agg({'Ground Time': 'mean'}).reset_index()
    moy2_count = flight_reg.groupby(['cat_ft']).agg({'Ground Time': 'count'}).reset_index()
    moy2_std = flight_reg.groupby(['cat_ft']).agg({'Ground Time': 'std'}).reset_index()
    moyenne_1 = np.array(moy1['Ground Time'])[1:-1]
    inc_1 = np.array(moy1_std['Ground Time'])[1:-1] / np.array(moy1_count['Ground Time'])[1:-1] ** 0.5
    moyenne_2 = np.array(moy2['Ground Time'])[1:-1]
    inc_2 = np.array(moy2_std['Ground Time'])[1:-1] / np.array(moy2_count['Ground Time'])[1:-1] ** 0.5

    coef_stud = 1.96
    sns.set_style(style='whitegrid')
    taille = np.exp(bornes)
    abscisse = np.exp((np.array(bornes[1:-2]) + np.array(bornes[2:-1])) / 2)
    plt.figure(figsize=(8, 5))
    plt.fill_between(abscisse, coefs[resolution - 1:-1] + 0.3 * model.intercept_ / 3 - coef_stud * std_errors_bis[
        resolution - 1:-1],
                     coefs[resolution - 1:-1] + 0.3 * model.intercept_ / 3 + coef_stud * std_errors_bis[
                         resolution - 1:-1], edgecolor='#72f872', color='#72f872', alpha=0.3)
    plt.scatter(abscisse, coefs[resolution - 1:-1] + 0.3 * model.intercept_ / 3, marker='+',
                label='Est. cont. to previous GT', linewidth=0.4 * abscisse ** 0.5, s=10 * abscisse ** 0.7,
                color='#72f872')

    plt.fill_between(abscisse,
                     coefs[:resolution - 2] + 2.7 * model.intercept_ / 3 - coef_stud * std_errors_bis[:resolution - 2],
                     coefs[:resolution - 2] + 2.7 * model.intercept_ / 3 + coef_stud * std_errors_bis[:resolution - 2],
                     color='#878787', alpha=0.2)
    plt.scatter(abscisse, coefs[:resolution - 2] + 2.7 * model.intercept_ / 3, marker='+',
                label='Est. cont. to next GT', linewidth=0.4 * abscisse ** 0.5, s=10 * abscisse ** 0.7, color='#878787')

    plt.scatter(abscisse, coefs[resolution - 1:-1] + coefs[:resolution - 2] + model.intercept_, color='green',
                label='Estimated total \n contribution to GT', linewidth=0.5 * abscisse ** 0.5, marker='+',
                s=20 * abscisse ** 0.7, zorder=2)
    plt.fill_between(abscisse,
                     coefs[resolution - 1:-1] + coefs[:resolution - 2] + model.intercept_ - coef_stud * sigma_sum,
                     coefs[resolution - 1:-1] + coefs[:resolution - 2] + model.intercept_ + coef_stud * sigma_sum,
                     color='green', alpha=0.2)

    plt.scatter(abscisse, moyenne_1, color='orange', label=r'Avg GT $\bf{before}$ flight',
                linewidth=0.5 * taille ** 0.5, marker='x', s=10 * abscisse ** 0.7)
    plt.fill_between(abscisse, moyenne_1 - coef_stud * inc_1, moyenne_1 + coef_stud * inc_1, color='orange', alpha=0.3)

    plt.scatter(abscisse, moyenne_2, color='red', label=r'Avg GT $\bf{after}$ flight', linewidth=0.5 * taille ** 0.5,
                marker='x', s=10 * abscisse ** 0.7)
    plt.fill_between(abscisse, moyenne_2 - coef_stud * inc_2, moyenne_2 + coef_stud * inc_2, color='red', alpha=0.2)

    handles, labels = plt.gca().get_legend_handles_labels()
    facecolor_t = to_rgba('black', alpha=0.05)
    ellipse_proxy = Patch(edgecolor='black', facecolor=facecolor_t, linewidth=1)
    handles += [ellipse_proxy]
    labels += ['Interpolated 95% \nconfidence intervals']
    legend1 = plt.legend(handles=handles, labels=labels, framealpha=0.8, fontsize=11, loc='upper left', ncol=2)

    for handle in legend1.legend_handles[:2]:
        handle.set_linewidth(1.5)  # Définit l'épaisseur des handles
        handle.set_sizes([80])  # Définit la taille des marqueurs
    for handle in legend1.legend_handles[2:3]:
        handle.set_linewidth(2)  # Définit l'épaisseur des handles
        handle.set_sizes([120])  # Définit la taille des marqueurs
    for handle in legend1.legend_handles[3:-1]:
        handle.set_linewidth(2)  # Définit l'épaisseur des handles
        handle.set_sizes([85])  # Définit la taille des marqueurs

    # plt.xscale('log')
    plt.grid(linestyle='--')
    x_ticks = np.array([0.5, 1] + list(range(2, 16, 2)))
    plt.xticks(x_ticks)
    plt.xlim(0.49, abscisse[-1] + 0.1)
    plt.ylim(0)
    plt.xlabel('Filed block time (h)', fontsize=14)
    plt.ylabel('Ground time (h)', fontsize=14)
    plt.savefig('figures/productivity_figures/'+'GT_contr_estimates_'+title+'.pdf')
    plt.show()
    save_df = pd.DataFrame({'abscisse': abscisse, 'GT_contr': coefs[resolution - 1:-1] + coefs[:resolution - 2] + model.intercept_})
    save_df.to_excel( 'data/productivity_measures/GT_contr_estimates_'+title+'.xlsx')
    return(None)


