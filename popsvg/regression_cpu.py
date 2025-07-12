import anndata as ad
import numpy as np
from formulaic import model_matrix
from joblib import Parallel, delayed
from numba import njit
from numpy.polynomial.polynomial import Polynomial
from numpy.typing import NDArray
from scipy.special import factorial
from tqdm.auto import tqdm, trange


def power_weights_mess(W: NDArray, y: NDArray, q: int = 10):
    n, d = y.shape
    res = np.empty((n, q, d), dtype=float)
    res[:, 0, :] = y
    for i in trange(1, q, desc="Computing matrix power series", leave=False):
        res[:, i, :] = W @ res[:, i - 1, :]
    return res


@njit
def orth_proj_norm_sq(YG_i: NDArray, Q: NDArray):
    QTYG_i = Q.T @ YG_i
    return YG_i.T @ YG_i - QTYG_i.T @ QTYG_i


def mess_polynomial(Q: NDArray, YG_i: NDArray, powers: NDArray, offsets: NDArray, K: NDArray, trace: float):
    H = orth_proj_norm_sq(YG_i, Q)
    H_flipped = np.fliplr(H)
    # solve the polynomial
    coeffs = np.fromiter(map(H_flipped.trace, offsets), dtype=float)
    roots = Polynomial(coeffs).deriv(m=1).roots()  # numba doesn't even support np.polyder
    alpha = np.real(roots)[np.isreal(roots)][0]

    # compute asymptotic variance of alpha from its fisher info
    optim_v = alpha**powers
    sigma2 = optim_v.T @ H @ optim_v / YG_i.shape[0]
    Sy = YG_i @ optim_v
    WXb = K @ Sy
    # alpha_var = 1.0 / (trace + np.square(WXb).sum() / sigma2)
    alpha_var = 1.0 / (trace + orth_proj_norm_sq(WXb, Q) / sigma2)
    return alpha, alpha_var


def spatial_lag_regression(adatas: list[ad.AnnData], formula: str = "1", q: int = 10, n_workers: int = 1):
    alpha, alpha_var = [], []
    for adata in tqdm(adatas, desc="Fitting spatial lag regression on each slice"):
        expr_mtx, X = adata.X, model_matrix(formula, adata.obs, output="numpy")
        n_spots, n_features = adata.n_obs, adata.n_vars

        # prepare the weight matrix
        binary_neighbor_mtx = adata.obsp["spatial_connectivities"]
        row_sums = binary_neighbor_mtx.sum(1)
        W_row_stand = binary_neighbor_mtx / row_sums

        # prepare intermediate variables
        powers = np.arange(q)
        Y = power_weights_mess(W_row_stand, expr_mtx, q=q)  # (n_spots, n_terms, n_features); n_terms = n_degrees + 1
        YG = Y / factorial(powers).reshape(-1, 1)
        Q, _ = np.linalg.qr(X, mode="reduced")  # QR decomposition for numerical stability; Q (n_spots, n_covar + 1)
        trace = (W_row_stand.T @ W_row_stand + W_row_stand @ W_row_stand).trace()
        K = W_row_stand @ Q @ Q.T  # K = W @ X @ inv(X.T @ X) @ X.T; (n_spots, n_spots)
        offsets = np.concatenate((np.flip(powers), -powers - 1))

        # compute for each gene
        res = np.array(
            Parallel(n_jobs=n_workers)(
                delayed(mess_polynomial)(Q, YG[:, :, i], powers, offsets, K, trace) for i in range(n_features)
            )
        )
        alpha.append(res[:, 0])
        alpha_var.append(res[:, 1])

        del Y, YG, Q, K, W_row_stand, binary_neighbor_mtx, row_sums, expr_mtx, X, offsets, trace, powers

    all_alpha, all_alpha_var = np.column_stack(alpha), np.column_stack(alpha_var)
    all_rho = 1 - np.exp(all_alpha)

    stats = {"all_alpha": all_alpha, "all_alpha_var": all_alpha_var, "all_rho": all_rho}
    return stats  # 3 * (n_features, n_samples)
