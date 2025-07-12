import os

import anndata as ad
import numpy as np

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.97"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

try:
    import jax
except ImportError:
    raise ImportError("Jax is not installed!")
devices = jax.devices("gpu")
if not devices:
    raise RuntimeError("The GPU is not available!")

import jax.numpy as jnp
from formulaic import model_matrix
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


def orth_proj_norm_sq(YG_i: jax.Array, Q: jax.Array):
    QTYG_i = Q.T @ YG_i
    return YG_i.T @ YG_i - QTYG_i.T @ QTYG_i


def mess_polynomial(Q: jax.Array, YG_i: jax.Array, powers: jax.Array, offsets: jax.Array, K: jax.Array, trace: float):
    H = orth_proj_norm_sq(YG_i, Q)
    H_flipped = jnp.fliplr(H)
    # solve the polynomial
    get_traces = jax.vmap(lambda mtx, offset: mtx.trace(offset), in_axes=(None, 0))
    coeffs = get_traces(H_flipped, offsets)
    roots = jnp.roots(jnp.polyder(coeffs[::-1]), strip_zeros=False)
    alpha = jnp.where(jnp.isreal(roots), jnp.real(roots), 0).sum()  # to comply with jax.jit

    # compute asymptotic variance of alpha from its fisher info
    optim_v = alpha**powers
    sigma2 = optim_v.T @ H @ optim_v / YG_i.shape[0]
    Sy = YG_i @ optim_v
    WXb = K @ Sy
    # alpha_var = 1.0 / (trace + jnp.square(K @ Sy).sum() / sigma2)
    alpha_var = 1.0 / (trace + orth_proj_norm_sq(WXb, Q) / sigma2)
    return alpha, alpha_var


def spatial_lag_regression(adatas: list[ad.AnnData], formula: str = "1", q: int = 10):
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

        # move to gpu
        Q, YG, powers, offsets = jnp.asarray(Q), jnp.asarray(YG), jnp.asarray(powers), jnp.asarray(offsets)
        K, trace = jnp.asarray(K), jnp.asarray(trace)

        # compute for each gene
        if n_spots <= 5e4:
            vec_mess_polynomial = jax.vmap(mess_polynomial, in_axes=(None, 2, None, None, None, None))
            res = vec_mess_polynomial(Q, YG, powers, offsets, K, trace)
            alpha.append(res[0])
            alpha_var.append(res[1])
        else:
            res = jnp.array(
                [
                    mess_polynomial(Q, YG[:, :, i], powers, offsets, K, trace)
                    for i in trange(n_features, desc="Fitting spatial lag models on each gene", leave=False)
                ]
            )
            alpha.append(res[:, 0])
            alpha_var.append(res[:, 1])

        del Q, YG, powers, offsets, K, trace

    all_alpha, all_alpha_var = jnp.column_stack(alpha), jnp.column_stack(alpha_var)
    all_alpha, all_alpha_var = np.asarray(all_alpha), np.asarray(all_alpha_var)
    all_rho = 1 - np.exp(all_alpha)

    stats = {"all_alpha": all_alpha, "all_alpha_var": all_alpha_var, "all_rho": all_rho}
    return stats  # 3 * (n_features, n_samples)
