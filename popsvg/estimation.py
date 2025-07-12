from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.cluster import k_means, kmeans_plusplus


def _estimate_gaussian_parameters(X: NDArray, resp: NDArray, post_vars: NDArray):
    # https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73a86f5f11781a0e21f24c8f47979ec67/sklearn/mixture/_gaussian_mixture.py#L287
    epsilon = 10 * np.finfo(resp.dtype).eps

    nk = resp.sum(axis=1) + epsilon  # (n_features, 2)
    resp_0, resp_1 = resp[..., 0], resp[..., 1]

    if X.ndim == 2:  # initialization
        post_means_0, post_means_1 = X, X
    elif X.ndim == 3:
        post_means_0, post_means_1 = X[..., 0], X[..., 1]

    means_data = np.einsum("gs,gs->g", resp_1, post_means_1) / nk[:, 1]
    means_neg = np.minimum(means_data, -epsilon)  # (n_features,)

    sigma2 = np.mean(
        post_vars + resp_0 * np.square(post_means_0) + resp_1 * np.square(post_means_1 - means_neg[:, np.newaxis]),
        axis=1,
    )
    weights = nk / nk.sum(axis=1, keepdims=True)
    return weights, means_neg, sigma2


def _estimate_populational_parameters(popu_svg_prob: NDArray, is_popu_svg: NDArray):
    n_features = popu_svg_prob.shape[0]
    effective_prob_sum = np.dot(popu_svg_prob, is_popu_svg)
    gk = np.array((n_features - effective_prob_sum, effective_prob_sum)) + 10 * np.finfo(popu_svg_prob.dtype).eps
    rate = gk / gk.sum()
    return rate


class TwoStageExpectationMaximizer:
    def __init__(
        self,
        rho_cutoff: float = 5e-2,
        tol: float = 1e-3,
        max_iter: int = 100,
        init_params: Literal["kmeans", "k-means++"] = "kmeans",
        random_state: int = 0,
        n_workers: int = 1,
    ):
        self.rho_cutoff = rho_cutoff
        self.alpha_cutoff = np.log(1 - rho_cutoff)
        self.tol = tol
        self.max_iter = max_iter
        self.init_params = init_params
        self.random_state = random_state
        self.n_workers = n_workers

        self.n_components = 2

    def _initialize_parameters(self, X: NDArray, kappa2: NDArray):
        n_features, n_samples = X.shape

        resp = np.zeros((n_features, n_samples, self.n_components))
        clutser_func_map = {"kmeans": k_means, "k-means++": kmeans_plusplus}
        cluster_func = clutser_func_map[self.init_params]
        X_expanded = X[..., np.newaxis]

        gen = Parallel(n_jobs=self.n_workers, return_as="generator")(
            delayed(cluster_func)(X_expanded[g], self.n_components, random_state=self.random_state)
            for g in range(n_features)
        )

        for g, (centroid, label, *_) in enumerate(gen):
            if centroid[0, 0] > centroid[1, 0]:
                label = 1 - label
            resp[g, np.arange(n_samples), label] = 1

        # for g in range(n_features):
        #     centroid, label, *_ = cluster_func(X_expanded[g], self.n_components, random_state=self.random_state)
        #     if centroid[0, 0] > centroid[1, 0]:
        #         label = 1 - label
        #     resp[g, np.arange(n_samples), label] = 1

        self._initialize(X, resp)
        self.kappa2_ = kappa2  # (n_features, n_samples)

    def _initialize(self, X: NDArray, resp: NDArray):
        # (n_features, 2), (n_features,), (n_features,)
        self.weights_, self.means_, self.sigma2_ = _estimate_gaussian_parameters(X, resp, post_vars=0)
        is_popu = self.means_ < self.alpha_cutoff
        self.rate_ = _estimate_populational_parameters(is_popu.astype(float), is_popu)  # (2,)

    def _estimate_log_prob(self, X: NDArray):
        """Estimate Gaussian log PDF"""
        n_features, n_samples = X.shape
        log_prob = np.empty((n_features, n_samples, self.n_components))
        std = np.sqrt(self.kappa2_ + self.sigma2_[:, np.newaxis])
        log_prob[..., 0] = norm.logpdf(X, loc=np.zeros((n_features, 1)), scale=std)
        log_prob[..., 1] = norm.logpdf(X, loc=self.means_[:, np.newaxis], scale=std)
        return log_prob

    def _estimate_log_weights(self) -> NDArray:
        return np.log(self.weights_)[:, np.newaxis]  # (n_features, 1, 2)

    def _estimate_log_rate(self):
        return np.log(self.rate_)[np.newaxis, :]  # (1, 2)

    def _estimate_weighted_log_prob(self, X: NDArray, stage: Literal[1, 2]) -> NDArray:
        if stage == 1:
            log_prob = self._estimate_log_prob(X)
            log_ratio = self._estimate_log_weights()
        elif stage == 2:
            log_prob = self._estimate_weighted_log_prob(X, stage=1).sum(axis=1)
            log_ratio = self._estimate_log_rate()
        return log_prob + log_ratio  # stage 1: (n_features, n_samples, 2); stage 2: (n_features, 2)

    def _estimate_log_prob_resp(self, X: NDArray, stage: Literal[1, 2]) -> tuple[NDArray, NDArray]:
        # stage 1: (n_features, n_samples, 2); stage 2: (n_features, 2)
        weighted_log_prob = self._estimate_weighted_log_prob(X, stage=stage)
        # stage 1: (n_features, n_samples); stage 2: (n_features,)
        log_prob_norm = logsumexp(weighted_log_prob, axis=3 - stage)  # f(stage) = 3 - stage; f(1) = 2, f(2) = 1
        # stage 1: (n_features, n_samples, 2); stage 2: (n_features, 2)
        log_resp = weighted_log_prob - log_prob_norm[..., np.newaxis]
        return log_prob_norm, log_resp

    def _e_step(self, X: NDArray, stage: Literal[1, 2]) -> tuple[float, NDArray]:
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X, stage=stage)
        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X: NDArray, log_resp: NDArray, stage: Literal[1, 2]):
        resp = np.exp(log_resp)  # stage 1: (n_features, n_samples, 2);  stage 2: (n_features, 2)  TODO: check sum == 1

        if stage == 1:
            post_vars = 1 / (1.0 / self.kappa2_ + 1.0 / self.sigma2_[:, np.newaxis])  # (n_features, n_samples)
            X_scaled = X / self.kappa2_  # (n_features, n_samples)
            means_scaled = self.means_ / self.sigma2_  # (n_features,)
            to_stack = (X_scaled, X_scaled + means_scaled[:, np.newaxis])
            post_means = np.stack(to_stack, axis=2) * post_vars[..., np.newaxis]  # (n_features, n_samples, 2)

            self.weights_, self.means_, self.sigma2_ = _estimate_gaussian_parameters(post_means, resp, post_vars)
        elif stage == 2:
            self.rate_ = _estimate_populational_parameters(resp[:, 1], self.is_group_spatial)

    def fit(self, X: NDArray, kappa2: NDArray):
        self._initialize_parameters(X, kappa2)
        self.converged = np.full((2,), fill_value=False)

        for stage in (1, 2):
            log_likelihood = -np.inf  # initialize the log-likelihood
            for n_iter in range(1, self.max_iter + 1):
                prev_log_likelihood = log_likelihood
                # E-step
                log_likelihood, log_resp = self._e_step(X, stage=stage)
                # M-step
                self._m_step(X, log_resp, stage=stage)
                # check convergence
                change = abs(log_likelihood - prev_log_likelihood)
                is_converged = change < self.tol
                print(
                    f"Stage {stage} iteration {n_iter:03d} - log LL change: {change:5f} | current log LL: {log_likelihood:5f}"
                )
                if is_converged:
                    print(f"Stage {stage} converged!")
                    break
                if n_iter == self.max_iter + 1:
                    print(f"Reached the maximum number of iterations ({self.max_iter}) ")
            self.converged[stage - 1] = is_converged

            if stage == 1:  # assign Z values to genes based on estimated means
                self.is_group_spatial = self.means_ < self.alpha_cutoff  # (n_features,)
                # break

    def predict_proba(self, X: NDArray, force_latent: bool = True):
        """Predict the probability that each gene is a population-level SVG"""
        _, log_resp = self._estimate_log_prob_resp(X, stage=2)
        resp = np.exp(log_resp)
        if force_latent:
            resp[~self.is_group_spatial] = [1.0, 0.0]
        return resp
