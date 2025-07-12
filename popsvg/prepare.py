import math
from typing import Literal, Union

import numpy as np
import scanpy as sc
import squidpy as sq
from numpy.typing import NDArray
from scipy.sparse import issparse
from tqdm.auto import tqdm
from tqdm.contrib import tenumerate


def _format_expression(adata: sc.AnnData):
    """Check whether the slice has raw counts in .X and convert the sparse expression matrix to dense."""
    if np.all(adata.X.data % 1 == 0):
        if issparse(adata.X):  # is sparse and contains raw counts
            adata.X = adata.X.toarray()
    else:
        raise ValueError("`adata.X` must contain raw counts.")


def _compute_spot_neighbors(
    adata: sc.AnnData,
    coord_type: Literal["hex", "square", "generic"],
    spatial_key: str = "spatial",
):
    """Compute the nearest neighbor graph."""
    if coord_type in ("hex", "square"):
        sq.gr.spatial_neighbors(adata, spatial_key=spatial_key, coord_type="grid", n_rings=1)
    elif coord_type == "generic":
        sq.gr.spatial_neighbors(adata, spatial_key=spatial_key, coord_type="generic", delaunay=True)
    else:
        raise ValueError(f"`coord_type` can only be 'hex', 'square', or 'generic', got {coord_type}.")


def _get_unwanted_genes(
    adata: sc.AnnData, prefixes: Union[str, tuple] = ("MT-", "mt-"), min_counts: int = 3, min_pct: float = 0.5
):
    is_special = adata.var_names.str.startswith(prefixes)
    min_spots = math.ceil(adata.n_obs * min_pct / 100)
    is_low_expressed = (adata.X >= min_counts).sum(0) < min_spots
    return is_special | is_low_expressed


def _log_normalize(adata: list[sc.AnnData], target_sum: float | None = None, scale: bool = False):
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata)


def preprocess_slices(
    adatas: list[sc.AnnData],
    coord_type: Literal["hex", "square", "generic"],
    spatial_key: str = "spatial",
    prefixes: str | tuple = ("MT-", "mt-"),
    min_counts: int = 1,
    min_pct: float = 1,
    target_sum: float | None = None,
    scale: bool = False,
):
    feature_count, feature_names = None, None
    feature_unwanted = []

    for n_adata, adata in tenumerate(adatas, desc="Checking slices", start=1):
        # check features are unique
        if adata.var_names.has_duplicates:
            adata.var_names_make_unique()

        # check features are consistent
        if n_adata == 1:
            feature_count, feature_names = adata.n_vars, adata.var_names
        else:
            if adata.n_vars != feature_count or np.any(adata.var_names != feature_names):
                raise ValueError(f"Inconsistent features found in the {n_adata}-th slice.")

        _format_expression(adata)
        _compute_spot_neighbors(adata, coord_type, spatial_key)

        is_unwanted = _get_unwanted_genes(adata, prefixes, min_counts, min_pct)
        feature_unwanted.append(is_unwanted)

        # filter out isolated spatial spots
        not_isolated = adata.obsp["spatial_connectivities"].sum(1) > 0
        if (num_connected := not_isolated.sum()) < adata.n_obs:
            print(f"Filtering {adata.n_obs - num_connected} isolated spots in the {n_adata}-th slice.")
            adata._inplace_subset_obs(not_isolated)

    unwanted_summary = np.row_stack(feature_unwanted)  # shape (n_subjects, n_genes)
    is_wanted = ~unwanted_summary.any(axis=0)
    for adata in tqdm(adatas, desc="Filtering features and normalizing in slices"):
        adata._inplace_subset_var(is_wanted)
        _log_normalize(adata, target_sum, scale)

    print(f"Preserved {is_wanted.sum()} features after dropping mito and low-expressed features in all slices.")
    print("Features in all slices have been log-normalized" + (" and scaled" if scale else "") + ".")
