from typing import Union

from sklearn.base import ClusterMixin

from cytocluster.methods.consensus_k import KConsensusClustering
from cytocluster.methods.flowgrid import FlowGrid
from cytocluster.methods.flowsom import FlowSOM
from cytocluster.methods.latent import LatentClustering
from cytocluster.methods.parc import PARC
from cytocluster.methods.pg import Phenograph
from cytocluster.methods.spade import CytoSPADE
from cytocluster.methods.wrapper import ClusterMethod


def build_clustering_method(
        method: Union[str, ClusterMixin],
        verbose: bool = True,
        **parameters
) -> ClusterMethod:
    """
    Convenience function for generating a ClusterMethod object from either the name of a supported
    clustering method or a valid Scikit-Learn object (or one that shares the Scikit-Learn clustering signature).

    Parameters
    ----------
    method: Union[str, ClusterMixin]
        Either a string or valid Scikit-Learn object (or one that shares the Scikit-Learn clustering signature). If
        a string it must be one of the following supported clustering methods:

        * phenograph - graph based clustering using community detection (Louvain method)
        * flowsom - self-organising maps
        * k_consensus - consensus clustering with automated selection of K
        * flowgrid - scalable and fast density-based clustering solution
        * spade - density-dependent down sampling with hierarchical clustering and then up sampling by nearest
        neighbours
        * latent - choice of any ClusterMethod but applied to embedding from a method such as UMAP, tSNE or PHATE
        * parc - optimised version of phenograph

    verbose: bool (default=True)
        Display a progress bar
    parameters: optional
        Additional keyword arguments passed to constructor if method is a string

    Returns
    -------
    ClusterMethod
    """
    if method == "phenograph":
        method = ClusterMethod(klass=Phenograph(**parameters), verbose=verbose)
    elif method == "flowsom":
        method = ClusterMethod(klass=FlowSOM(**parameters), verbose=verbose)
    elif method == "k_consensus":
        method = ClusterMethod(klass=KConsensusClustering(**parameters), verbose=verbose)
    elif method == "flowgrid":
        method = ClusterMethod(klass=FlowGrid(**parameters), verbose=verbose)
    elif method == "spade":
        method = ClusterMethod(klass=CytoSPADE(**parameters), verbose=verbose)
    elif method == "latent":
        method = ClusterMethod(klass=LatentClustering(**parameters), verbose=verbose)
    elif method == "parc":
        method = ClusterMethod(klass=PARC(**parameters), verbose=verbose)
    elif isinstance(method, str):
        valid_str_methods = ["phenograph", "flowsom", "spade", "latent", "k_consensus", "parc"]
        raise ValueError(f"If a string is given must be one of {valid_str_methods}")
    elif isinstance(method, ClusterMixin):
        method = ClusterMethod(klass=method, verbose=verbose)
    else:
        raise ValueError(
            "Must provide a valid string, a ClusterMethod object, or a valid Scikit-Learn like "
            "clustering class (must have 'fit_predict' method)."
        )
    return method
