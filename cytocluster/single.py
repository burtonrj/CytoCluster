from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from sklearn.base import ClusterMixin, TransformerMixin

from cytocluster.base import Clustering
from cytocluster.methods.wrapper import ClusterMethod
from cytocluster.methods.utils import build_clustering_method
from cytocluster.base import remove_null_features


class SingleClustering(Clustering):
    """
    High-dimensional clustering offers the advantage of an unbiased approach
    to classification of single cells whilst also exploiting all available variables
    in your data (all your fluorochromes/isotypes). In cytopy, the clustering is
    performed on a Population of a FileGroup. The resulting clusters are saved
    as new Populations. We can compare the clustering results of many FileGroup's
    by 'clustering the clusters', to do this we summarise their clusters and perform meta-clustering.

    The Clustering class provides all the apparatus to perform high-dimensional clustering
    using any of the following functions from the cytopy.utils.clustering.main module:

    * sklearn_clustering - access any of the Scikit-Learn cluster/mixture classes for unsupervised learning;
      currently also provides access to HDBSCAN
    * phenograph_clustering - access to the PhenoGraph clustering algorithm
    * flowsom_clustering - access to the FlowSOM clustering algorithm

    In addition, meta-clustering (clustering or clusters) can be performed with any of the following from
    the same module:
    * sklearn_metaclustering
    * phenograph_metaclustering
    * consensus_metaclustering

    The Clustering class is algorithm agnostic and only requires that a function be
    provided that accepts a Pandas DataFrame with a column name 'sample_id' as the
    sample identifier, 'cluster_label' as the clustering results, and 'meta_label'
    as the meta clustering results. The function should also accept 'features' as
    a list of columns to use to construct the input space to the clustering algorithm.
    This function must return a Pandas DataFrame with the cluster_label/meta_label
    columns populated accordingly. It should also return two null value OR can optionally
    return a graph object, and modularity or equivalent score. These will be saved
    to the Clustering attributes.


    Parameters
    ----------
    experiment: Experiment
        Experiment to access for FileGroups to be clustered
    features: list
        Features (fluorochromes/cell markers) to use for clustering
    sample_ids: list, optional
        Name of FileGroups load from Experiment and cluster. If not given, will load all
        samples from Experiment.
    root_population: str (default="root")
        Name of the Population to use as input data for clustering
    transform: str (default="asinh")
        How to transform the data prior to clustering, see cytopy.utils.transform for valid methods
    transform_kwargs: dict, optional
        Additional keyword arguments passed to Transformer
    verbose: bool (default=True)
        Whether to provide output to stdout
    population_prefix: str (default='cluster')
        Prefix added to populations generated from clustering results

    Attributes
    ----------
    features: list
        Features (fluorochromes/cell markers) to use for clustering
    experiment: Experiment
        Experiment to access for FileGroups to be clustered
    metrics: float or int
        Metric values such as modularity score from Phenograph
    data: Pandas.DataFrame
        Feature space and clustering results. Contains features and additional columns:
        - sample_id: sample identifier
        - subject_id: subject identifier
        - cluster_label: cluster label (within sample)
        - meta_label: meta cluster label (between samples)
    """

    def local_clustering(
        self,
        method: Union[str, ClusterMethod, ClusterMixin],
        overwrite_features: Optional[List[str]] = None,
        **kwargs,
    ):
        overwrite_features = overwrite_features or self.features
        features = remove_null_features(self.data, features=overwrite_features)
        method = build_clustering_method(method=method, verbose=self.verbose, **kwargs)
        self.data = method.cluster(data=self.data, features=features)
        return self

    def global_clustering(
        self,
        method: Union[str, ClusterMethod, ClusterMixin],
        overwrite_features: Optional[List[str]] = None,
        scale: Optional[TransformerMixin] = None,
        clustering_params: Optional[Dict] = None,
    ):
        overwrite_features = overwrite_features or self.features
        features = remove_null_features(self.data, features=overwrite_features)

        data = self.data
        if scale is not None:
            data, _ = self.scale_data(features=features, scale=scale)

        clustering_params = clustering_params or {}
        method = build_clustering_method(method=method, verbose=self.verbose, **clustering_params)
        data = method.global_clustering(data=data, features=features)
        self.data["cluster_label"] = data["cluster_label"]
        return self

    def meta_clustering(
        self,
        method: Union[str, ClusterMethod],
        overwrite_features: Optional[List[str]] = None,
        summary_method: str = "median",
        scale: Optional[TransformerMixin] = None,
        **kwargs,
    ):
        overwrite_features = overwrite_features or self.features
        features = remove_null_features(self.data, features=overwrite_features)
        method = build_clustering_method(method=method, verbose=self.verbose, **kwargs)
        data = method.meta_clustering(
            data=self.data,
            features=features,
            summary_method=summary_method,
            scale=scale
        )
        self.data["meta_label"] = data["meta_label"]
        return self

    def rename_meta_clusters(self, mappings: dict):
        """
        Given a dictionary of mappings, replace the current IDs stored
        in meta_label column of the data attribute with new IDs

        Parameters
        ----------
        mappings: dict
            Mappings; {current ID: new ID}

        Returns
        -------
        None
        """
        self.data["meta_label"].replace(mappings, inplace=True)

    def reset_meta_clusters(self):
        """
        Reset meta clusters to None

        Returns
        -------
        self
        """
        self.data["meta_label"] = None
        return self

