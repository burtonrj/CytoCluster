from typing import Optional, List

import numpy as np
import pandas as pd
import hdmedians as hd
from sklearn.base import TransformerMixin
from cytotools.feedback import progress_bar


def assign_metalabels(data: pd.DataFrame, metadata: pd.DataFrame):
    """
    Given the original clustered data (data) and the meta-clustering results of
    clustering the clusters of this original data (metadata), assign the meta-cluster
    labels to the original data and return the modified dataframe with the meta cluster
    labels in a new column called 'meta_label'

    Parameters
    ----------
    data: Pandas.DataFrame
    metadata: Pandas.DataFrame

    Returns
    -------
    Pandas.DataFrame
    """
    if "meta_label" in data.columns.values:
        data = data.drop("meta_label", axis=1)
    return data.merge(
        metadata[["sample_id", "cluster_label", "meta_label"]],
        on=["sample_id", "cluster_label"],
    )


def summarise_clusters(
    data: pd.DataFrame,
    features: list,
    scale: Optional[TransformerMixin] = None,
    summary_method: str = "geomedian",
):
    """
    Average cluster parameters along columns average to generate a centroid for
    meta-clustering

    Parameters
    ----------
    data: Pandas.DataFrame
        Clustering results to average
    features: list
        List of features to use when generating centroid
    summary_method: str (default='geomedian')
        Average method, should be geomedian, mean or median
    scale: TransformerMixin, optional
        A Scikit-Learn transformer such as MinMaxScaler or StandardScaler

    Returns
    -------
    Pandas.DataFrame

    Raises
    ------
    ValueError
        If invalid method provided
    """
    if summary_method == "median":
        data = data.groupby(["sample_id", "cluster_label"])[features].median().reset_index()
    elif summary_method == "mean":
        data = data.groupby(["sample_id", "cluster_label"])[features].mean().reset_index()
    elif summary_method == "geomedian":
        cluster_geometric_median = []
        for (sample_id, cluster_id), cluster_data in data.groupby(["sample_id", "cluster_label"]):
            x = np.array(hd.geomedian(cluster_data[features].T.values)).reshape(-1, 1)
            x = pd.DataFrame(x, index=features).T
            x["sample_id"] = sample_id
            x["cluster_label"] = cluster_id
            cluster_geometric_median.append(x)
        data = pd.concat(cluster_geometric_median).reset_index(drop=True)
    else:
        raise ValueError("summary_method should be 'geomedian', 'mean' or 'median'")
    if scale is not None:
        data[features] = scale.fit_transform(data[features])
    return data


class ClusterMethod:
    """
    Wrapper for clustering methods providing access to three types of clustering:

    * cluster: groups data by "sample_id" and clusters each sample separately.
    * global_clustering: cluster all the data together (all samples).
    * meta_clustering: expects a cluster label already exists (as a column called 'cluster_label') and summarises the
    clusters by taking the mean, median, or geometric median, then clusters the summarised clusters.

    Parameters
    -----------
    klass: ClusterMixin
        Clustering object with Scikit-Learn like signatures i.e. must contain the 'fit_predict' method
    verbose: bool (default=True)
        Show progress bar

    """
    def __init__(
            self,
            klass,
            verbose: bool = True
    ):
        self.verbose = verbose
        self.method = klass
        self.valid_method()

    def valid_method(self):
        """
        Checks whether provided clustering object has 'fit_predict' method.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Invalid Class as clustering method
        """
        try:
            fit_predict = getattr(self.method, "fit_predict", None)
            assert fit_predict is not None
            assert callable(fit_predict)
        except AssertionError:
            raise ValueError("Invalid Class as clustering method, must have function 'fit_predict'")

    def _cluster(self, data: pd.DataFrame, features: List[str]):
        return self.method.fit_predict(data[features])

    def cluster(self, data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Groups data by 'sample_id' column and clusters each sample separately.

        Parameters
        ----------
        data: Pandas.DataFrame
        features: List[str]

        Returns
        -------
        Pandas.DataFrame
            DataFrame with new column 'cluster_label'
        """
        data["cluster_label"] = None
        for _id, df in progress_bar(data.groupby("sample_id"), verbose=self.verbose):
            labels = self._cluster(df, features)
            data.loc[df.index, ["cluster_label"]] = labels
        return data

    def global_clustering(self, data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Clusters entire DataFrame (all samples clustered together).

        Parameters
        ----------
        data: Pandas.DataFrame
        features: List[str]

        Returns
        -------
        Pandas.DataFrame
            DataFrame with new column 'cluster_label'
        """
        data["cluster_label"] = self._cluster(data, features)
        return data

    def meta_clustering(
        self,
        data: pd.DataFrame,
        features: List[str],
        summary_method: str = "geomedian",
        scale: Optional[TransformerMixin] = None,
    ) -> pd.DataFrame:
        """
        Generate meta-clusters from existing clusters. Expects a cluster label already exists (as a column called
        'cluster_label') and summarises the clusters by taking the mean, median, or geometric median, then clusters the
        summarised clusters.

        Parameters
        ----------
        data: Pandas.DataFrame
        features: List[str]
        summary_method: str (default='geomedian')
        scale: TransformerMixin, optional
            A Scikit-Learn transformer such as MinMaxScaler or StandardScaler

        Returns
        -------
        Pandas.DataFrame
            DataFrame with new column 'meta_label'
        """
        data = data.copy()
        metadata = summarise_clusters(
            data=data, features=features, summary_method=summary_method, scale=scale
        )
        metadata["meta_label"] = self._cluster(metadata, features)
        data = assign_metalabels(data, metadata)
        return data


