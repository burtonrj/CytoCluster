#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
All clustering objects inherit from the Clustering class which provides tools for visualising cluster results.

Copyright 2022 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math
from collections import defaultdict

from cytoplots.general import density_plot, scatterplot, box_swarm_plot, ColumnWrapFigure
from cytotools.dimension_reduction import DimensionReduction
from cytotools.feedback import progress_bar
from cytotools.transform import apply_transform
from matplotlib import pyplot as plt
from sklearn.base import TransformerMixin
from typing import Optional, List, Dict, Tuple, Union, Iterable, Callable
import seaborn as sns
import pandas as pd
import numpy as np
import logging

from sklearn.cluster import AgglomerativeClustering

from cytocluster.methods.consensus_k import KConsensusClustering
from cytocluster.methods.utils import build_clustering_method
from cytocluster.methods.wrapper import ClusterMethod
from cytocluster.metrics import InternalMetric, init_internal_metrics
from cytocluster.plotting import plot_meta_clusters, clustered_heatmap, boxswarm_and_source_count, silhouette_analysis

logger = logging.getLogger(__name__)


def simpson_di(cluster_counts: Dict[str, int]) -> float:
    """
    Calculate the simpson diversity index for clusters of N subjects.

    Parameters
    ----------
    cluster_counts: Dict[str, int]
        Key should correspond to cluster name and value the unique number of subjects within that cluster.

    Returns
    -------
    float
    """
    N = sum(cluster_counts.values())
    cluster_counts = {k: v for k, v in cluster_counts.items() if v != 0}
    return sum((float(n) / N) ** 2 for n in cluster_counts.values())


def remove_null_features(data: pd.DataFrame, features: Optional[List[str]] = None) -> List[str]:
    """
    Check for null values in the dataframe.
    Returns a list of column names for columns with no missing values.

    Parameters
    ----------
    data: Pandas.DataFrame
    features: List[str], optional

    Returns
    -------
    List
        List of valid columns
    """
    features = features or data.columns.tolist()
    null_cols = data[features].isnull().sum()[data[features].isnull().sum() > 0].index.values
    if null_cols.size != 0:
        logger.warning(
            f"The following columns contain null values and will be excluded from clustering analysis: {null_cols}"
        )
    return [x for x in features if x not in null_cols]


class Clustering:
    """
    Base class for clustering helper objects (see single and ensemble modules). Provides access to high level
    methods for visualisation and interrogation of clustering results.

    Parameters
    ----------
    data: Pandas.DataFrame
        Input space for clustering
    features: List[str]
        Names of columns in data to be treated as features in clustering
    verbose: bool (default=True)
        Provide progress bars as feedback
    random_state: int (default=42)
    n_sources: Dict, optional
    pre_embedded: bool (default=False)
        Set True if dimension reduction already performed on data and latent variables are stored in columns
    labels: Dict[str, Union[pd.Series, np.ndarray]], optional
        Only required for ensemble clustering - should contain the labels from base clusterings
    """
    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        verbose: bool = True,
        random_state: int = 42,
        n_sources: Optional[Dict] = None,
        pre_embedded: bool = False,
        labels: Optional[Dict[str, Union[pd.Series, np.ndarray]]] = None
    ):
        np.random.seed(random_state)
        self.verbose = verbose
        self.features = features
        self.data = data
        self._embedding_cache = None
        self._n_sources = n_sources or {}
        self.pre_embedded = pre_embedded
        self.labels = labels

    def __repr__(self):
        return (
            "Clustering object\n"
            f"data: {self.data.shape[0]} observations, {len(self.features)} features, {self.data.shape[1]} columns\n"
            f"N samples: {self.data.sample_id.nunique()}\n"
            f"pre-embedded latent variables: {self.pre_embedded}\n"
            f"embeddings cache: {self._embedding_cache is not None}\n"
        )

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        features: list,
        verbose: bool = True,
        random_state: int = 42,
        pre_embedded: bool = False,
        labels: Optional[Dict[str, Union[pd.Series, np.ndarray]]] = None
    ):
        """
        Generate a new Clustering object from a DataFrame

        Parameters
        ----------
        data: Pandas.DataFrame
        Input space for clustering
        features: List[str]
            Names of columns in data to be treated as features in clustering
        verbose: bool (default=True)
            Provide progress bars as feedback
        random_state: int (default=42)
        pre_embedded: bool (default=False)
            Set True if dimension reduction already performed on data and latent variables are stored in columns
        labels: Dict[str, Union[pd.Series, np.ndarray]], optional
            Only required for ensemble clustering - should contain the labels from base clusterings

        Returns
        -------
        New Clustering object
        """
        if "sample_id" not in data.columns.values:
            raise ValueError("Data missing 'sample_id' column,")
        n_sources = None
        if "meta_label" not in data.columns:
            data["meta_label"] = None
        if "cluster_label" not in data.columns:
            data["cluster_label"] = None
        else:
            if "n_sources" in data.columns:
                n_sources = {
                    cluster: n
                    for cluster, n in data[["cluster_label", "n_sources"]].drop_duplicates().itertuples(index=False)
                }
        return cls(
            data=data.copy(),
            features=features,
            verbose=verbose,
            random_state=random_state,
            n_sources=n_sources,
            pre_embedded=pre_embedded,
            labels=labels
        )

    def scale_data(
            self,
            features: List[str],
            scale: TransformerMixin
    ) -> Tuple[pd.DataFrame, TransformerMixin]:
        """
        Scale features using some Scikit-Learn transformer

        Parameters
        ----------
        features: List[str]
        scale: Scikit-Learn transformer e.g. MinMaxScaler

        Returns
        -------
        Pandas.DataFrame, Scaling object
        """
        data = self.data.copy()
        data[features] = scale.fit_transform(self.data[features])
        return data, scale

    def transform_features(self, method: str, features: Optional[List[str]] = None, **kwargs):
        """
        Apply a transformation to the features e.g. Logicle or Arcsine transform.

        Parameters
        ----------
        method: str
            Valid methods include asinh, logicle, hyperlog, or log
        features: List[str], optional
            Defaults to features specified on construction
        kwargs
            Additional keyword arguments passed to transform

        Returns
        -------
        None
        """
        features = features or self.features
        self.data = apply_transform(data=self.data, features=features, method=method, **kwargs)

    def reset_clusters(self):
        """
        Resets cluster and meta cluster labels to None

        Returns
        -------
        self
        """
        self.data["cluster_label"] = None
        self.data["meta_label"] = None
        return self

    def rename_clusters(self, sample_id: str, mappings: dict):
        """
        Given a dictionary of mappings, replace the current IDs stored
        in cluster_label column for a particular sample

        Parameters
        ----------
        sample_id: str
        mappings: dict
            Mappings; {current ID: new ID}

        Returns
        -------
        None
        """
        if sample_id != "all":
            idx = self.data[self.data.sample_id == sample_id].index
            self.data.loc[idx, "cluster_label"] = self.data.loc[idx]["cluster_label"].replace(mappings)
        else:
            self.data["cluster_label"] = self.data["cluster_label"].replace(mappings)

    def dimension_reduction(
        self,
        n: Optional[int] = 1000,
        sample_id: Optional[str] = None,
        overwrite_cache: bool = False,
        method: str = "UMAP",
        replace: bool = False,
        weights: Optional[Iterable] = None,
        random_state: int = 42,
        **dim_reduction_kwargs,
    ):
        """
        Perform and cache dimension reduction. The results are cached as a pandas dataframe, which is optionally
        a sample of the original data.

        Parameters
        ----------
        n: int, optional
            Optional size of sample of data to perform dimension reduction on
        sample_id: str, optional
            Optional sample_id to perform dimension reduction on
        overwrite_cache: bool (default=False)
            If True, replace existing cached results, otherwise return cached results if available for selected method
        method: str (default="UMAP")
            Dimension reduction method to use e.g. umap, phate, or tsne - see cytotools.dimension_reduction
        replace: bool (default=False)
            Sample with replacement
        weights: Iterable, optional
            Weights for sampling
        random_state: int (default=42)
        dim_reduction_kwargs
            Additional keyword arguments passed to DimensionReduction object

        Returns
        -------
        Pandas.DataFrame
        """
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        dim_reduction_kwargs["n_components"] = dim_reduction_kwargs.get("n_components", 2)
        if self.pre_embedded:
            if len([x for x in self.data.columns if method in x]) < 2:
                raise ValueError(
                    f"Object constructed as a pre-embedded experiment but {method} embeddings not found "
                    f"in data. Either change method to the correct value or check that the experiment "
                    f"has undergone dimensionality reduction."
                )
            return self.data
        reducer = DimensionReduction(method=method, **dim_reduction_kwargs)
        if sample_id and self._embedding_cache is not None:
            if self._embedding_cache.sample_id.nunique() > 1:
                # Embedding previously captures multiple samples
                overwrite_cache = True
            elif self.data.sample_id.unique()[0] != sample_id:
                # Embedding previously captures another sample
                overwrite_cache = True
        if self._embedding_cache is not None and not overwrite_cache:
            if f"{method}1" not in self._embedding_cache.columns:
                self._embedding_cache = reducer.fit_transform(data=self._embedding_cache, features=self.features)
            else:
                return self._embedding_cache
        if overwrite_cache or self._embedding_cache is None:
            data = self.data.copy()
            if sample_id:
                data = data[data.sample_id == sample_id]
                if self.data.shape[0] > n:
                    data = self.data.sample(n)
            elif n is not None:
                data = data.groupby("sample_id").sample(
                    n=n, replace=replace, weights=weights, random_state=random_state
                )
            self._embedding_cache = reducer.fit_transform(data=data, features=self.features)
        if sample_id:
            self._embedding_cache["cluster_label"] = self.data[self.data.sample_id == sample_id]["cluster_label"]
            self._embedding_cache["meta_label"] = self.data[self.data.sample_id == sample_id]["meta_label"]
        else:
            self._embedding_cache["cluster_label"] = self.data["cluster_label"]
            self._embedding_cache["meta_label"] = self.data["meta_label"]
        return self._embedding_cache

    def plot_density(
        self,
        n: int = 1000,
        sample_id: Optional[str] = None,
        overwrite_cache: bool = False,
        method: str = "UMAP",
        dim_reduction_kwargs: Optional[Dict] = None,
        subset: Optional[str] = None,
        plot_n: Optional[int] = None,
        **plot_kwargs,
    ):
        """
        Plot a two dimension density plot in some embedded space using dimension reduction. This calls the
        'dimension_reduction' method and will use cache results if available for the chosen method or if
        'overwrite_cache' is True.

        Parameters
        ----------
        n: int, optional
            Optional size of sample of data to perform dimension reduction on
        sample_id: str, optional
            Optional sample_id to perform dimension reduction on
        overwrite_cache: bool (default=False)
            If True, replace existing cached results, otherwise return cached results if available for selected method
        method: str (default="UMAP")
            Dimension reduction method to use e.g. umap, phate, or tsne - see cytotools.dimension_reduction
        dim_reduction_kwargs
            Additional keyword arguments passed to DimensionReduction object
        subset: str, optional
            String value passed to 'query' method of the Pandas.DataFrame stored in 'data', use this to subset
            data prior to plotting
        plot_n: int, optional
            Downsample data further prior to plotting
        plot_kwargs
            Additional keyword arguments passed to 'density_plot' function

        Returns
        -------
        matplotlib.Figure
        """
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        data = self.dimension_reduction(
            n=n, sample_id=sample_id, overwrite_cache=overwrite_cache, method=method, **dim_reduction_kwargs
        )
        if subset:
            data = data.query(subset).copy()
        if plot_n and (data.shape[0] > plot_n):
            data = data.sample(plot_n)
        return density_plot(data=data, x=f"{method}1", y=f"{method}2", **plot_kwargs)

    def plot(
        self,
        label: str,
        discrete: bool = True,
        n: int = 1000,
        sample_id: Optional[str] = None,
        overwrite_cache: bool = False,
        method: str = "UMAP",
        dim_reduction_kwargs: Optional[Dict] = None,
        subset: Optional[str] = None,
        **plot_kwargs,
    ):
        """
        Plot a two dimension plot in some embedded space using dimension reduction. This calls the
        'dimension_reduction' method and will use cache results if available for the chosen method or if
        'overwrite_cache' is True. Data points are coloured using some 'label' that should correspond
        to a column in 'data'.

        Parameters
        ----------
        label: str
            Name of column to use to colour data points
        discrete: bool (default=True)
            Should the label be treated as a discrete variable?
        n: int, optional
            Optional size of sample of data to perform dimension reduction on
        sample_id: str, optional
            Optional sample_id to perform dimension reduction on
        overwrite_cache: bool (default=False)
            If True, replace existing cached results, otherwise return cached results if available for selected method
        method: str (default="UMAP")
            Dimension reduction method to use e.g. umap, phate, or tsne - see cytotools.dimension_reduction
        dim_reduction_kwargs
            Additional keyword arguments passed to DimensionReduction object
        subset: str, optional
            String value passed to 'query' method of the Pandas.DataFrame stored in 'data', use this to subset
            data prior to plotting
        plot_kwargs
            Additional keyword arguments passed to 'density_plot' function

        Returns
        -------
        matplotlib.Figure
        """
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        data = self.dimension_reduction(
            n=n, sample_id=sample_id, overwrite_cache=overwrite_cache, method=method, **dim_reduction_kwargs
        )
        if subset:
            data = data.query(subset).copy()
        return scatterplot(data=data, x=f"{method}1", y=f"{method}2", label=label, discrete=discrete, **plot_kwargs)

    def plot_cluster_membership(
        self,
        n: int = 1000,
        sample_id: Optional[str] = None,
        overwrite_cache: bool = False,
        method: str = "UMAP",
        dim_reduction_kwargs: Optional[Dict] = None,
        subset: Optional[str] = None,
        **plot_kwargs,
    ):
        """
        Same as the plot function but defaults to using the 'cluster_label' as 'label' for colouring data points.
        Assumes clustering has been performed and 'cluster_label' column has been populated.

        Parameters
        ----------
        n: int, optional
            Optional size of sample of data to perform dimension reduction on
        sample_id: str, optional
            Optional sample_id to perform dimension reduction on
        overwrite_cache: bool (default=False)
            If True, replace existing cached results, otherwise return cached results if available for selected method
        method: str (default="UMAP")
            Dimension reduction method to use e.g. umap, phate, or tsne - see cytotools.dimension_reduction
        dim_reduction_kwargs
            Additional keyword arguments passed to DimensionReduction object
        subset: str, optional
            String value passed to 'query' method of the Pandas.DataFrame stored in 'data', use this to subset
            data prior to plotting
        plot_kwargs
            Additional keyword arguments passed to 'scatterplot' function

        Returns
        -------
        matplotlib.Figure
        """
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        data = self.dimension_reduction(
            n=n, sample_id=sample_id, overwrite_cache=overwrite_cache, method=method, **dim_reduction_kwargs
        )
        data["cluster_label"] = self.data["cluster_label"]
        if subset:
            data = data.query(subset)
        return scatterplot(
            data=data, x=f"{method}1", y=f"{method}2", label="cluster_label", discrete=True, **plot_kwargs
        )

    def plot_meta_cluster_centroids(
        self,
        label: str = "meta_label",
        discrete: bool = True,
        method: str = "UMAP",
        dim_reduction_kwargs: Optional[Dict] = None,
        subset: Optional[str] = None,
        **kwargs,
    ):
        """
        Plot each cluster as a data point in embedded space (uses the dimension_reduction method to generate embeddings).
        Each cluster can be coloured by some label, by default it assumes meta-clustering has been performed and will
        colour data points using the 'meta_label' column.

        Parameters
        ----------
        label: str (default='meta_label')
            Name of column to use to colour data points
        discrete: bool (default=True)
            Should the label be treated as a discrete variable?
        method: str (default="UMAP")
            Dimension reduction method to use e.g. umap, phate, or tsne - see cytotools.dimension_reduction
        dim_reduction_kwargs
            Additional keyword arguments passed to DimensionReduction object
        subset: str, optional
            String value passed to 'query' method of the Pandas.DataFrame stored in 'data', use this to subset
            data prior to plotting
        kwargs
            Additional keyword arguments passed to 'plot_meta_clusters' function

        Returns
        -------
        matplotlib.Figure
        """
        if "meta_label" not in self.data.columns:
            raise KeyError("Meta-clustering has not been performed")
        data = self.data
        if subset:
            data = data.query(subset)
        return plot_meta_clusters(
            data=data,
            features=self.features,
            colour_label=label,
            discrete=discrete,
            method=method,
            dim_reduction_kwargs=dim_reduction_kwargs,
            **kwargs,
        )

    def heatmap(
        self,
        features: Optional[List[str]] = None,
        sample_id: Optional[str] = None,
        meta_label: bool = False,
        include_labels: Optional[List[str]] = None,
        subset: Optional[str] = None,
        plot_orientation: str = "vertical",
        **kwargs,
    ):
        """
        Generate a heatmap of cluster phenotypes. Defaults to using the 'cluster_label' column for clusters unless
        'meta_label' is True.

        Parameters
        ----------
        features: List[str], optional
            List of columns to use as features in heatmaps, defaults to 'features' attribute
        sample_id: str, optional
            Just display phenotypes for one sample_id
        meta_label: bool (default=False)
            Use 'meta_label' column for clusters
        include_labels: List[str], optional
            Filter the clusters to only include these labels
        subset: str, optional
            String value passed to 'query' method of the Pandas.DataFrame stored in 'data', use this to subset
            data prior to plotting
        plot_orientation: str (default='vertical')
        kwargs:
            Additional keyword arguments passed to cytocluster.plotting.clustered_heatmap

        Returns
        -------
        Seaborn.ClusterGrid
        """
        features = features or self.features
        data = self.data.copy()
        if subset:
            data = data.query(subset)
        if include_labels:
            if meta_label:
                data = data[data["meta_label"].isin(include_labels)]
            else:
                data = data[data["cluster_label"].isin(include_labels)]
        return clustered_heatmap(
            data=data,
            features=features,
            sample_id=sample_id,
            meta_label=meta_label,
            plot_orientation=plot_orientation,
            **kwargs,
        )

    @staticmethod
    def _count_to_proportion(df: pd.DataFrame):
        df["Percentage"] = (df["Count"] / df["Count"].sum()) * 100
        return df

    @staticmethod
    def _fill_null_clusters(data: pd.DataFrame, label: str):
        labels = data[label].unique()
        updated_data = []
        for sample_id, sample_df in data.groupby("sample_id"):
            missing_labels = [i for i in labels if i not in sample_df[label].unique()]
            updated_data.append(
                pd.concat(
                    [
                        sample_df,
                        pd.DataFrame(
                            {
                                "sample_id": [sample_id for _ in range(len(missing_labels))],
                                "Count": [0 for _ in range(len(missing_labels))],
                                label: missing_labels,
                            }
                        ),
                    ]
                )
            )
        return pd.concat(updated_data).reset_index(drop=True)

    def simpsons_diversity_index(
        self, groupby: str = "cluster_label", cell_identifier: str = "sample_id"
    ) -> pd.DataFrame:
        """
        Generate a dataframe of simpson diversity index for each cluster

        Parameters
        ----------
        groupby: str (default="cluster_label")
        cell_identifier: str (default="sample_id")

        Returns
        -------
        Pandas.DataFrame
        """
        sdi = {}
        for cid, df in self.data.groupby(groupby):
            sdi[cid] = simpson_di(df[cell_identifier].value_counts().to_dict())
        return (
            pd.DataFrame(sdi, index=["Simpson's Diversity Index"]).T.reset_index().rename(columns={"index": "Cluster"})
        )

    def plot_simpsons_diversity_index(
        self, groupby: str = "cluster_label", cell_identifier: str = "sample_id", **bar_plot_kwargs
    ):
        """
        Generates a bar plot of simpsons diversity index for each cluster

        Parameters
        ----------
        groupby: str (default="cluster_label")
        cell_identifier: str (default="sample_id")
        bar_plot_kwargs:
            Passed to Seaborn.barplot

        Returns
        -------
        Matplotlib.Axes
        """
        sdi = self.simpsons_diversity_index(groupby, cell_identifier)
        sdi = sdi.sort_values("Simpson's Diversity Index")

        bar_plot_kwargs = bar_plot_kwargs or {}
        bar_plot_kwargs["color"] = bar_plot_kwargs.get("color", "royalblue")
        bar_plot_kwargs["order"] = bar_plot_kwargs.get("order", sdi.Cluster.values)

        ax = sns.barplot(data=sdi, x="Cluster", y="Simpson's Diversity Index", **bar_plot_kwargs)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        return ax

    def cluster_proportions(
        self,
        label: str = "cluster_label",
        filter_clusters: Optional[List] = None,
        hue: Optional[str] = None,
        plot_source_count: bool = False,
        log10_percentage: bool = False,
        replace_null_population: float = 0.01,
        y_label: str = "Percentage",
        subset: Optional[str] = None,
        return_data: bool = False,
        **plot_kwargs,
    ):
        data = self.data.copy()
        if subset:
            data = data.query(subset).copy()
        x = data.groupby("sample_id")[label].value_counts()
        x.name = "Count"
        x = x.reset_index()
        plot_data = x.groupby("sample_id").apply(self._count_to_proportion).reset_index()
        plot_data = self._fill_null_clusters(data=plot_data, label=label)
        plot_data.rename(columns={"Percentage": y_label}, inplace=True)

        if filter_clusters:
            plot_data = plot_data[plot_data[label].isin(filter_clusters)]

        if hue:
            colour_mapping = self.data[["sample_id", hue]].drop_duplicates()
            plot_data = plot_data.merge(colour_mapping, on="sample_id")

        if log10_percentage:
            plot_data[f"log10({y_label})"] = np.log10(
                plot_data[y_label].apply(lambda i: replace_null_population if i == 0 else i)
            )
            y_label = f"log10({y_label})"

        if plot_source_count:
            plot_data["n_sources"] = plot_data[label].map(self._n_sources)
            ax = boxswarm_and_source_count(plot_data=plot_data, x=label, y=y_label, hue=hue, **plot_kwargs)
            if return_data:
                return ax, plot_data
            return ax

        ax = box_swarm_plot(data=plot_data, x=label, y=y_label, hue=hue, **plot_kwargs)
        if return_data:
            return ax, plot_data
        return ax

    def internal_performance(
        self,
        sample_n: int = 10000,
        resamples: int = 10,
        metrics: Optional[List[Union[str, InternalMetric]]] = None,
        features: Optional[List[str]] = None,
        labels: Union[Iterable, str] = "cluster_label",
        verbose: bool = True,
    ):
        if isinstance(labels, list) or isinstance(labels, np.ndarray):
            self.data["tmp"] = labels
            labels = "tmp"

        features = features or self.features
        metrics = init_internal_metrics(metrics=metrics)
        results = defaultdict(list)

        for _ in progress_bar(range(resamples), verbose=verbose, total=resamples):
            try:
                sampled_df = self.data.groupby("sample_id").sample(n=sample_n)
            except ValueError:
                raise ValueError(
                    "Cannot take a sample bigger than number of events. Either set 'replace' as True, if "
                    "grouping by sample_id/cluster_label check smallest sample/cluster size, or increase "
                    "the sample size."
                )
            for m in metrics:
                results[m.name].append(m(data=sampled_df, features=features, labels=sampled_df[labels].values))
        if "tmp" in self.data.columns.values:
            self.data.drop("tmp", axis=1, inplace=True)
        return pd.DataFrame(results)

    def performance(
        self,
        metrics: Optional[List[Union[str, InternalMetric]]] = None,
        sample_n: int = 10000,
        resamples: int = 10,
        features: Optional[List[str]] = None,
        labels: Union[Iterable, str] = "cluster_label",
        plot: bool = True,
        verbose: bool = True,
        col_wrap: int = 2,
        figure_kwargs: Optional[Dict] = None,
        **plot_kwargs,
    ):
        if sample_n > self.data.shape[0]:
            raise ValueError(f"sample_n is greater than the total number of events: {sample_n} > {self.data.shape[0]}")
        features = features or self.features
        metrics = init_internal_metrics(metrics=metrics)
        results = defaultdict(list)
        if isinstance(labels, list) or isinstance(labels, np.ndarray):
            self.data["tmp"] = labels
            labels = "tmp"

        for _ in progress_bar(range(resamples), verbose=verbose, total=resamples):
            df = self.data.sample(n=sample_n)
            for m in metrics:
                results[m.name].append(m(data=df, features=features, labels=df[labels].values))
        if "tmp" in self.data.columns.values:
            self.data.drop("tmp", axis=1, inplace=True)
        if plot:
            figure_kwargs = figure_kwargs or {}
            figure_kwargs["figsize"] = figure_kwargs.get("figure_size", (10, 10))
            fig = ColumnWrapFigure(n=len(metrics), col_wrap=col_wrap, **figure_kwargs)
            for i, (m, data) in enumerate(results.items()):
                box_swarm_plot(
                    data=pd.DataFrame({"Method": [m] * len(data), "Score": data}),
                    x="Method",
                    y="Score",
                    ax=fig.add_wrapped_subplot(),
                    **plot_kwargs,
                )
            return results, fig
        return results

    def k_performance(
        self,
        max_k: int,
        cluster_n_param: str,
        method: Union[str, ClusterMethod],
        metric: InternalMetric,
        overwrite_features: Optional[List[str]] = None,
        sample_id: Optional[str] = None,
        clustering_params: Optional[Dict] = None,
    ):
        clustering_params = clustering_params or {}

        overwrite_features = overwrite_features or self.features
        features = remove_null_features(self.data, features=overwrite_features)
        data = self.data.copy()
        if sample_id is not None:
            data = data[data.sample_id == sample_id].copy()

        ylabel = metric.name
        x = []
        y = []
        for k in progress_bar(np.arange(1, max_k + 1, 1)):
            df = data.copy()
            clustering_params[cluster_n_param] = k
            method = build_clustering_method(method=method, verbose=self.verbose, **clustering_params)
            df = method.cluster(data=df, features=features)
            x.append(k)
            y.append(metric(data=df, features=features, labels=df["cluster_label"]))
        ax = sns.lineplot(x=x, y=y, markers=True)
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        return ax

    def silhouette_analysis(
        self,
        n: int = 5000,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[int, int]] = (7.5, 7.5),
        xlim: Tuple[int, int] = (-1, 1),
    ):
        data = self.data.sample(n=n)
        return silhouette_analysis(data=data, features=self.features, ax=ax, figsize=figsize, xlim=xlim)

    def merge_clusters(
        self,
        k_range: Optional[Iterable[int]] = None,
        summary: Union[str, Callable] = "median",
        cluster_method: Optional[ClusterMethod] = None,
        **kwargs,
    ):
        if summary == "median":
            data = self.data.groupby(["cluster_label"])[self.features].median()
        elif summary == "mean":
            data = self.data.groupby(["cluster_label"])[self.features].median()
        else:
            data = self.data.groupby(["cluster_label"])[self.features].apply(summary)

        cluster_method = cluster_method or AgglomerativeClustering()
        if k_range is None:
            k_range = [2, math.ceil(self.data.cluster_label.nunique() / 2)]
        kconsensus = KConsensusClustering(
            clustering_klass=cluster_method, smallest_cluster_n=k_range[0], largest_cluster_n=k_range[1], **kwargs
        )
        data["cluster_label"] = kconsensus.fit_predict(data=data.values)
        data.index.name = "original_cluster_label"
        data.reset_index(drop=False, inplace=True)
        mapping = {o: n for o, n in data[["original_cluster_label", "cluster_label"]].values}
        self.data.cluster_label = self.data.cluster_label.replace(mapping)
        return self

    def remove_outliers(
        self, lower_quantile: float = 0.001, upper_quantile: float = 0.999, features: Optional[List[str]] = None
    ):
        features = features or self.features
        outliers = []
        for var in features:
            x = self.data[var] > self.data[var].quantile(upper_quantile)
            outliers = self.data[x].index.tolist() + outliers
            x = self.data[var] < self.data[var].quantile(lower_quantile)
            outliers = self.data[x].index.tolist() + outliers
        outliers = list(set(outliers))
        logger.info(
            f"{len(outliers)} events outside the defined bounds will be removed "
            f"(that's {len(outliers)/self.data.shape[0]*100}% of all events)"
        )
        self.data = self.data.loc[~self.data.index.isin(outliers)]
