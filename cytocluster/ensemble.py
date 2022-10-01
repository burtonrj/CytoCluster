import logging
from collections import Counter
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import hdmedians as hd
import numpy as np
import pandas as pd
import seaborn as sns
from ensembleclustering.ClusterEnsembles import cluster_ensembles
from joblib import delayed
from joblib import Parallel
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial import distance
from sklearn.base import ClusterMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from cytoplots.general import box_swarm_plot
from cytocluster.base import Clustering
from cytocluster.methods.wrapper import ClusterMethod
from cytocluster.methods.utils import build_clustering_method
from cytocluster.base import simpson_di
from cytocluster.metrics import comparison_matrix
from cytocluster.metrics import InternalMetric
from cytotools.feedback import progress_bar

logger = logging.getLogger(__name__)


def distortion_score(data: pd.DataFrame, features: List[str], clusters: List[str], metric: str = "euclidean") -> Dict:
    score = {}
    clusters = [c for c in clusters if c in data.columns]
    for c in clusters:
        df = data[data[c] == 1]
        if df.shape[0] == 0:
            continue
        center = df[features].mean().values
        distances = pairwise_distances(df[features], center.reshape(1, -1), metric=metric)
        score[c] = (distances ** 2).sum() / df.shape[0]
    return score


def majority_vote_with_distortion_score_weighting(
    row: pd.Series,
    cluster_column_names: List[str],
    consensus_clusters: pd.DataFrame,
    distortion_score_dict: Dict[str, Dict[str, float]],
):
    clusters = row[cluster_column_names].replace({0: None}).dropna().index.tolist()
    clusters = [c for c in clusters if c in consensus_clusters.index.tolist()]

    consensus_cluster_score = []
    for cid, cc_data in consensus_clusters.loc[clusters].groupby("cluster_label"):
        consensus_cluster_score.append(
            [
                cid,
                cc_data.shape[0]
                / (sum([distortion_score_dict.get(i) for i in cc_data.index.values]) / cc_data.shape[0]),
                cc_data.shape[0],
            ]
        )
    if len(set([x[1] for x in consensus_cluster_score])) == 1:
        # All consensus clusters have equal scores, return consensus cluster label with the greatest number of votes
        return sorted(consensus_cluster_score, key=lambda x: x[2])[::-1][0][0]
    # Return consensus cluster label with the highest score
    return sorted(consensus_cluster_score, key=lambda x: x[1])[::-1][0][0]


def majority_vote_with_distance_weighting(
    row: pd.Series,
    cluster_column_names: List[str],
    consensus_clusters: pd.DataFrame,
    features: List[str],
    metric: str = "cityblock",
):
    clusters = row[cluster_column_names].replace({0: None}).dropna().index.tolist()
    clusters = [c for c in clusters if c in consensus_clusters.index.tolist()]

    consensus_cluster_score = []
    for cid, cc_data in consensus_clusters.loc[clusters].groupby("cluster_label"):
        scores = sum([getattr(distance, metric)(row[features], cc_row[features]) for _, cc_row in cc_data.iterrows()])
        consensus_cluster_score.append([cid, cc_data.shape[0] / (scores / cc_data.shape[0]), cc_data.shape[0]])
    if len(set([x[1] for x in consensus_cluster_score])) == 1:
        # All consensus clusters have equal scores, return consensus cluster label with the greatest number of votes
        return sorted(consensus_cluster_score, key=lambda x: x[2])[::-1][0][0]
    # Return consensus cluster label with the highest score
    return sorted(consensus_cluster_score, key=lambda x: x[1])[::-1][0][0]


def majority_vote(row: pd.Series, cluster_column_names: List[str], consensus_clusters: pd.DataFrame):
    clusters = row[cluster_column_names].replace({0: None}).dropna().index.tolist()
    clusters = [c for c in clusters if c in consensus_clusters.index.tolist()]
    consensus_cluster_labels = Counter([consensus_clusters.loc[cid, "cluster_label"] for cid in clusters])
    score = 0
    winner = None

    for label, count in consensus_cluster_labels.items():
        if count == score:
            logger.warning(
                f"{label} and {winner} have equal scores, observation will be assigned to {winner}, "
                f"to avoid this provide weights for cluster voting."
            )
        if count > score:
            score = count
            winner = label
    return winner


class EnsembleClustering(Clustering):
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
        super().__init__(data, features, verbose, random_state, n_sources, pre_embedded, labels)
        self.original_cluster_labels = labels
        cluster_membership = pd.concat(
            [
                pd.get_dummies(x, prefix=cluster_id, prefix_sep="_")
                for cluster_id, x in self.original_cluster_labels.items()
            ],
            axis=1
        )
        self.data = pd.concat([self.data, cluster_membership], axis=1)
        self.clusters = cluster_membership.columns.tolist()
        self._cluster_weights = {}
        self._event_distance_to_cluster_center = {}

    @property
    def cluster_groups(self) -> Dict:
        cluster_groups = defaultdict(list)
        for cluster in self.clusters:
            prefix = cluster.split("_")[0]
            cluster_groups[prefix].append(cluster)
        return cluster_groups

    @property
    def cluster_sizes(self) -> pd.Series:
        return self.data[self.clusters].sum(axis=0)

    @property
    def scaled_data(self):
        event_data = self.data.copy()
        event_data[self.features] = MinMaxScaler().fit_transform(event_data[self.features].T).T
        return event_data

    @property
    def cell_cluster_assignments(self) -> pd.DataFrame:
        data = self.data[self.clusters].reset_index(drop=False).rename(
            columns={"Index": "cell_id", "index": "cell_id"}
        )
        data = data.melt(id_vars="cell_id", value_name="assignment", var_name="cluster")
        data = data[data.assignment == 1]
        return data.drop("assignment", axis=1)

    def _reconstruct_labels(self, encoded: bool = False):
        labels = {}
        for prefix, clusters in self.cluster_groups.items():
            labels[prefix] = self.data[clusters].idxmax(axis=1)
        if encoded:
            return np.array([LabelEncoder().fit_transform(x) for x in labels.values()])
        return labels

    def _check_for_cluster_parents(self):
        for prefix, clusters in self.cluster_groups.items():
            if not (self.data[clusters].sum(axis=1) == 1).all():
                logger.warning(
                    f"Some observations are assigned to multiple clusters under the cluster prefix {prefix},"
                    f" either ensure cluster prefixes are unique to a cluster solution or remove parent "
                    f"populations with the 'ignore_clusters' method."
                )

    def ignore_clusters(self, clusters: List[str]):
        clusters = [x for x in clusters if x in self.clusters]
        self.clusters = [x for x in self.clusters if x not in clusters]
        self.data.drop(clusters, axis=1, inplace=True)
        self._cluster_weights = {}

    def remove_outliers(
        self, lower_quantile: float = 0.001, upper_quantile: float = 0.999, features: Optional[List[str]] = None
    ):
        super().remove_outliers(lower_quantile=lower_quantile, upper_quantile=upper_quantile, features=features)
        self.remove_small_clusters(n=5)

    def remove_small_clusters(self, n: int = 100):
        small_clusters = self.cluster_sizes[self.cluster_sizes < n].index.tolist()
        self.ignore_clusters(clusters=small_clusters)

    def simpsons_diversity_index(self, cell_identifier: str = "sample_id", groupby=None) -> pd.DataFrame:
        si_scores = {}
        for cluster in self.clusters:
            df = self.data[self.data[cluster] == 1]
            si_scores[cluster] = simpson_di(df[cell_identifier].value_counts().to_dict())
        return pd.DataFrame(si_scores, index=["SimpsonIndex"]).T

    def plot_simpsons_diversity_index(self, *args, **kwargs):
        raise ValueError("plot_simpsons_diversity_index not implemented for EnsembleClustering")

    def compute_cluster_centroids(self, diversity_threshold: Optional[float] = None, scale: bool = True):
        cluster_geometric_median = []
        for cluster in self.clusters:
            cluster_data = self.data[self.data[cluster] == 1][self.features].T.values
            x = np.array(hd.geomedian(cluster_data)).reshape(-1, 1)
            x = pd.DataFrame(x, columns=[cluster], index=self.features)
            cluster_geometric_median.append(x.T)
        centroids = pd.concat(cluster_geometric_median)
        if scale:
            centroids[self.features] = MinMaxScaler().fit_transform(centroids[self.features])
        if diversity_threshold:
            si_scores = self.simpsons_diversity_index()
            si_scores = si_scores[si_scores.SimpsonIndex <= diversity_threshold]
            centroids = centroids.loc[si_scores.index]
        return centroids

    def cluster_distortion_score(
        self, plot: bool = True, n_jobs=-1, distortion_metric: str = "euclidean", verbose: bool = True, **plot_kwargs
    ):
        data = self.scaled_data
        if self._cluster_weights.get("metric", None) != distortion_metric:
            with Parallel(n_jobs=n_jobs) as parallel:
                sample_ids = data.sample_id.unique()
                weights = parallel(
                    delayed(distortion_score)(
                        data=data[data.sample_id == sid],
                        features=self.features,
                        metric=distortion_metric,
                        clusters=self.clusters,
                    )
                    for sid in progress_bar(sample_ids, verbose=verbose)
                )
                self._cluster_weights["metric"] = distortion_metric
                self._cluster_weights["weights"] = {sid: w for sid, w in zip(sample_ids, weights)}
        if plot:
            plot_kwargs = plot_kwargs or {}
            plot_df = (
                pd.DataFrame(self._cluster_weights["weights"])
                .T.melt(var_name="Cluster", value_name="Distortion score")
                .sort_values("Distortion score")
            )
            ax = box_swarm_plot(data=plot_df, x="Cluster", y="Distortion score", **plot_kwargs)
            return self._cluster_weights["weights"], ax
        return self._cluster_weights["weights"], None

    def _consensus_clusters_count_sources(self, consensus_results: pd.DataFrame):
        self._n_sources = {}
        for consensus_label, clusters in consensus_results.groupby("cluster_label"):
            self._n_sources[consensus_label] = len(set([c.split("_")[0] for c in clusters.index.unique()]))

    def similarity_matrix(self, diversity_threshold: Optional[float] = None, directional: bool = True):

        if diversity_threshold:
            si_scores = self.simpsons_diversity_index()
            si_scores = si_scores[si_scores.SimpsonIndex <= diversity_threshold]
            clusters = si_scores.index.values
        else:
            clusters = self.clusters

        matrix = np.zeros((len(clusters), len(clusters)))

        for i, cluster_i_id in enumerate(clusters):
            cluster_i_data = set(self.data[self.data[cluster_i_id] == 1].index.values)

            for j, cluster_j_id in enumerate(clusters):
                cluster_j_data = set(self.data[self.data[cluster_j_id] == 1].index.values)
                intersect = cluster_i_data.intersection(cluster_j_data)

                if directional:
                    if len(cluster_i_data) < len(cluster_j_data):
                        matrix[i, j] = len(intersect) / len(cluster_i_data)
                    else:
                        matrix[i, j] = len(intersect) / len(cluster_j_data)
                else:
                    union = cluster_i_data.union(cluster_j_data)
                    matrix[i, j] = len(intersect) / len(union)

        return pd.DataFrame(matrix, index=clusters, columns=clusters)

    def clustered_similarity_heatmap(
        self,
        directional: bool = True,
        diversity_threshold: Optional[float] = None,
        figsize: Tuple[int, int] = (10, 12),
        dendrogram_dimensions: Tuple[float, float, float, float] = (0.3, 0.9, 0.6, 0.2),
        similarity_heatmap_dimensions: Tuple[float, float, float, float] = (0.3, 0.5, 0.6, 0.4),
        centroid_heatmap_dimensions: Tuple[float, float, float, float] = (0.3, 0.08, 0.6, 0.4),
        method: str = "average",
        metric: str = "euclidean",
        cmap: str = "coolwarm",
        xticklabels: bool = False,
        scale: bool = True,
    ):
        similarity_matrix = self.similarity_matrix(diversity_threshold=diversity_threshold, directional=directional)
        centroids = self.compute_cluster_centroids(diversity_threshold=diversity_threshold, scale=scale)

        fig = plt.figure(figsize=figsize)

        # Dendrogram
        dendrogram_ax = fig.add_axes(dendrogram_dimensions)
        linkage_matrix = linkage(similarity_matrix.values, method=method, metric=metric)
        dendro = dendrogram(linkage_matrix, color_threshold=0, above_threshold_color="black")
        dendrogram_ax.set_xticks([])
        dendrogram_ax.set_yticks([])

        # Similarity matrix
        axmatrix_sim = fig.add_axes(similarity_heatmap_dimensions)
        idx = dendro["leaves"]
        d = similarity_matrix.values[idx, :]
        axmatrix_sim.matshow(d[:, idx], aspect="auto", origin="lower", cmap=cmap)
        axmatrix_sim.set_xticks([])
        axmatrix_sim.set_yticks([])

        # Centroid heatmap
        axmatrix_centroids = fig.add_axes(centroid_heatmap_dimensions)
        idx = dendro["leaves"]
        d = centroids.iloc[idx].T
        sns.heatmap(data=d, ax=axmatrix_centroids, cmap=cmap, cbar=False, xticklabels=xticklabels)

        return fig

    def geometric_median_heatmap(
        self,
        method: str = "ward",
        metric: str = "euclidean",
        plot_orientation: str = "vertical",
        diversity_threshold: Optional[float] = None,
        scale: bool = True,
        clusters: Optional[List[str]] = None,
        **kwargs,
    ):
        kwargs = kwargs or {}

        centroids = self.compute_cluster_centroids(diversity_threshold=diversity_threshold, scale=scale)
        if clusters:
            centroids = centroids.loc[clusters]

        if plot_orientation == "horizontal":
            g = sns.clustermap(data=centroids.T, method=method, metric=metric, **kwargs)
        else:
            g = sns.clustermap(data=centroids, method=method, metric=metric, **kwargs)
        return g

    def _vote_filter(self, consensus_results: pd.DataFrame):
        cell_assignments = self.cell_cluster_assignments
        consensus_results = consensus_results[["cluster_label"]].reset_index().rename(columns={"index": "cluster"})
        cell_assignments = (
            cell_assignments.merge(consensus_results, on="cluster")
            .drop("cluster", axis=1)
            .rename(columns={"cluster_label": "cluster"})
        )
        cell_assignments.drop_duplicates(inplace=True)
        consensus_counts = cell_assignments.cell_id.value_counts()
        one_consensus_label = cell_assignments[
            cell_assignments.cell_id.isin(consensus_counts[consensus_counts == 1].index)
        ]
        data = self.scaled_data
        labelled = data.loc[one_consensus_label.cell_id.values].copy()
        unlabelled = data[~data.index.isin(one_consensus_label.cell_id.values)].copy()
        labelled["cluster_label"] = one_consensus_label["cluster"].values
        return labelled, unlabelled

    def _majority_vote(
        self,
        consensus_data: pd.DataFrame,
        vote_weighting_method: str,
        distortion_metric: str,
        distance_to_center_metric: str,
        verbose: bool,
        n_jobs: int,
    ) -> np.ndarray:
        logger.info(f"Assigning clusters by majority vote (vote_weighting_method={vote_weighting_method})")
        labelled, unlabelled = self._vote_filter(consensus_results=consensus_data)
        logger.info(
            f"{round(unlabelled.shape[0] / self.data.shape[0] * 100, 3)}% of events assigned to more than one "
            f"consensus label, resolving by vote"
        )

        if vote_weighting_method == "distortion_score":
            weights = self.cluster_distortion_score(plot=False, distortion_metric=distortion_metric, verbose=verbose)[
                0
            ]
            with Parallel(n_jobs=n_jobs) as parallel:
                labels = parallel(
                    delayed(majority_vote_with_distortion_score_weighting)(
                        row=row,
                        cluster_column_names=self.clusters,
                        consensus_clusters=consensus_data[["cluster_label"]],
                        distortion_score_dict=weights[row["sample_id"]],
                    )
                    for _, row in progress_bar(unlabelled.iterrows(), verbose=verbose, total=unlabelled.shape[0])
                )
        elif vote_weighting_method == "distance_to_center":
            with Parallel(n_jobs=n_jobs) as parallel:
                labels = parallel(
                    delayed(majority_vote_with_distance_weighting)(
                        row=row,
                        cluster_column_names=self.clusters,
                        consensus_clusters=consensus_data,
                        metric=distance_to_center_metric,
                        features=self.features,
                    )
                    for _, row in progress_bar(unlabelled.iterrows(), verbose=verbose, total=unlabelled.shape[0])
                )
        else:
            with Parallel(n_jobs=n_jobs) as parallel:
                labels = parallel(
                    delayed(majority_vote)(
                        row=row,
                        cluster_column_names=self.clusters,
                        consensus_clusters=consensus_data[["cluster_label"]],
                    )
                    for _, row in progress_bar(unlabelled.iterrows(), verbose=verbose, total=unlabelled.shape[0])
                )
        unlabelled["cluster_label"] = labels
        self.data["cluster_label"] = pd.concat([labelled, unlabelled])["cluster_label"]
        return labels

    def geowave(
        self,
        method: Union[str, ClusterMethod, ClusterMixin] = "k_consensus",
        method_kwargs: Optional[Dict] = None,
        n_jobs: int = -1,
        verbose: bool = True,
        diversity_threshold: Optional[float] = None,
        vote_weighting_method: Optional[str] = "distance_to_center",
        distortion_metric: str = "cityblock",
        distance_to_center_metric: str = "cityblock",
        return_labels: bool = False,
        scale: bool = True,
    ):
        method_kwargs = method_kwargs or {}
        method = build_clustering_method(method=method, verbose=self.verbose, **method_kwargs)
        self._check_for_cluster_parents()

        logger.info("Calculating geometric median of each cluster")
        consensus_data = self.compute_cluster_centroids(diversity_threshold=diversity_threshold, scale=scale)

        logger.info("Performing consensus clustering on geometric medians")
        consensus_data = method.global_clustering(data=consensus_data, features=self.features)

        logger.info(
            f"Generated {consensus_data.cluster_label.nunique()} consensus clusters: {consensus_data.cluster_label.unique()}"
        )
        labels = self._majority_vote(
            consensus_data=consensus_data,
            vote_weighting_method=vote_weighting_method,
            distortion_metric=distortion_metric,
            distance_to_center_metric=distance_to_center_metric,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        logger.info("Consensus clustering complete!")
        if return_labels:
            return labels
        return self

    def simwave(
        self,
        method: Union[str, ClusterMethod, ClusterMixin] = "k_consensus",
        method_kwargs: Optional[Dict] = None,
        n_jobs: int = -1,
        verbose: bool = True,
        diversity_threshold: Optional[float] = None,
        vote_weighting_method: Optional[str] = "distance_to_center",
        distortion_metric: str = "cityblock",
        distance_to_center_metric: str = "cityblock",
        return_labels: bool = False,
        directional_similarity: bool = True,
    ):
        method_kwargs = method_kwargs or {}
        method = build_clustering_method(method=method, verbose=self.verbose, **method_kwargs)
        self._check_for_cluster_parents()

        logger.info("Calculating similarity matrix")
        consensus_data = self.similarity_matrix(
            diversity_threshold=diversity_threshold, directional=directional_similarity
        )

        logger.info("Performing consensus clustering on similarity matrix")
        consensus_data = method.global_clustering(data=consensus_data, features=self.clusters)

        logger.info(
            f"Generated {consensus_data.cluster_label.nunique()} consensus clusters: {consensus_data.cluster_label.unique()}"
        )
        labels = self._majority_vote(
            consensus_data=consensus_data,
            vote_weighting_method=vote_weighting_method,
            distortion_metric=distortion_metric,
            distance_to_center_metric=distance_to_center_metric,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        logger.info("Consensus clustering complete!")
        if return_labels:
            return labels
        return self

    def _consensus_count_sources(self, original_labels: List):
        data = self.data.copy()
        data["original_cluster_label"] = original_labels
        for consensus_label, clusters in data.groupby("cluster_label"):
            self._n_sources[consensus_label] = clusters.original_cluster_label.nunique()

    def graph_consensus_clustering(
        self, consensus_method: str, k: int, random_state: int = 42, labels: Optional[List] = None
    ):
        labels = labels if labels is not None else self._reconstruct_labels(encoded=True)
        if consensus_method == "cspa" and self.data.shape[0] > 5000:
            logger.warning("CSPA is not recommended when n>5000, consider a different method")
            self.data["cluster_label"] = cluster_ensembles(labels, nclass=k, solver="cspa")
        elif consensus_method == "hgpa":
            self.data["cluster_label"] = cluster_ensembles(labels, nclass=k, random_state=random_state, solver="hgpa")
        elif consensus_method == "mcla":
            self.data["cluster_label"] = cluster_ensembles(labels, nclass=k, random_state=random_state, solver="mcla")
        elif consensus_method == "hbgf":
            self.data["cluster_label"] = cluster_ensembles(labels, nclass=k, solver="hbgf")
        elif consensus_method == "nmf":
            self.data["cluster_label"] = cluster_ensembles(labels, nclass=k, random_state=random_state, solver="nmf")
        else:
            raise ValueError("Invalid consensus method, must be one of: cspa, hgpa, mcla, hbgf, or nmf")

    def comparison(self, method: str = "adjusted_mutual_info", **kwargs):
        kwargs["figsize"] = kwargs.get("figsize", (10, 10))
        kwargs["cmap"] = kwargs.get("cmap", "coolwarm")
        data = comparison_matrix(cluster_labels=self._reconstruct_labels(), method=method)
        return sns.clustermap(
            data=data,
            **kwargs,
        )

    def smallest_cluster_n(self):
        return self.cluster_sizes.sort_values().index[0], self.cluster_sizes.sort_values().iloc[0]

    def largest_cluster_n(self):
        return (
            self.cluster_sizes.sort_values(ascending=False).index[0],
            self.cluster_sizes.sort_values(ascending=False).iloc[0],
        )

    def min_k(self):
        return min([len(x) for x in self.cluster_groups.values()])

    def max_k(self):
        return max([len(x) for x in self.cluster_groups.values()])

    def k_performance(
        self,
        k_range: Tuple[int, int],
        consensus_method: str,
        sample_n: int,
        resamples: int,
        random_state: int = 42,
        features: Optional[List[str]] = None,
        metrics: Optional[List[Union[InternalMetric, str]]] = None,
        return_data: bool = True,
        **kwargs,
    ):
        results = []
        for k in range(*k_range):
            logger.info(f"Calculating consensus with k={k}...")
            self.graph_consensus_clustering(consensus_method=consensus_method, k=k, random_state=random_state)
            perf = self.internal_performance(
                metrics=metrics,
                sample_n=sample_n,
                resamples=resamples,
                features=features,
                labels="cluster_label",
                verbose=True,
            )
            perf["K"] = k
            results.append(perf)
        results = pd.concat(results).reset_index(drop=True)
        results = pd.DataFrame(results).melt(id_vars="K", var_name="Metric", value_name="Value")
        return results
