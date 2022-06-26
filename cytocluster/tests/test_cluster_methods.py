import pandas as pd
from cytotools.dimension_reduction import DimensionReduction
from cytotools.read import polars_to_pandas, read_from_disk
from sklearn.cluster import KMeans, AgglomerativeClustering
import pytest

from cytocluster.methods.utils import build_clustering_method
from . import assets

FEATURES = [
    "CD45RA",
    "CD133",
    "CD19",
    "CD22",
    "CD11b",
    "CD4",
    "CD8",
    "CD34",
    "Flt3",
    "CD20",
    "CXCR4",
    "CD235ab",
    "CD45",
    "CD123",
    "CD321",
    "CD14",
    "CD33",
    "CD47",
    "CD11c",
    "CD7",
    "CD15",
    "CD16",
    "CD44",
    "CD38",
    "CD13",
    "CD3",
    "CD61",
    "CD117",
    "CD49d",
    "HLA-DR",
    "CD64",
    "CD41"
]

@pytest.fixture()
def dummy_data():
    return polars_to_pandas(read_from_disk(f"{assets.__path__._path[0]}/levine32.csv")).sample(n=1000)


@pytest.fixture()
def dummy_data_samples():
    data = []
    for i in range(10):
        df = polars_to_pandas(read_from_disk(f"{assets.__path__._path[0]}/levine32.csv")).sample(n=100)
        df["sample_id"] = f"sample_{i}"
        data.append(df)
    return pd.concat(data).reset_index(drop=True)


@pytest.mark.parametrize(
    "method,params",
    [
        ("phenograph", {"k": 5}),
        ("flowsom", {"sigma": 5.0}),
        ("k_consensus", {"clustering_klass": AgglomerativeClustering(), "smallest_cluster_n": 3, "largest_cluster_n": 10}),
        ("spade", {"min_k": 3, "max_k": 10, "sample_size": 100, "sampling_tree_size": 50}),
        ("parc", {"knn": 5}),
        (AgglomerativeClustering(), {}),
        (KMeans(n_clusters=5), {})
    ]
)
def test_clustering(dummy_data_samples, method, params):
    cluster_method = build_clustering_method(method=method, verbose=True, **params)
    data = cluster_method.cluster(data=dummy_data_samples, features=FEATURES)
    assert "cluster_label" in data.columns.tolist()


@pytest.mark.parametrize(
    "method,params",
    [
        ("phenograph", {"k": 5}),
        ("flowsom", {"sigma": 5.0}),
        ("k_consensus", {"clustering_klass": AgglomerativeClustering(), "smallest_cluster_n": 3, "largest_cluster_n": 10}),
        ("spade", {"min_k": 3, "max_k": 10, "sample_size": 100, "sampling_tree_size": 50}),
        ("parc", {"knn": 5}),
        ("latent",
         {"cluster_method":build_clustering_method(method=KMeans(n_clusters=5), verbose=True),
          "dimension_reduction": DimensionReduction(method="UMAP", n_components=2),
          "sample_size": 500
          }),
        (AgglomerativeClustering(), {}),
        (KMeans(n_clusters=5), {})
    ]
)
def test_global_clustering(dummy_data, method, params):
    cluster_method = build_clustering_method(method=method, verbose=True, **params)
    data = cluster_method.global_clustering(data=dummy_data, features=FEATURES)
    assert "cluster_label" in data.columns.tolist()


@pytest.mark.parametrize(
    "method,params",
    [
        ("phenograph", {"k": 5}),
        ("flowsom", {"sigma": 5.0}),
        ("k_consensus", {"clustering_klass": AgglomerativeClustering(), "smallest_cluster_n": 3, "largest_cluster_n": 10}),
        ("spade", {"min_k": 3, "max_k": 10, "sample_size": 100, "sampling_tree_size": 50}),
        ("parc", {"knn": 5}),
        (AgglomerativeClustering(), {}),
        (KMeans(n_clusters=5), {})
    ]
)
def test_meta_clustering(dummy_data_samples, method, params):
    cluster_method = build_clustering_method(method=method, verbose=True, **params)
    data = cluster_method.cluster(data=dummy_data_samples, features=FEATURES)
    assert "cluster_label" in data.columns.tolist()
    data = cluster_method.meta_clustering(data=data, features=FEATURES)
    assert "meta_label" in data.columns.tolist()
