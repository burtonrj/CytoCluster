import logging
from typing import Dict, Union
from typing import Optional

import numpy as np
import pandas as pd

from cytocluster.methods.wrapper import ClusterMethod
from cytotools.dimension_reduction import DimensionReduction
from cytotools.sampling import sample_dataframe
from cytotools.sampling import upsample_knn

logger = logging.getLogger(__name__)


class LatentClustering:
    def __init__(
        self,
        cluster_method: ClusterMethod,
        dimension_reduction: DimensionReduction,
        sample_size: int = 10000,
        downsample_method: str = "uniform",
        downsample_kwargs: Optional[Dict] = None,
        upsampling_kwargs: Optional[Dict] = None,
        n_components: Optional[int] = None,
    ):
        self.reducer = dimension_reduction
        self.n_components = n_components
        self.sample_size = sample_size
        self.cluster_method = cluster_method
        self.downsample_method = downsample_method
        self.downsample_kwargs = downsample_kwargs or {}
        self.upsampling_kwargs = upsampling_kwargs or {}

    def fit_predict(self, data: pd.DataFrame) -> np.ndarray:
        logger.info("Down-sampling data")
        sample = sample_dataframe(
            data=data, sample_size=self.sample_size, method=self.downsample_method, **self.downsample_kwargs
        )
        logger.info("Fitting dimension reduction model on sample")
        sample = self.reducer.fit_transform(data=sample, features=sample.columns.tolist())
        features = [x for x in sample.columns if self.reducer.name in x]
        logger.info(f"Clustering in down-sampled embedded space ({features})")
        labels = self.cluster_method.global_clustering(data=sample, features=features)["cluster_label"].values
        logger.info("Up-sampling to original space")
        labels = upsample_knn(
            sample=sample,
            original_data=data,
            labels=labels,
            features=data.columns.tolist(),
            **self.upsampling_kwargs,
        )
        logger.info("Clustering complete")
        return labels

