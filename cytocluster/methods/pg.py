from typing import Union
import pandas as pd
import numpy as np
import phenograph


class Phenograph:
    def __init__(self, **params):
        params = params or {}
        self.params = params

    def fit_predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        communities, graph, q = phenograph.cluster(data, **self.params)
        return communities
