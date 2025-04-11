import numpy as np


class RobustScaler():
    def __init__(
            self,
            data: np.ndarray
    ) -> None:
        # Attributes
        self.median = np.median(data, axis=0)
        self.iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
        self.iqr[self.iqr == 0] = 1e-6

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.median) / self.iqr
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (data * self.iqr) + self.median


class StandardScaler():
    def __init__(self, data: np.ndarray) -> None:
        # Attributes
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std == 0] = 1e-6

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std) + self.mean

