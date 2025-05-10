import os
import pandas as pd
import torch
from typing import Optional, List, Union, Tuple
from pykrx import stock
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, DataLoader
from scaler import RobustScaler, StandardScaler
from torch import Tensor


class PyKRX():
    def __init__(
            self,
            market: str,
            start_date: str,
            end_date: str,
            interval: str = "d",
            save_path: Optional[str] = None
    ) -> None:
        # Attributes
        self.market = market
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.save_path = save_path

    def get_data(self) -> None:
        path = os.path.join(self.save_path, self.market)

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        tickers = stock.get_index_ticker_list(market=self.market)
        for ticker in tqdm(tickers):
            try:
                df = stock.get_index_ohlcv(
                    self.start_date,
                    self.end_date,
                    ticker,
                    self.interval
                )
                df.to_csv(os.path.join(path, f"{ticker}.csv"), encoding="utf-8")
            except Exception as e:
                print(f"Failed to download {ticker}: {e}")
                continue


class KRXDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            in_feature_list: List[str],
            out_feature_list: List[str],
            window_size: int = 10,
            forecast_size: int = 1,
            train_ratio: float = 0.8,
            scaler: Union[RobustScaler, StandardScaler] = RobustScaler()
    ) -> None:
        # Read and preprocess data
        data = pd.read_csv(data_path, parse_dates=["날짜"], index_col="날짜")
        data = data.sort_index(ascending=True).dropna()
        self.data = data

        # Create sliding window indices
        total_len = len(data)
        self.indices = [
            (
                i,
                i + window_size,
                i + window_size + forecast_size
            )
            for i in range(total_len - window_size - forecast_size + 1)
        ]
        if not self.indices:
            raise ValueError(
                f"Insufficient data length: {total_len} "
                f"for window_size {window_size} and forecast_size {forecast_size}."
            )
        
        # Fit scaler on training data
        train_len = int(len(self.indices) * train_ratio)
        train_last_idx = self.indices[train_len - 1][1]

        inputs = self._data_values(in_feature_list)
        outputs = self._data_values(out_feature_list)

        self.in_scaler = scaler(inputs[:train_last_idx])
        self.out_scaler = scaler(outputs[:train_last_idx])
        self.in_features = self.in_scaler.fit_transform(inputs)
        self.out_features = self.out_scaler.fit_transform(outputs)

    def _data_values(self, list: List[str]) -> pd.DataFrame:
        return self.data[list].values

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        start_idx, end_idx, forecast_end_idx = self.indices[idx]
        x = torch.tensor(self.in_features[start_idx:end_idx], dtype=torch.float)
        y = torch.tensor(self.out_features[end_idx:forecast_end_idx], dtype=torch.float)
        return x, y


class KRXDataLoader():
    def __init__(
            self,
            dataset: KRXDataset,
            train_ratio: float = 0.8,
            valid_ratio: float = 0.1,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            pin_memory: bool = False
    ) -> None:
        # Attributes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Compute length of each subset
        dataset_len = len(dataset)
        train_len = int(dataset_len * train_ratio)
        valid_len = train_len + int(dataset_len * valid_ratio)

        # Create subsets
        self.train_set = Subset(dataset, range(0, train_len))
        self.valid_set = Subset(dataset, range(train_len, valid_len))
        self.test_set = Subset(dataset, range(valid_len, dataset_len))

    def get_train_loader(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        val_loader = DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        return train_loader, val_loader, test_loader

