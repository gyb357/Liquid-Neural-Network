import os
import pandas as pd
import torch
from typing import Optional, List, Tuple, Union
from tqdm import tqdm
from pykrx import stock

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
        path = f"{self.save_path}/{self.market}"

        # Check if the directory exists, if not create it
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # Get the list of tickers in the market
        for ticker in tqdm(stock.get_index_ticker_list(market=self.market)):
            try:
                df = stock.get_index_ohlcv(
                    self.start_date,
                    self.end_date,
                    ticker,
                    self.interval
                )
                df.to_csv(f"{path}/{ticker}.csv", encoding="utf-8")
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
            train_ratio: float = 0.8
    ) -> None:
        # Attributes
        self.window_size = window_size
        self.forecast_size = forecast_size

        # Data load and preprocessing
        data = pd.read_csv(data_path, parse_dates=['날짜'], index_col='날짜')
        data = data.sort_index(ascending=True).dropna()
        self.data = data

        # Sliding window indexing
        total_length = len(data)
        self.indices = [
            (
                i,
                i + window_size,
                i + window_size + forecast_size
            )
            for i in range(total_length - window_size - forecast_size + 1)
        ]
        if len(self.indices) == 0:
            raise ValueError(f"Insufficient data length: {total_length} for window_size {window_size} and forecast_size {forecast_size}.")
        
        # Scaler fitting
        train_length = int(len(self.indices) * train_ratio)
        max_origin_idx = self.indices[train_length - 1][1]

        train_data_in = data[in_feature_list].values[:max_origin_idx]
        train_data_out = data[out_feature_list].values[:max_origin_idx]
        
        self.in_scaler = StandardScaler(train_data_in)
        self.out_scaler = StandardScaler(train_data_out)
        self.in_features = self.in_scaler.fit_transform(data[in_feature_list].values)
        self.out_features = self.out_scaler.fit_transform(data[out_feature_list].values)

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        i, j, k = self.indices[idx]
        x = torch.tensor(self.in_features[i:j], dtype=torch.float)
        y = torch.tensor(self.out_features[j:k], dtype=torch.float)
        return x, y


class KRXDataLoader():
    def __init__(
            self,
            dataset: KRXDataset,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
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

        # Extract indices
        dataset_length = len(dataset)
        train_end = int(dataset_length * train_ratio)
        val_end = train_end + int(dataset_length * val_ratio)

        # Create subsets
        self.train_set = Subset(dataset, range(0, train_end))
        self.val_set = Subset(dataset, range(train_end, val_end))
        self.test_set = Subset(dataset, range(val_end, dataset_length))

    def get_dataloader(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        val_loader = DataLoader(
            self.val_set,
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

