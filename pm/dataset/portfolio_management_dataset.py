import os.path
import pandas as pd
from typing import List
from glob import glob
import numpy as np

from pm.registry import DATASET


@DATASET.register_module()
class PortfolioManagementDataset():
    def __init__(self,
                 root: str = None,
                 data_path: str = None,
                 stocks_path: str = None,
                 aux_stocks_path: str = None,
                 features_name: List[str] = None,
                 temporals_name: List[str] = None,
                 labels_name: List[str] = None):
        super(PortfolioManagementDataset, self).__init__()

        self.root = root
        self.data_path = data_path
        self.stocks_path = stocks_path
        self.features_name = features_name
        self.temporals_name = temporals_name
        self.labels_name = labels_name

        self.data_path = os.path.join(root, self.data_path)
        self.stocks_path = os.path.join(root, self.stocks_path)
        self.aux_stocks_path = os.path.join(root, aux_stocks_path)

        self.stocks = self._init_stocks()

        self.stocks2id = {stock: i for i, stock in enumerate(self.stocks)}
        self.id2stocks = {i: stock for i, stock in enumerate(self.stocks)}

        self.aux_stocks = self._init_aux_stocks()

        self.aux_stocks[0] = {
            "id":0,
            "type": "all",
            "name": "All",
            "stocks": self.stocks,
            "mask": np.zeros(len(self.stocks)),
        }

        self.stocks_df = self._init_stocks_df()

    def _init_stocks(self):
        print("init stocks...")
        stocks = []
        with open(self.stocks_path) as op:
            for line in op.readlines():
                line = line.strip()
                stocks.append(line)
        print("init stocks success...")
        return stocks

    def _init_stocks_df(self):
        print("init stocks dataframe...")
        stocks_df = []
        for stock in self.stocks:
            path = os.path.join(self.data_path, f"{stock}.csv")
            df = pd.read_csv(path, index_col=0)
            df = df.set_index("Date")
            df = df[self.features_name + self.temporals_name + self.labels_name]
            stocks_df.append(df)
        print("init stocks dataframe success...")
        return stocks_df

    def _init_aux_stocks(self)->dict:
        print("init aux stocks...")
        aux_stocks = {}
        aux_stocks_files = glob(os.path.join(self.aux_stocks_path, "*.txt"))
        for path in aux_stocks_files:
            name = os.path.basename(path).split(".")[0]
            id, name = name.split("_")
            id = int(id)

            with open(path) as op:
                stocks = []
                for line in op.readlines():
                    line = line.strip()
                    stocks.append(line)
            aux_stocks[id] = {
                "name": name,
                "type": "aux",
                "stocks": stocks,
                "num_stocks": len(stocks),
                "mask": np.array([0.0 if stock in stocks else 1.0 for stock in self.stocks])
            }

        for k,v in aux_stocks.items():
            print(f"aux stocks id: {k}, name: {v['name']}, num stocks: {v['num_stocks']}")
        print("init aux stocks success...")
        return aux_stocks