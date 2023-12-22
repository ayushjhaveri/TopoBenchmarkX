import hydra
import torch
from omegaconf import DictConfig

from topobenchmarkx.data.load.loader import AbstractLoader
from topobenchmarkx.data.utils import (
    get_Planetoid_pyg,
    get_TUDataset_pyg,
    load_hypergraph_pickle_dataset,
    load_split,
)


class HypergraphLoader(AbstractLoader):
    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)

    def load(
        self,
    ):
        data = load_hypergraph_pickle_dataset(self.cfg.dataset)
        data = load_split(data, self.cfg.dataset)

        # We need to add checks that:
        # All nodes belong to some edge, in case some not, create selfedge

        return [data]


class PYGLoader(AbstractLoader):
    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)

    def load(self):
        if (
            self.cfg.data_name in ["Cora", "CiteSeer", "PubMed"]
            and self.cfg.data_type == "cocitation"
        ):
            data = get_Planetoid_pyg(cfg=self.cfg)
            data = load_split(data, self.cfg)
            data_lst = [data]
        elif self.cfg.data_name == ["MUTAG", "ENZYMES", "PROTEINS", "COLLAB"]:
            data_lst = get_TUDataset_pyg(cfg=self.cfg)
        else:
            raise NotImplementedError(f"Dataset {self.cfg.data_name} not implemented")

        return data_lst
