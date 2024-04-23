import torch
from torch.utils.data import Dataset
import pyximport
import warnings
import numpy as np

from utils.commute import adj_pinv
from sklearn.metrics.pairwise import cosine_similarity
pyximport.install(setup_args={'include_dirs': np.get_include()})
from utils import algos

warnings.filterwarnings("ignore")


class NewDataset(Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.dataset = original_dataset
        self.sph = []

    def __getitem__(self, index):
        data = self.dataset[index]
        data['sph'] = self.sph[index]
        return data

    def __len__(self):
        return len(self.dataset)

    def process(self, index):
        data = self.dataset[index]
        N = data.x.shape[0]
        adj = torch.zeros([N, N])
        adj[data.edge_index[0, :], data.edge_index[1, :]] = True
        sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
        self.sph.append(sp)
        """
        self.sph.append(adj_pinv(data, topk=10)) # ectd
        mat = cosine_similarity(data.x, data.x)
        mat = 1 / mat
        self.sph.append(mat) # cosine similarity
        """






