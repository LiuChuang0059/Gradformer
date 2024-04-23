import os
import pickle

import torch
from torch.nn import LeakyReLU, ReLU
import torch.nn.functional as F
from tqdm import tqdm


def process_hop(sph, gamma, hop, slope=0.1):
    leakyReLU = LeakyReLU(negative_slope=slope)
    sph = sph.unsqueeze(1)
    sph = sph - hop
    sph = leakyReLU(sph)
    sp = torch.pow(gamma, sph)
    return sp


def process_sph(args, data, split=None):
    os.makedirs(f'./sph', exist_ok=True)
    if split is None:
        file = f'./sph/{args.dataset}.pkl'
    else:
        file = f'./sph/{args.dataset}_{split}.pkl'
    if not os.path.exists(file):
        print('pre-process start!')
        progress_bar = tqdm(desc='pre-processing Data', total=len(data), ncols=70)
        for i in range(len(data)):
            data.process(i)
            progress_bar.update(1)
        progress_bar.close()
        pickle.dump(data.sph, open(file, 'wb'))
        print('pre-process down!')
    else:
        data.sph = pickle.load(open(file, 'rb'))
        print('load sph down!')