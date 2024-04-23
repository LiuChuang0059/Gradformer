import os
import random
import time
import numpy as np
import torch
from ogb.graphproppred import Evaluator
from tqdm import tqdm
import torch.nn as nn
from config import gene_arg, set_config

import warnings
import torch.nn.functional as F
from datetime import datetime

from criterion.weight_loss import weighted_cross_entropy, accuracy_sbm
from data_utils.load_data import load_data
from data_utils.checkpoint import results_to_file
from model.Gradformer import Gradformer
from optimizer.ultra_optimizer import get_scheduler


def train(loader, task):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        node_num = data.sph.shape[-1]
        data.sph = data.sph.reshape(-1, node_num, node_num)
        optimizer.zero_grad()
        if data.edge_attr is None:
            data.edge_attr = data.edge_index.new_zeros(data.edge_index.shape[1])
        out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.sph)
        # get train loss
        if task == "multi_label":
            bce_loss = nn.BCEWithLogitsLoss()
            target = data.y.clone()
            is_labeled = target == target
            loss = bce_loss(out[is_labeled], target[is_labeled])
        elif task == 'multi_class':
            loss = F.cross_entropy(out, data.y)
        elif task == 'binary_class':
            loss = F.binary_cross_entropy_with_logits(out, data.y.float())
        elif task == 'multi_weight_class':
            loss = weighted_cross_entropy(out, data.y)
        else:
            loss = (out.squeeze() - data.y).abs().mean()  # MAE

        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()

    return total_loss / len(loader.dataset)


def test(loader, criterion, split):
    model.eval()

    y_true = []
    y_pred = []
    correct = 0
    sample = 0
    mae = 0
    for data in tqdm(loader, desc=split, leave=False):
        data = data.to(device)
        node_num = data.sph.shape[-1]
        data.sph = data.sph.reshape(-1, node_num, node_num)
        if data.edge_attr is None:
            data.edge_attr = data.edge_index.new_zeros(data.edge_index.shape[1])
        out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.sph)
        if criterion == 'accuracy':
            out = out.max(dim=1)[1]
            correct += out.eq(data.y).sum().item()
            sample += data.y.shape[0]
        if criterion == 'ogbg':
            y_true.append(data.y.view(out.shape).detach().cpu())
            y_pred.append(out.detach().cpu())
        if criterion == 'mae':
            mae += (out.squeeze() - data.y).abs().sum().item()
        if criterion == 'accuracy_sbm':
            y_true.append(data.y.detach().cpu())
            y_pred.append(out.detach().cpu())
    # select criterion
    if criterion == 'accuracy':
        result = correct / sample
    elif criterion == 'ogbg':
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        evaluator = Evaluator(args.dataset)
        _, result = evaluator.eval(input_dict).popitem()
    elif criterion == 'mae':
        result = mae / len(loader.dataset)
    elif criterion == 'accuracy_sbm':
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        result = accuracy_sbm(y_pred, y_true)
    else:
        raise ValueError("Invalid criterion. Supported criterion: 'accuracy', 'ogbg', 'mae'")

    return result


if __name__ == '__main__':
    now = datetime.now()
    now = now.strftime("%m_%d-%H_%M_%S")
    warnings.filterwarnings('ignore')

    args = gene_arg()
    set_config(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if torch.cuda.is_available():
            # cudnn.deterministic = True
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    print(args)

    run_name = f"{args.dataset}"
    args.save_path = f"exps/{run_name}-{now}"
    os.makedirs(os.path.join(args.save_path, str(args.seed)), exist_ok=True)

    device = torch.device('cuda:{}'.format(args.devices) if torch.cuda.is_available() else 'cpu')

    print(f"###Start with run_id : {args.run}###")
    if args.dataset == 'ZINC':
        best_val, final_test = 1e5, 1e5
    else:
        best_val, final_test = 0, 0

    train_loader, val_loader, test_loader, num_tasks, num_features, num_edge_features = load_data(args)

    model = Gradformer(args=args, node_dim=num_features, edge_dim=num_edge_features, num_tasks=num_tasks,
                       mpnn=args.mpnn, pool=args.pool).to(device)

    if args.dataset in ['NCI1', 'IMDB-BINARY', 'COLLAB', 'PROTEINS', 'MUTAG']:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=args.weight_decay)
        scheduler = get_scheduler(optimizer, args.warmup_epoch, args.epochs * len(train_loader), -1)

    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_param}")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train(train_loader, args.task)
        val_res = test(val_loader, args.criterion, "val")
        test_res = test(test_loader, args.criterion, "test")
        state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}

        if args.dataset == 'ZINC':
            if best_val > val_res:
                best_val = val_res
                # final_test = test_acc
                torch.save(state_dict, os.path.join(args.save_path, str(args.seed), "best_model.pt"))
        else:
            if best_val < val_res:
                best_val = val_res
                # final_test = test_acc
                torch.save(state_dict, os.path.join(args.save_path, str(args.seed), "best_model.pt"))

        if scheduler is not None:
            scheduler.step()

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_res:.4f}, '
              f'Test: {test_res:.4f}')

    # load best model
    state_dict = torch.load(os.path.join(args.save_path, str(args.seed), "best_model.pt"))
    model.load_state_dict(state_dict["model"])

    best_val_res = test(val_loader, args.criterion, "best_val")
    best_test_res = test(test_loader, args.criterion, "best_test")

    print(f'Val: {best_val_res:.4f}, '
          f'Test: {best_test_res:.4f}')

    print(f"Total time elapsed: {time.time() - start_time:.4f}s")
    results_to_file(args, best_val_res, best_test_res)
