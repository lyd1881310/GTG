from logging import getLogger
import random
import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.loader import DataLoader

from dataloader import get_network, gen_cluster_dataset
from models import GTGModel
from supervise import sup_loss

from config import cfg


logger = getLogger(__name__)


def ortho_loss(sem_reps: torch.Tensor, dom_reps: torch.Tensor):
    cosine_sim = F.cosine_similarity(sem_reps.unsqueeze(1),
                                     dom_reps.unsqueeze(0), dim=2)
    return (cosine_sim ** 2).mean()


def get_dataloaders():
    cities = {'src': cfg['dataset_source'], 'trg': cfg['dataset_target']}
    loaders = dict()
    for dom in ['src', 'trg']:
        network = get_network(cities[dom])
        for phase in ['train', 'valid']:
            label_df = pd.read_csv(os.path.join('data', cities[dom], f'traj/{phase}_label.csv'))
            dataset = gen_cluster_dataset(network, label_df)
            # 加了 Cluster 采样之后, batch_size 设为 1
            loaders[f'{dom}_{phase}'] = DataLoader(dataset, batch_size=1, shuffle=False)
    return loaders


def train(model, optimizer, loaders):
    loss_rcd = {'o_loss': 0, 'sem_pred': 0, 'sem_clf': 0, 'dom_pred': 0, 'dom_clf': 0, 'total': 0}
    # 聚类采样后 batch_size 固定为 1
    batch_num = len(loaders['src_train'])
    for src_batch in loaders['src_train']:
        src_batch = src_batch.to(cfg['device'])
        # 从目标城市随机采一个样本
        trg_batch = loaders['trg_train'].dataset.random_sample().to(cfg['device'])

        sem_reps_src, dom_reps_src, sem_dur, sem_spd, sem_clf_src, dom_dur, dom_spd, dom_clf_src = (
            model.disentangle(src_batch.x, src_batch.edge_index, src_batch.edge_attr))
        # 仅在源城市预测标签
        sem_dur, sem_spd = sem_dur[src_batch.mapping], sem_spd[src_batch.mapping]
        dom_dur, dom_spd = dom_dur[src_batch.mapping], dom_spd[src_batch.mapping]

        sem_reps_trg, dom_reps_trg, _, _, sem_clf_trg, _, _, dom_clf_trg = (
            model.disentangle(trg_batch.x, trg_batch.edge_index, trg_batch.edge_attr))

        # 正交损失项
        o_loss = ortho_loss(sem_reps_src, dom_reps_src) + ortho_loss(sem_reps_trg, dom_reps_trg)

        # 预测损失
        _, _, sem_pred_loss = sup_loss(sem_dur, sem_spd, src_batch.y1, src_batch.y2)
        _, _, dom_pred_loss = sup_loss(dom_dur, dom_spd, src_batch.y1, src_batch.y2)

        # 域分类损失
        label_src = torch.ones_like(sem_clf_src)
        label_trg = torch.zeros_like(sem_clf_trg)
        sem_clf_loss = F.binary_cross_entropy(sem_clf_src, label_src) + F.binary_cross_entropy(sem_clf_trg, label_trg)
        dom_clf_loss = F.binary_cross_entropy(dom_clf_src, label_src) + F.binary_cross_entropy(dom_clf_trg, label_trg)

        # 总损失
        sem_loss = sem_pred_loss + cfg['disc_weight'] * sem_clf_loss
        dom_loss = dom_pred_loss + cfg['disc_weight'] * dom_clf_loss
        # loss = sem_loss + dom_loss
        loss = sem_loss + dom_loss + cfg['or_weight'] * o_loss

        loss_rcd['o_loss'] += o_loss.item()
        loss_rcd['sem_pred'] += sem_pred_loss.item()
        loss_rcd['sem_clf'] += sem_clf_loss.item()
        loss_rcd['dom_pred'] += dom_pred_loss.item()
        loss_rcd['dom_clf'] += dom_clf_loss.item()
        loss_rcd['total'] += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for loss_type in loss_rcd:
        loss_rcd[loss_type] /= batch_num
    return loss_rcd


def disentangle_train(model, save_model):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    loaders = get_dataloaders()

    for epoch in range(cfg['epoch_disent']):
        train_loss = train(model, optimizer, loaders)
        logger.info(f'Disentangle train, epoch {epoch} ' +
                    ','.join([f'{tp}: {val:.6f}' for tp, val in train_loss.items()]))
        # 保存
        if save_model and epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(cfg['model_dir'], 'disent.pth'))


def multi_source_disentangle_train(model, loaders, save_model):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    for epoch in range(cfg['epoch_disent']):
        train_loss = train(model, optimizer, loaders)
        logger.info(f'Disentangle train, epoch {epoch} ' +
                    ','.join([f'{tp}: {val:.6f}' for tp, val in train_loss.items()]))
        # 保存
        if save_model and epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(cfg['model_dir'], 'disent.pth'))
