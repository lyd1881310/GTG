import logging
from logging import getLogger

import numpy as np

from models import GTGModel
from config import cfg
import torch
import torch.optim as optim
import torch.nn.functional as F
import os

logger = getLogger(__name__)


def rank_loss(pred: torch.Tensor, label: torch.Tensor):
    pred_diff = torch.sigmoid(pred.unsqueeze(1) - pred.unsqueeze(0))
    label_diff = label.unsqueeze(1) - label.unsqueeze(0)
    label_train = torch.where(label_diff > 0, 1., 0.).to(pred_diff.device)
    label_train[label_diff == 0] = 0.5
    loss = F.binary_cross_entropy(pred_diff.flatten(), label_train.flatten())
    return loss
 

def sup_loss(dur_pred, spd_pred, dur_gt, spd_gt):
    # 直接预测的损失
    pred_loss = F.mse_loss(dur_pred, dur_gt) + F.mse_loss(spd_pred, spd_gt)
    # RankNet 损失
    rk_loss = rank_loss(dur_pred, dur_gt) + rank_loss(spd_pred, spd_gt)
    loss = pred_loss + cfg['rank_weight'] * rk_loss
    return pred_loss, rk_loss, loss


def train(model, device, train_loader, optimizer):
    model.train()
    pred_loss_mean, rk_loss_mean = 0, 0
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        dur_pred, spd_pred = model.pred_obs_cost(batch.x, batch.edge_index, batch.edge_attr)
        dur_pred = dur_pred[batch.mapping]
        spd_pred = spd_pred[batch.mapping]

        pred_loss, rk_loss, loss = sup_loss(dur_pred, spd_pred, batch.y1, batch.y2)
        loss.backward()
        optimizer.step()

        pred_loss_mean += pred_loss.item()
        rk_loss_mean += rk_loss.item()
    return pred_loss_mean / len(train_loader), rk_loss_mean / len(train_loader)


def validate(model, device, valid_loader):
    model.eval()
    valid_loss = 0
    dur, dur_err, speed, spd_err = 0, 0, 0, 0
    with torch.no_grad():
        for batch in valid_loader:
            batch.to(device)

            dur_pred, spd_pred = model.pred_obs_cost(batch.x, batch.edge_index, batch.edge_attr)
            dur_pred = dur_pred[batch.mapping]
            spd_pred = spd_pred[batch.mapping]

            loss = F.mse_loss(dur_pred, batch.y1) + F.mse_loss(spd_pred, batch.y2)
            valid_loss += loss.item()

            dur_pred = dur_pred.detach().cpu().numpy()
            spd_pred = spd_pred.detach().cpu().numpy()

            dur_gt = batch.y1.detach().cpu().numpy()
            spd_gt = batch.y2.detach().cpu().numpy()

            dur_err += np.fabs(dur_gt - dur_pred).sum()
            spd_err += np.fabs(spd_gt - spd_pred).sum()
            dur += np.fabs(dur_gt).sum()
            speed += np.fabs(spd_gt).sum()

    valid_loss /= len(valid_loader)
    dur_err /= dur
    spd_err /= speed
    return valid_loss, dur_err, spd_err


def supervise(model, train_loader, valid_loader, freeze: bool, save_model: bool):
    model.to(cfg['device'])
    if freeze:
        for name, param in model.named_parameters():
            if name.startswith('sem_pred'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

    best_loss = float('inf')
    for epoch in range(1, cfg['epoch_supervise'] + 1):
        pred_loss, rk_loss = train(model, cfg['device'], train_loader, optimizer)

        valid_loss, dur_err, speed_err = validate(model, cfg['device'], valid_loader)
        logger.info(f'Epoch: {epoch:4}, pred loss: {pred_loss:12.6f}, rank loss: {rk_loss:8.6f}, '
                    f'valid dur error: {dur_err:10.4f}, valid speed error: {speed_err:10.4f}')

        if valid_loss < best_loss and save_model:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'{cfg["model_dir"]}/supervise.pth')

        if cfg['debug_mode']:
            break

