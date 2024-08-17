from logging import getLogger
import random
import torch
from tqdm import tqdm

from models import get_total_costs, GTGModel
from pathfinding import shortest_path, build_graph
from config import cfg
from evaluate.micro_similarity import calc_f1_score

logger = getLogger(__name__)


def validate_pref(model: GTGModel, network, valid_dataset):
    valid_dataset = sorted(valid_dataset, key=lambda x: x[1])
    mean_f1 = 0
    curr_t = 0
    _, _, _, total_cost = get_total_costs(model=model, network=network, device=cfg['device'], t=curr_t, grad=False)
    graph = build_graph(network, total_cost.cpu())
    for idx, (rid_list_lbl, t) in enumerate(valid_dataset):
        if t != curr_t:
            _, _, _, total_cost = get_total_costs(model=model, network=network, device=cfg['device'], t=t, grad=False)
            graph = build_graph(network, total_cost.cpu())
            curr_t = t
        rid_s, rid_e = rid_list_lbl[0], rid_list_lbl[-1]
        rid_list_pred = shortest_path(graph, rid_s, rid_e)
        mean_f1 += calc_f1_score(rid_list_pred, rid_list_lbl)
    mean_f1 /= len(valid_dataset)
    return mean_f1


def preference_train(model, network, train_traj, valid_traj, save_model=True, max_patience=500):
    # 冻结 encoder 和 obs cost pred 的参数
    for name, param in model.named_parameters():
        if name.startswith('hidden') or name.startswith('obs_wt'):
            param.requires_grad = True
        else:
            param.requires_grad = False

    best_score, patience = 0, 0
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    for epoch in range(cfg['epoch_pref']):
        for idx, (rid_list_lbl, t) in tqdm(enumerate(train_traj), total=len(train_traj),
                                           desc=f'epoch {epoch}'):
            _, _, hidden_cost, pref = get_total_costs(model=model, network=network,
                                                            device=cfg['device'], t=t, grad=True)
            graph = build_graph(network, pref.cpu())
            rid_s, rid_e = rid_list_lbl[0], rid_list_lbl[-1]
            rid_list_pred = shortest_path(graph, rid_s, rid_e)

            # loss = hidden_cost[rid_list_lbl].sum() - hidden_cost[rid_list_pred].sum()
            loss = pref[rid_list_lbl].sum() - pref[rid_list_pred].sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                score = validate_pref(model, network, valid_traj)
                logger.info(f'epoch {epoch} traj {idx}, F1 score {score:.8f}')
                if score > best_score and save_model:
                    torch.save(model.state_dict(), f'{cfg["model_dir"]}/pref.pth')
                    best_score = score
                    patience = 0
                else:
                    patience += 1
                    if patience >= max_patience:
                        break

            if cfg['debug_mode']:
                break

    # 参数解冻
    for name, param in model.named_parameters():
        param.requires_grad = True
