import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

from models import GTGModel, get_total_costs
from dataloader import get_network
from config import cfg
from pathfinding import build_graph, shortest_path


def pred_and_build_graph(model: GTGModel, network: Data, t):
    model.eval()
    dur_cost, speed_cost, _, pref = get_total_costs(model, network, device=cfg['device'], t=t, grad=False)
    return build_graph(network, pref.cpu())


def generate_traj(model: GTGModel, network: Data, gt_df: pd.DataFrame):
    gt_df['time_index'] = gt_df['start_time'].apply(
        lambda x: int(x / 86400 * cfg['num_time']) % cfg['num_time'])

    gt_df = gt_df.sort_values('time_index')

    curr_t = 0
    graph = pred_and_build_graph(model=model, network=network, t=0)
    gen_list = []
    for _, row in tqdm(gt_df.iterrows(), total=gt_df.shape[0], desc='generate traj'):
        rid_list = [int(rid) for rid in row['rid_list'].split(',')]
        if row['time_index'] != curr_t:
            curr_t = row['time_index']
            graph = pred_and_build_graph(model=model, network=network, t=curr_t)
        path = shortest_path(graph, rid_list[0], rid_list[-1])
        path_str = ','.join([str(rid) for rid in path])
        gen_list.append((row['traj_id'], path_str))
    return pd.DataFrame(data=gen_list, columns=['traj_id', 'rid_list'])


def run_generate():
    exp_id = cfg['exp_tag']
    src_city, trg_city = cfg['dataset_source'], cfg['dataset_target']

    model_pth = f'ckpt/exp_{exp_id}/{src_city}_to_{trg_city}/pref.pth'
    print(f'exp_{cfg["exp_tag"]} generating {src_city} -> {trg_city}', model_pth)

    model = GTGModel(cfg).to(cfg['device'])
    model.load_state_dict(torch.load(model_pth, map_location=cfg['device']))
    trg_network = get_network(trg_city)

    gt_df = pd.read_csv(f'data/{trg_city}/traj/test.csv', sep=';')
    gen_df = generate_traj(model=model, network=trg_network, gt_df=gt_df)
    gen_df.to_csv(f'ckpt/exp_{exp_id}/{src_city}_to_{trg_city}/generate.csv', index=False)
