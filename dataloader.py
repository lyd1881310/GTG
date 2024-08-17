import random

import numpy as np
import pymetis
from os.path import join
from typing import Any, List

import pandas as pd
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from torch_geometric.utils import k_hop_subgraph
import torch
from tqdm import tqdm

from config import cfg


class SubGraphData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'mapping':
            return self.num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)


class SimpleDataset(Dataset):
    # note: PyG 的 Dataset 类需要实现 len 和 get 方法, 区别于 torch
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def random_sample(self):
        return random.choice(self.data_list)


def metis_cluster(network: Data, num_clusters):
    """
    为每个结点分配聚类标签
    参考 ClusterGCN
    """
    adj_list = [[] for _ in range(network.num_nodes)]
    edge_index = network.edge_index.cpu().t().numpy()
    # metis 算法要求构造无向图, 需要讲两个方向的边都加进去, 否则会 core dump
    for u, v in edge_index:
        if v not in adj_list[u]:
            adj_list[u].append(v)
        if u not in adj_list[v]:
            adj_list[v].append(u)

    num_cuts, labels = pymetis.part_graph(num_clusters, adjacency=adj_list)
    return labels


def get_node_feature(syntax_pth):
    # HACK 祈祷 link_id 和 syntax 直接对齐
    road_df = pd.read_csv(syntax_pth, sep=',', index_col=False, dtype={'osm_way_id': str})
    road_df = road_df[['link_type', 'from_biway', 'is_link', 'length',
                       'angCONN', 'axCONN', 'CONN',
                       'CH', 'CHr500m', 'CHr1000m', 'CHr2000m', 'CHr4000m',
                       'CH[segLEN]', 'CH[segLE_1', 'CH[segLE_2', 'CH[segLE_3', 'CH[segLE_4',
                       'INT', 'INTr1000m', 'INTr2000m', 'INTr4000m', 'INTr500m',
                       'INT[segLEN', 'INT[segL_1', 'INT[segL_2', 'INT[segL_3', 'INT[segL_4', 'NC',
                       'NCr1000m', 'NCr2000m', 'NCr4000m', 'NCr500m',
                       'TD', 'TDr1000m', 'TDr2000m', 'TDr4000m', 'TDr500m',
                       'TD[segLEN]', 'TD[segLE_1', 'TD[segLE_2', 'TD[segLE_3', 'TD[segLE_4',
                       'NACH', 'NACHr1000m', 'NACHr2000m', 'NACHr4000m', 'NACHr500m',
                       'NACH[segLE', 'NACH[seg_1', 'NACH[seg_2', 'NACH[seg_3', 'NACH[seg_4',
                       'NAIN', 'NAINr1000m', 'NAINr2000m', 'NAINr4000m', 'NAINr500m',
                       'NAIN[segLE', 'NAIN[seg_1', 'NAIN[seg_2', 'NAIN[seg_3', 'NAIN[seg_4']]

    # 对连续值进行标准化处理，否则不同城市之间值大小可能差异很大
    dis_cols = ["link_type", "from_biway", "is_link"]
    for c in road_df.columns:
        if c in dis_cols:
            continue
        road_df[c] = (road_df[c] - road_df[c].mean()) / road_df[c].std()

    road_df = road_df.fillna(0)

    return road_df.values.tolist()


def get_network(city):
    # 读取路网
    nodes = get_node_feature(join('data', city, cfg['road_pth']))
    nodes = torch.tensor(nodes, dtype=torch.float).contiguous()
    edge_df = pd.read_csv(join('data', city, cfg['edge_pth']))
    index = edge_df[['src', 'trg']].values.tolist()

    # 加上边的空间信息, 输入到 GAT 的注意力计算函数中
    prop = edge_df[['length', 'dist', 'angle', 'bet']].values.tolist()
    edge_index = torch.tensor(index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(prop, dtype=torch.float32).contiguous()
    return Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr)


def get_supervise_loader(city):
    network = get_network(city)
    loaders = dict()
    for phase in ['train', 'valid']:
        label_df = pd.read_csv(join('data', city, f'traj/{phase}_label.csv'))
        dataset = gen_cluster_dataset(network, label_df)
        loaders[phase] = DataLoader(dataset, batch_size=1, shuffle=True)
    return loaders['train'], loaders['valid']


def gen_cluster_dataset(network, label_df):
    """
    采样 ClusterGCN 的数据集
    """
    # 去掉标签太大和太小的样本
    low_dur = label_df['dur_mean'].quantile(0.01)
    upper_dur = label_df['dur_mean'].quantile(0.99)
    label_df = label_df[(label_df['dur_mean'] > low_dur) & (label_df['dur_mean'] < upper_dur)].copy()
    # speed 越小则 cost 越大
    label_df['speed_mean'] = cfg['max_spd'] - label_df['speed_mean']
    assert label_df['speed_mean'].min() >= 0, 'negative speed cost'

    num_clusters = int(network.num_nodes / cfg['local_size'])
    clus_label = metis_cluster(network, num_clusters)

    data_list = []
    label_groups = label_df.groupby('time_index')

    # for time_idx in tqdm(range(cfg['num_time']), desc='gen labeled data'):
    for time_idx, grp in tqdm(label_groups, desc='gen labeled data'):
        dur_dict = {int(row['rid']): row['dur_mean'] for _, row in grp.iterrows()}
        spd_dict = {int(row['rid']): row['speed_mean'] for _, row in grp.iterrows()}
        dur_mean = torch.tensor([dur_dict.get(i, -1) for i in range(network.num_nodes)], dtype=torch.float32)
        spd_mean = torch.tensor([spd_dict.get(i, -1) for i in range(network.num_nodes)], dtype=torch.float32)

        # 确保每个簇都采样到
        for c in range(num_clusters):
            clus_list = list(range(num_clusters))
            clus_list.remove(c)
            sample_clus = random.sample(clus_list, cfg['sample_cnum'] - 1)
            sample_clus.append(c)
            sample_rids = [idx for idx, clu in enumerate(clus_label) if clu in sample_clus]

            subset, sub_edge_index, _, edge_mask = k_hop_subgraph(
                node_idx=sample_rids, num_hops=0, edge_index=network.edge_index, relabel_nodes=True,
                num_nodes=network.num_nodes,
                directed=False
            )
            subset = subset.tolist()
            sub_x = network.x[subset]
            sub_edge_attr = network.edge_attr[edge_mask]

            sub_dur, sub_spd = dur_mean[subset], spd_mean[subset]
            # 排除没有值 (-1) 的结点
            mapping = [idx for idx, val in enumerate(sub_dur) if val > 0]

            # 排除没有真值的样本
            if len(mapping) == 0:
                continue

            sub_dur = sub_dur[mapping]
            sub_spd = sub_spd[mapping]

            t = torch.tensor(time_idx, dtype=torch.float)
            data_list.append(SubGraphData(
                x=torch.cat([t.repeat(sub_x.size(0), 1), sub_x], dim=-1),
                edge_index=sub_edge_index,
                edge_attr=sub_edge_attr,
                y1=sub_dur,
                y2=sub_spd,
                mapping=mapping)
            )
    random.shuffle(data_list)
    return SimpleDataset(data_list)


def get_traj_dataset(dataset_name, valid_num=10000):
    """
    获取 train 轨迹数据, 切分验证集
    """
    traj_pth = join('data', dataset_name, cfg['traj_pth']['train'])
    traj_df = pd.read_csv(traj_pth, sep=';', index_col=False,
                          dtype={'start_time': int, 'rid_list': str})
    traj_df = traj_df[['start_time', 'rid_list']]
    dataset = []
    for row in traj_df.itertuples():
        rid_list = [int(rid) for rid in row.rid_list.split(',')]
        time_index = int(row.start_time / 86400 * cfg['num_time']) % cfg['num_time']
        dataset.append((rid_list, time_index))
    random.shuffle(dataset)
    valid_set, train_set = dataset[:valid_num], dataset[valid_num:]
    return train_set, valid_set
