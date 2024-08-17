from graph_tool.all import *
import math
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from os.path import join
from shapely.geometry import LineString
from tqdm import tqdm
from config import cfg


def gen_label(traj_df: pd.DataFrame, road_df: pd.DataFrame):
    rid_length = [row['length'] for _, row in road_df.iterrows()]
    traj_df['time_index'] = traj_df['start_time'].apply(
        lambda x: int(x / 86400 * cfg['num_time']) % cfg['num_time'])

    dur_dicts = [dict() for _ in range(cfg['num_time'])]
    speed_dicts = [dict() for _ in range(cfg['num_time'])]
    print(f'start mapping...', file=sys.stderr)
    for idx, row in tqdm(traj_df.iterrows()):
        rid_list = [int(ele) for ele in row['rid_list'].split(',')]
        dur_list = [int(ele) for ele in row['dur_list'].split(',')]
        dur_dict = dur_dicts[row['time_index']]
        speed_dict = speed_dicts[row['time_index']]
        for rid, dur in zip(rid_list, dur_list):
            # 收集样本的时候对数据的合理性进行检查 (主要是速度)
            # 极端值会严重影响后面平均数的计算
            speed = rid_length[rid] / (dur + 1e-5)
            if 3.6 * speed < 5 or 3.6 * speed > 100:
                continue
            if rid not in dur_dict:
                dur_dict[rid] = []
                speed_dict[rid] = []
            dur_dict[rid].append(dur)
            speed_dict[rid].append(speed)

    print(f'start reducing...', file=sys.stderr)
    # calc dur mean and speed mean for each dict
    for dur_dict, speed_dict in zip(dur_dicts, speed_dicts):
        for rid in dur_dict:
            dur_list, speed_list = dur_dict[rid], speed_dict[rid]
            dur_mean, dur_std = np.mean(dur_list), np.std(dur_list)
            speed_mean, speed_std = np.mean(speed_list), np.std(speed_list)
            dur_list = [ele for ele in dur_list if abs(ele - dur_mean) < 3 * dur_std]
            speed_list = [ele for ele in speed_list if abs(ele - speed_mean) < 3 * speed_std]
            dur_dict[rid] = np.mean(dur_list) if dur_list else -1
            speed_dict[rid] = np.mean(speed_list) if speed_list else -1

    # 把 train 轨迹产生的 cost 标签切分训练集和验证集, 主要是验证 cost 预测的准确性
    data_list = []
    for time_index, (dur_dict, speed_dict) in enumerate(zip(dur_dicts, speed_dicts)):
        for rid in dur_dict:
            dur_mean, speed_mean = dur_dict[rid], speed_dict[rid]
            if dur_mean > 0 and speed_mean > 0:
                data_list.append((time_index, rid, dur_mean, speed_mean))
    label_df = pd.DataFrame(data=data_list, columns=['time_index', 'rid', 'dur_mean', 'speed_mean'])
    train_num = int(len(label_df) * 0.8)
    label_df = label_df.sample(frac=1.0)
    train_df = label_df.iloc[:train_num, :].copy()
    valid_df = label_df.iloc[train_num:, :].copy()
    return train_df, valid_df


def gen_all_label():
    # 生成 cost 标签的时候用 train 的轨迹
    traj_pth = join('data', cfg['dataset_source'], cfg['traj_pth']['train'])
    road_pth = join('data', cfg['dataset_source'], cfg['road_pth'])
    train_label_pth = join('data', cfg['dataset_source'], cfg['traj_label_pth']['train'])
    valid_label_pth = join('data', cfg['dataset_source'], cfg['traj_label_pth']['valid'])
    # label_pth = join('data', cfg['dataset_source'], cfg['traj_label_pth'])

    # traj_df (traj_id;start_time;rid_list;dur_list)
    traj_df = pd.read_csv(traj_pth, sep=';')
    road_df = pd.read_csv(road_pth, sep=',').sort_values('link_id')
    train_df, valid_df = gen_label(traj_df, road_df)
    train_df.to_csv(train_label_pth, index=False)
    valid_df.to_csv(valid_label_pth, index=False)
    print(f'{cfg["dataset_source"]} label generated', file=sys.stderr)


def gen_roadmap():
    road_pth = join('data', cfg['dataset_source'], cfg['road_pth'])
    road_df = pd.read_csv(road_pth, sep=',', index_col=False, dtype={'osm_way_id': str})
    road_df['link_id'] = range(road_df.shape[0])
    road_df = road_df[['link_id', 'from_node_id', 'to_node_id']]

    # 生成路网图
    road_out = (road_df.groupby('from_node_id')['link_id'].apply(list)
                .reset_index().rename(columns={'link_id': 'out_links'}))
    road_in = (road_df.groupby('to_node_id')['link_id'].apply(list)
               .reset_index().rename(columns={'link_id': 'in_links'}))

    roadmap_df = road_in.merge(road_out, left_on='to_node_id', right_on='from_node_id')
    roadmap_df = roadmap_df[['in_links', 'out_links']]

    road_out = roadmap_df.explode('in_links').rename(columns={'in_links': 'link_id'})
    road_in = roadmap_df.explode('out_links').rename(columns={'out_links': 'link_id'})

    roadmap_df = road_in.merge(road_out, on='link_id', how='outer')

    roadmap_pth = join('data', cfg['dataset_source'], cfg['roadmap_pth'])

    with open(roadmap_pth, 'w') as f:
        f.write('link_id;in_links;out_links\n')
        for idx, row in roadmap_df.iterrows():
            if isinstance(row['in_links'], float):
                row['in_links'] = []
            if isinstance(row['out_links'], float):
                row['out_links'] = []
            f.write(f"{row['link_id']};{','.join(map(str, row['in_links']))};{','.join(map(str, row['out_links']))}\n")

    print(f'{cfg["dataset_source"]} roadmap generated', file=sys.stderr)


def gen_finetune_data():
    # percent = [1, 2, 5, 10, 20]
    percent = [0.1, 0.2, 0.5]
    city = cfg['dataset_source']
    traj_pth = join('data', city, cfg['traj_pth']['train'])
    road_pth = join('data', city, cfg['road_pth'])
    traj_df = pd.read_csv(traj_pth, sep=';')
    road_df = pd.read_csv(road_pth, sep=',').sort_values('link_id')
    for per in percent:
        sample_df = traj_df.sample(frac=per / 100)
        train_df, valid_df = gen_label(sample_df, road_df)
        sample_df.to_csv(f'data/{city}/traj/train_p{per}.csv', index=False, sep=';')
        train_df.to_csv(f'data/{city}/traj/train_label_p{per}.csv', index=False)
        valid_df.to_csv(f'data/{city}/traj/valid_label_p{per}.csv', index=False)


def gen_finetune_data_num():
    # group_num = 8
    # traj_nums = [int(100 * 2 ** i) for i in range(group_num)]
    traj_nums = [25600, 51200, 102400]
    print('sample traj num', traj_nums)
    city = cfg['dataset_source']
    traj_pth = join('data', city, cfg['traj_pth']['train'])
    road_pth = join('data', city, cfg['road_pth'])
    traj_df = pd.read_csv(traj_pth, sep=';')
    road_df = pd.read_csv(road_pth, sep=',').sort_values('link_id')
    for num in traj_nums:
        sample_df = traj_df.sample(n=num)
        train_df, valid_df = gen_label(sample_df, road_df)
        sample_df.to_csv(f'data/{city}/traj/train_n{num}.csv', index=False, sep=';')
        train_df.to_csv(f'data/{city}/traj/train_label_n{num}.csv', index=False)
        valid_df.to_csv(f'data/{city}/traj/valid_label_n{num}.csv', index=False)


def calc_angle(line: LineString):
    start, end = line.coords[0], line.coords[-1]
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    return angle


def gen_edge_data():
    road_pth = join('data', cfg['dataset_source'], cfg['road_pth'])
    edge_pth = join('data', cfg['dataset_source'], 'map/edge.csv')
    road_df = pd.read_csv(road_pth)
    edge_list = []
    g = Graph(directed=True)
    wt = g.new_edge_property('double')
    for _, from_row in road_df.iterrows():
        nbr_df = road_df[road_df['from_node_id'] == from_row['to_node_id']].copy()
        for _, to_row in nbr_df.iterrows():
            length = (from_row['length'] + to_row['length']) / 2
            edge_list.append((from_row['link_id'], to_row['link_id'], length))
    g.add_edge_list(edge_list, eprops=[wt])
    g.ep.wt = wt

    data_list = []
    geom = gpd.GeoSeries.from_wkt(road_df['geometry'], crs=4326).to_crs(32649)
    vp, ep = betweenness(g, weight=wt)
    for e in tqdm(g.edges()):
        src, trg = int(e.source()), int(e.target())
        bet = ep[e]
        src_line, trg_line = geom.iloc[src], geom.iloc[trg]
        length = (src_line.length + trg_line.length) / 2
        dist = src_line.centroid.distance(trg_line.centroid)
        theta = calc_angle(trg_line) - calc_angle(src_line)
        data_list.append((src, trg, length, dist, theta, bet))
    edge_df = pd.DataFrame(data=data_list, columns=['src', 'trg', 'length', 'dist', 'angle', 'bet'])
    edge_df.to_csv(edge_pth, index=False)


if __name__ == '__main__':
    for c in ['xianshi', 'chengdushi', 'beijing']:
        cfg['dataset_source'] = c
        # gen_all_label()
        # gen_roadmap()
        # gen_finetune_data()
        gen_finetune_data_num()
        # gen_edge_data()
