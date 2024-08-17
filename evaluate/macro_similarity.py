import json
import math
import pandas as pd
import pyproj
from tqdm import tqdm
from geopy import distance
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from .evaluate_funcs import get_geogradius, js_divergence


def work_count(exe_args):
    """
    每个进程的统计操作
    """
    traj_df, road_gps, road_num = exe_args
    latlon2xy = pyproj.Transformer.from_crs(4326, 32649)
    travel_distance_total = []
    travel_radius_total = []
    rid_cnt = np.zeros(road_num, dtype=np.int32)
    for index, row in tqdm(traj_df.iterrows(), total=traj_df.shape[0], desc='count trajectory'):
        rid_list = [int(x) for x in row['rid_list'].split(',')]
        rid_list = np.array(rid_list)
        if len(rid_list) == 0:
            continue

        rid_df = pd.DataFrame(data=rid_list, columns=['rid'])
        rid_df[['lon', 'lat']] = rid_df.apply(lambda rw: road_gps[str(rw['rid'])], axis=1, result_type='expand')
        rid_df['x'], rid_df['y'] = latlon2xy.transform(rid_df['lat'], rid_df['lon'])
        rid_df['dist'] = np.linalg.norm([rid_df['x'].diff().fillna(0), rid_df['y'].diff().fillna(0)], axis=0)
        x_mean, y_mean = rid_df['x'].mean(), rid_df['y'].mean()
        rid_df['radius'] = np.linalg.norm([rid_df['x'] - x_mean, rid_df['y'] - y_mean], axis=0)
        # 单位转为 km
        travel_distance = rid_df['dist'].sum() / 1000
        travel_radius = rid_df['radius'].mean() / 1000

        for rid in rid_list:
            rid_cnt[rid] += 1

        travel_distance_total.append(travel_distance)
        travel_radius_total.append(travel_radius)

    # 除以总数，用频率来计算，不然可能出现 js 散度大于 1; 还要防止除以 0
    # rid_freq = rid_cnt / (rid_cnt.sum() + 1e-8)
    return {
        'dist': travel_distance_total,
        'radius': travel_radius_total,
        'rid_cnt': rid_cnt
    }


def parallel_statistic(traj_df, road_gps, road_num, chunk_num=10):
    """
    统计轨迹的特性
    出行距离分布、出行覆盖面积分布
    轨迹集合的网格访问频次与路段访问频次
    """
    chunk_size = math.ceil(len(traj_df) / chunk_num)
    chunks = [(traj_df.iloc[i: i + chunk_size], road_gps, road_num)
              for i in range(0, len(traj_df), chunk_size)]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(work_count, chunks))

    dist_total, radius_total = [], []
    rid_freq = np.zeros(road_num)
    for res_dict in results:
        dist_total += res_dict['dist']
        radius_total += res_dict['radius']
        rid_freq += res_dict['rid_cnt']

    return {
        'dist': dist_total,
        'radius': radius_total,
        'rid_freq': rid_freq
    }


def calc_macro_similarity_parallel(gen_df, gt_df, rid_gps, chunk_num=20):
    traj_num = min(len(gt_df), len(gen_df))
    gt_df = gt_df.iloc[:traj_num, :].copy()
    gen_df = gen_df.iloc[:traj_num, :].copy()

    real_max_distance = 30  # 统计了三个数据集中的轨迹，绝大部分轨迹长度小于 4 km
    real_max_radius = 10  #
    travel_distance_bins = np.arange(0, real_max_distance, float(real_max_distance) / 1000).tolist()
    travel_radius_bins = np.arange(0, real_max_radius, float(real_max_radius) / 100).tolist()

    all_rids = [int(rid) for rid in rid_gps.keys()]
    road_num = max(all_rids) + 1

    gt_stat = parallel_statistic(traj_df=gt_df, road_gps=rid_gps, road_num=road_num, chunk_num=chunk_num)
    gen_stat = parallel_statistic(traj_df=gen_df, road_gps=rid_gps, road_num=road_num, chunk_num=chunk_num)

    # 频率归一化
    js_loc_freq = js_divergence(gen_stat["rid_freq"] / gen_stat["rid_freq"].sum(),
                                gt_stat["rid_freq"] / gt_stat["rid_freq"].sum())

    true_distance_distri, _ = np.histogram(gt_stat["dist"], travel_distance_bins, density=True)
    true_radius_distri, _ = np.histogram(gt_stat["radius"], travel_radius_bins, density=True)
    gen_distance_distri, _ = np.histogram(gen_stat["dist"], travel_distance_bins, density=True)
    gen_radius_distri, _ = np.histogram(gen_stat["radius"], travel_radius_bins, density=True)

    js_distance = js_divergence(gen_distance_distri, true_distance_distri)
    js_radius = js_divergence(gen_radius_distri, true_radius_distri)

    print(f"distance: {js_distance}, radius: {js_radius}, loc_freq: {js_loc_freq}")
    return {
        'Distance': js_distance,
        'Radius': js_radius,
        'LocFreq': js_loc_freq
    }
