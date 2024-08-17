# 从轨迹个体层面比较轨迹的相似度
# 选用指标编辑距离（edit distance）、Hausdorff、DTW 三个指标
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# from evaluate.evaluate_funcs import edit_distance, hausdorff_metric, dtw_metric, s_edr
from .evaluate_funcs import edit_distance, hausdorff_metric, dtw_metric, s_edr


def lcs(t1, t2):
    n, m = len(t1), len(t2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, (n + 1)):
        for j in range(1, (m + 1)):
            # 如果dp[1][1]
            # 判断t1下标为零元素 和t2下标为零元素是否相同
            # 相同，dp二维矩阵里[1][1]位置的值等于左上角【即[0][0]】+1
            if t1[i - 1] == t2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def calc_f1_score(gen_rid_list, gt_rid_list):
    if len(gen_rid_list) == 0 or len(gt_rid_list) == 0:
        return 0
    correct = lcs(gen_rid_list, gt_rid_list)
    prec, recall = correct / len(gen_rid_list), correct / len(gt_rid_list)
    score = 2 * prec * recall / (prec + recall + 1e-5)
    return score


def calc_accuracy(gen_df: pd.DataFrame, gt_df: pd.DataFrame, match_mode: str = "traj_id"):
    traj_pairs = match_with_gt(gen_df, gt_df, match_mode)
    total_gen, total_gt, total_correct = 0, 0, 0
    for gen_rids, gt_rids in tqdm(traj_pairs, desc='calc accuracy'):
        total_correct += lcs(gen_rids, gt_rids)
        total_gt += len(gt_rids)
        total_gen += len(gen_rids)
    eps = 1e-5
    return total_correct / (total_gen + eps), total_correct / (total_gt + eps)


def match_with_gt(gen_traj: pd.DataFrame, gt_traj: pd.DataFrame, mode: str = "traj_id"):
    """
    与真实轨迹进行对应
    """
    assert mode == 'traj_id' or mode == 'start_rid', 'Match assert error'
    if mode == 'start_rid':
        gen_traj["start_rid"] = gen_traj.apply(lambda r: int(r["rid_list"].split(",")[0]), axis=1)
        gt_traj["start_rid"] = gt_traj.apply(lambda r: int(r["rid_list"].split(",")[0]), axis=1)

    gt_traj = gt_traj.drop_duplicates(subset=mode, keep='first')
    merge_df = pd.merge(gen_traj, gt_traj, on=mode, suffixes=('_gen', '_gt'))
    traj_pairs = []
    for idx, row in tqdm(merge_df.iterrows(), total=merge_df.shape[0], desc='matching'):
        gen_rid_list = [int(x) for x in row['rid_list_gen'].split(',')]
        gt_rid_list = [int(x) for x in row['rid_list_gt'].split(',')]
        traj_pairs.append((gen_rid_list, gt_rid_list))

    if mode == 'start_rid':
        del gen_traj["start_rid"]
        del gt_traj["start_rid"]
    return traj_pairs


def work_total_distance(args):
    """
    计算所有轨迹对的总距离
    """
    traj_pairs, rid_gps, metrics = args

    dist_dict = {metric: 0 for metric in metrics}
    for gen_list, gt_list in tqdm(traj_pairs, desc='seq dist'):
        gen_gps = np.array([rid_gps[str(rid)] for rid in gen_list])
        gt_gps = np.array([rid_gps[str(rid)] for rid in gt_list])
        gen_gps[:, [0, 1]] = gen_gps[:, [1, 0]] # 纬度在前
        gt_gps[:, [0, 1]] = gt_gps[:, [1, 0]]
        for metric in metrics:
            if metric == 'Hausdorff':
                dist = hausdorff_metric(gt_gps, gen_gps)
            elif metric == 'DTW':
                dist = dtw_metric(gt_gps, gen_gps)
            elif metric == 'EDT':
                dist = edit_distance(gt_list, gen_list)
            elif metric == 'EDR':
                dist = s_edr(gt_gps, gen_gps, eps=100)
            elif metric == 'Precision':
                lcs_len = lcs(gt_list, gen_list)
                dist = lcs_len / len(gen_list)
            else:
                assert metric == 'Recall'
                lcs_len = lcs(gt_list, gen_list)
                dist = lcs_len / len(gt_list)
            dist_dict[metric] += dist
    return dist_dict


def calc_micro_similarity(gen_df, gt_df, rid_gps, match_mode, metrics=None):
    """
    非并行计算
    """
    if metrics is None:
        metrics = ['Hausdorff', 'DTW', 'EDT', 'EDR', 'Precision', 'Recall']
    traj_pairs = match_with_gt(gen_df, gt_df, match_mode)
    dist_dict = work_total_distance((traj_pairs, rid_gps, metrics))
    mean_dist_dict = dict()
    for metric in metrics:
        mean_dist_dict[metric] = dist_dict[metric] / (len(traj_pairs) + 1e-8)
    print(f'total matched traj {len(traj_pairs)}, total true traj {len(gt_df)}')
    print(mean_dist_dict)
    return mean_dist_dict


def calc_micro_similarity_parallel(gen_df, gt_df, rid_gps, match_mode, chunk_num=20, metrics=None):
    if metrics is None:
        metrics = ['Hausdorff', 'DTW', 'EDT', 'EDR', 'Precision', 'Recall']
    traj_pairs = match_with_gt(gen_df, gt_df, match_mode)
    chunk_size = math.ceil(len(traj_pairs) / chunk_num)
    chunks = [(traj_pairs[i: i + chunk_size], rid_gps, metrics)
              for i in range(0, len(traj_pairs), chunk_size)]

    print('start execute')
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(work_total_distance, chunks))

    mean_dist_dict = dict()
    for metric in metrics:
        total_dist = np.sum([ck_dict[metric] for ck_dict in results])
        mean_dist_dict[metric] = total_dist / (len(traj_pairs) + 1e-8)
    print(f'total matched traj {len(traj_pairs)}, total true traj {len(gt_df)}')
    print(mean_dist_dict)
    return mean_dist_dict

