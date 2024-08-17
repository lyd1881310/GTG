import pandas as pd
from typing import Dict
from .macro_similarity import calc_macro_similarity_parallel
from .micro_similarity import calc_micro_similarity_parallel


def evaluate_generate(gen_df: pd.DataFrame, gt_df: pd.DataFrame, rid_gps: Dict, match_mode, chunk_num=30):
    """
    gen_df: 生成轨迹数据集
    gt_df: 真实轨迹数据集
    轨迹数据集 dataframe 包括两列: (idx, rid_list) 或者 (traj_id, rid_list)
        如果是 traj_id, 则 match_mode = 'traj_id', 按轨迹编号一一对应
        如果是 idx, 则 match_mode = 'start_rid', 自动在 gt_df 中寻找和每一条生成轨迹起点相同的轨迹进行对应
    rid_gps: 一个 dict, 将路段id映射为 gps, 形如
        {"0": (lon, lat), "1": (lon, lat), ... }
    """
    macro_dict = calc_macro_similarity_parallel(gen_df=gen_df, gt_df=gt_df, rid_gps=rid_gps, chunk_num=chunk_num)
    micro_dict = calc_micro_similarity_parallel(gen_df=gen_df, gt_df=gt_df, rid_gps=rid_gps, match_mode=match_mode,
                                                chunk_num=chunk_num)
    eval_dict = {**macro_dict, **micro_dict}
    return eval_dict
