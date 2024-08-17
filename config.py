cfg = {
    'debug_mode': False,

    # 路经设置
    'dataset_source': 'xianshi',
    'dataset_target': 'chengdushi',
    'traj_pth': {
        'train': 'traj/train.csv',
        'test': 'traj/test.csv',
    },

    'traj_label_pth': {
        'train': 'traj/train_label.csv',
        'valid': 'traj/valid_label.csv',
    },

    'road_pth': 'map/road.csv',
    'roadmap_pth': 'map/roadmap.csv',
    'edge_pth': 'map/edge.csv',

    # 生成轨迹使用的 cost 类型
    'use_pref': True,

    # 确保速度是正数: max_spd - speed_mean
    'max_spd': 100 / 3.6,

    # 训练设置
    'device': 'cuda',

    'epoch_recons': 200,
    'epoch_cluster': 500,

    'epoch_supervise': 300,

    'epoch_domain_adapt': 1000,
    'outer_adapt_itr': 200,
    'inner_dis_itr': 5000 * 4,
    'inner_gen_itr': 1000,

    'epoch_pref': 1,

    'epoch_disent': 600, # 解耦学习轮数

    # 域分类损失权重
    'disc_weight': 100,
    # rank loss 的权重
    'rank_weight': 50,
    # 正交损失权重
    # 'or_weight': 200,
    'or_weight': 5,

    # 各项 cost 的权重
    'w_time': 0.33,
    'w_speed': 0.33,
    'w_hidden': 0.33,

    'validate_epoch': 10,
    'patience': 5,
    'lr': 1e-5,
    'dropout': 0.2,
    'batch_size': 32,

    # preference train
    'pref_train_ckpt': [100, 500, 1000, 5000, 10000, 50000, 100000],

    # cluster train (废弃)
    'num_cluster': 10,

    # cluster GCN
    'local_size': 50,
    'sample_cnum': 3, # 每个输入样本采样的簇个数

    # finetune
    'finetune_ratio': [0.1, 0.2, 0.5, 1, 2, 5, 10, 20],

    # 手动设置 tag
    'exp_tag': 4,

    # 模型设置
    'num_hops': 6,
    'num_layers': 4,
    'dim_gat_hidden': 128,

    # 边的属性
    'dim_edge': 4,

    # 'num_time': 96,
    'num_time': 24,
    'num_type': 16,

    # 考虑对部分道路属性做 embedding
    'dim_time': 32,
    'dim_type': 32,
    'dim_biway': 16,
    'dim_islink': 16,

    'dim_gat_heads': 8,
    'dim_node_feature': 62,
    'dim_con_feature': 59, # 连续值的特征
    'dim_predict_hidden': 32,
    'dim_predict_output': 2,
}
