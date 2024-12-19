import graph_tool
import logging
import os
from argparse import ArgumentParser

from config import cfg
from models import GTGModel
from unsupervise import preference_train
from dataloader import get_network, get_traj_dataset
from disentangle import disentangle_train


logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s](%(asctime)s):%(message)s',
    datefmt='%H:%M:%S'
)
file_handler = logging.FileHandler('exp_log.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s](%(asctime)s):%(message)s')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)


def set_cfg():
    parser = ArgumentParser()
    parser.add_argument("--exp_tag", type=int)
    parser.add_argument("--src", type=str)
    parser.add_argument("--trg", type=str)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    cfg["exp_tag"] = args.exp_tag
    cfg["dataset_source"] = args.src
    cfg["dataset_target"] = args.trg
    cfg["device"] = args.device

    exp_dir = f"ckpt/exp_{cfg['exp_tag']}/{args.src}_to_{args.trg}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    cfg['model_dir'] = exp_dir


def main():
    set_cfg()
    model = GTGModel(cfg)
    model.to(cfg['device'])

    # Cost Prediction
    disentangle_train(model, save_model=True)

    # Preference Learning
    network = get_network(cfg['dataset_source'])
    train_traj, valid_traj = get_traj_dataset(cfg['dataset_source'])
    preference_train(model, network, train_traj, valid_traj, save_model=True)


if __name__ == '__main__':
    main()
