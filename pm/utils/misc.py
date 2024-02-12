# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import torch
import os.path as osp
import warnings
from typing import Union
import prettytable

from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log

def print_table(stats):
    table = prettytable.PrettyTable()
    for key, value in stats.items():
        table.add_column(key, value)
    return table

def find_latest_checkpoint(path, suffix='pth'):
    """Find the latest checkpoint from the working directory.

    Args:
        path(str): The path to find checkpoints.
        suffix(str): File extension.
            Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.
    References:
        .. [1] https://github.com/microsoft/SoftTeacher
                  /blob/main/ssod/utils/patch.py
    """
    if not osp.exists(path):
        warnings.warn('The path of checkpoints does not exist.')
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'checkpoint_*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('There are no checkpoints in the path.')
        return None
    latest = 0
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path

def save_checkpoint(episode, agent, exp_path, if_best = False):
    state_dict = agent.get_state_dict()
    state_dict["episode"] = episode

    if if_best:
        checkpoint_path = os.path.join(exp_path, "best.pth".format(episode))
    else:
        checkpoint_path = os.path.join(exp_path, "checkpoint_{:04d}.pth".format(episode))
    print(f"{checkpoint_path} saving......")
    torch.save(state_dict, checkpoint_path)
    print(f"{checkpoint_path} saved !!!")

def load_checkpoint(agent, checkpoint_path):

    print("resume from {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    agent.set_state_dict(checkpoint)

    episode = checkpoint["episode"]
    del checkpoint
    torch.cuda.empty_cache()
    return episode

def update_data_root(cfg, root):
    if isinstance(cfg, Config):
        cfg.root = root
        for key, value in cfg.items():
            update_data_root(value, root)
    elif isinstance(cfg, dict):
        for key, value in cfg.items():
            if key == "root":
                cfg[key] = root
            elif isinstance(value, dict):
                update_data_root(value, root)

def get_test_pipeline_cfg(cfg: Union[str, ConfigDict]) -> ConfigDict:
    """Get the test dataset pipeline from entire config.

    Args:
        cfg (str or :obj:`ConfigDict`): the entire config. Can be a config
            file or a ``ConfigDict``.

    Returns:
        :obj:`ConfigDict`: the config of test dataset.
    """
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)

    def _get_test_pipeline_cfg(dataset_cfg):
        if 'pipeline' in dataset_cfg:
            return dataset_cfg.pipeline
        # handle dataset wrapper
        elif 'dataset' in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.dataset)
        # handle dataset wrappers like ConcatDataset
        elif 'datasets' in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.datasets[0])

        raise RuntimeError('Cannot find `pipeline` in `test_dataloader`')

    return _get_test_pipeline_cfg(cfg.test_dataloader.dataset)
