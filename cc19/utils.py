import json
import os
import random
from os import path as osp

import numpy as np
import torch
import yaml

from . import datasets, models

PROJECT_DIR = os.environ.get('PROJECT_DIR')


def init_random(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def update_hparams(base_params, keys, config):
    if keys is None:
        return base_params

    tmp = base_params.copy()
    for k in keys:
        tmp[k] = config[k]
    return tmp


def update_keys(base_dict, update):
    if update is None:
        return base_dict

    tmp = base_dict.copy()
    for key in tmp:
        if key in update:
            tmp[key] = update[key]
    return tmp


def resolve_symbols(conf, symbols):
    if isinstance(conf, list):
        for i, v in enumerate(conf):
            conf[i] = resolve_symbols(v, symbols)
    elif isinstance(conf, dict):
        for k, v in conf.items():
            conf[k] = resolve_symbols(v, symbols)
    elif isinstance(conf, str):
        for s, v in symbols.items():
            assert v is not None, f"Symbol {s} is None"
            conf = conf.replace('%' + s, v)
    return conf


def trial_str_creator(trial):
    return "{}_{}".format(trial.trainable_name, trial.trial_id)


def get_tune_configs(log_dir):
    assert osp.isdir(log_dir), 'Log dir does not exists.'

    if osp.exists(osp.join(log_dir, 'tune.yaml')):
        tune_configs = load_dict(osp.join(log_dir, 'tune.yaml'))

        return get_configs(log_dir), tune_configs
    else:
        raise RuntimeError('No tune configs found')


def get_configs(log_dir):
    assert osp.isdir(log_dir), 'Log dir does not exists.'

    if osp.exists(osp.join(log_dir, 'conf.yaml')):
        exp_configs = load_dict(osp.join(log_dir, 'conf.yaml'))
        symbols = {'p': PROJECT_DIR, 'l': log_dir}
        exp_configs = resolve_symbols(exp_configs, symbols)
        return exp_configs

    if osp.exists(osp.join(log_dir, 'params.json')):
        exp_configs = load_dict(osp.join(log_dir, 'params.json'))
        # if 'runner_config' in exp_configs:
        #     return exp_configs['runner_config']

        # if osp.exists(log_dir) and clear:
        #     shutil.rmtree(log_dir)
        # os.mkdir(log_dir)
        return exp_configs

    raise RuntimeError('No configs found')


def load_dict(dict_path):
    _, ext = osp.splitext(dict_path)
    with open(dict_path, 'r') as stream:
        if ext in ['.json']:
            yaml_dict = json.load(stream)
        elif ext in ['.yml', '.yaml']:
            yaml_dict = yaml.safe_load(stream)
    return yaml_dict


def model_creator(config):
    model_params = config['model_params']
    model_params = update_hparams(model_params, config['hparams'].get('model_params'), config)
    model_cls = getattr(models, config['model'])
    return model_cls(**model_params)


def data_creator(config):
    data_params = config['data_params']
    batch_size = config.get("batch_size", 126)
    # TODO: check gpus flag
    return datasets.setup_datasets_np(**data_params, batch_size=batch_size, use_cuda=True)


def optimizer_creator(model, config):
    optim_params = config['optim_params']
    optim_params = update_hparams(optim_params, config['hparams'].get('optim_params'), config)
    optimizer_cls = getattr(torch.optim, config['optim'])
    return optimizer_cls(model.parameters(), **optim_params)


def loss_creator(config):
    criterion = getattr(torch.nn, config['loss'])()
    return criterion


def scheduler_creator(optimizer, config):
    sched_params = config['sched_params']
    sched_params = update_hparams(sched_params, config['hparams'].get('sched_params'), config)
    scheduler_cls = getattr(torch.optim.lr_scheduler, config['lr_sched'])
    return scheduler_cls(optimizer, **sched_params)
