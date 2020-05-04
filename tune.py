import argparse
import logging
from os import path as osp

from ax.service.ax_client import AxClient

import ray
from cc19 import utils
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.ax import AxSearch
from ray.util.sgd.torch import TorchTrainer


def main(args):
    utils.init_random()

    exp_configs, tune_configs = utils.get_tune_configs(args.logdir)

    hparams = {}
    parameters = []
    for param_subset, params in tune_configs.items():
        hparams[param_subset] = []
        for param, options in params.items():
            parameters.append({'name': param, **options})
            hparams[param_subset].append(param)

            # tune.grid_search(sorted(values, reverse=True))
    exp_configs['hparams'] = hparams

    exp_configs['data_params']['subset'] = args.subset
    max_epochs = 2 if args.smoke else args.max_epochs
    num_samples = 2 if args.smoke else args.num_samples
    exp_configs.update({'num_gpus': 1})

    # ray.init()
    ray.init(memory=2000 * 1024 * 1024,
             object_store_memory=200 * 1024 * 1024,
             driver_object_store_memory=100 * 1024 * 1024)

    scheduler = AsyncHyperBandScheduler(time_attr="training_iteration",
                                        metric="val_accuracy",
                                        mode="max",
                                        grace_period=5,
                                        max_t=max(max_epochs, 5))
    # algo = HyperOptSearch(space, max_concurrent=4, metric="scalar/val/loss", mode="min")
    client = AxClient(enforce_sequential_optimization=True)
    client.create_experiment(parameters=parameters, objective_name='val_accuracy')
    search_alg = AxSearch(client, max_concurrent=1, mode='max')
    # search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)
    reporter = CLIReporter()
    reporter.add_metric_column("val_accuracy")
    reporter.add_metric_column("train_loss")
    trainable = TorchTrainer.as_trainable(model_creator=utils.model_creator,
                                          data_creator=utils.data_creator,
                                          optimizer_creator=utils.optimizer_creator,
                                          loss_creator=utils.loss_creator,
                                          scheduler_creator=utils.scheduler_creator,
                                          scheduler_step_freq="epoch",
                                          use_gpu=True,
                                          config={"batch_size": 512},
                                          num_workers=args.workers)
    analysis = tune.run(trainable,
                        num_samples=num_samples,
                        config=exp_configs,
                        trial_name_creator=utils.trial_str_creator,
                        progress_reporter=reporter,
                        scheduler=scheduler,
                        search_alg=search_alg,
                        stop={"training_iteration": max_epochs},
                        local_dir=args.logdir,
                        checkpoint_freq=10,
                        checkpoint_at_end=True,
                        keep_checkpoints_num=3,
                        resume=args.resume,
                        checkpoint_score_attr='val_accuracy',
                        max_failures=2,
                        verbose=1)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-v', '--verbose', action='store_true')
    PARSER.add_argument('-s', '--smoke', action='store_true')
    PARSER.add_argument('-r', '--resume', action='store_true')
    PARSER.add_argument('--logdir')
    PARSER.add_argument('--max-epochs', type=int, default=100)
    PARSER.add_argument('--num-samples', type=int, default=1)
    PARSER.add_argument('--workers', type=int, default=2)
    PARSER.add_argument('--subset', type=float, default=1.0)

    ARGS = PARSER.parse_args()

    loglevel = 'DEBUG' if ARGS.verbose else 'INFO'
    level = logging.getLevelName(loglevel)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level,
                        format=log_format,
                        filename=osp.join(ARGS.logdir, 'train.log'),
                        filemode='a')

    main(ARGS)
