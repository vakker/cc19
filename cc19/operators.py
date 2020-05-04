import torch
from ray.util.sgd.torch import TrainingOperator


class Operator(TrainingOperator):
    def __init__(self, config=None):
        raise NotImplementedError
        super().__init__(config)

    def _log_graph(self, logger, samples):
        logger.add_graph(self.model, samples[:2])

    def tng_step(self, samples):
        inputs = samples[0]
        init_values = samples[1]
        targets = samples[2]

        outputs = self.model(inputs, init_values)

        loss = self.criterion(outputs, targets)
        r2 = get_r2(outputs, targets)
        acc = acc_score(outputs, targets)

        metrics = {'loss': loss, 'acc': acc}
        metrics.update(r2)
        return loss, metrics

    def post_tng_step(self, batch_outputs):
        scalar = {
            'tng/loss': np.mean([ba['loss'] for ba in batch_outputs]),
            'tng/r2': np.mean([ba['r2'] for ba in batch_outputs]),
            'tng/acc': np.mean([ba['acc'] for ba in batch_outputs]),
        }

        if self._bests.get('tng/loss', np.inf) > scalar['tng/loss']:
            self._bests['tng/loss'] = scalar['tng/loss']
        if self._bests.get('tng/r2', -np.inf) < scalar['tng/r2']:
            self._bests['tng/r2'] = scalar['tng/r2']
        if self._bests.get('tng/acc', -np.inf) < scalar['tng/acc']:
            self._bests['tng/acc'] = scalar['tng/acc']

        scalar.update({f'{k}_best': v for k, v in self._bests.items() if k.startswith('tng')})

        num_steps = len(batch_outputs[0]['r2_sep'])
        scalar.update({
            f'tng/r2_{i}': np.mean([ba['r2_sep'][i] for ba in batch_outputs])
            for i in range(num_steps)
        })
        return {'scalar': scalar}

    def val_step(self, samples):
        inputs = samples[0]
        init_values = samples[1]
        targets = samples[2]

        outputs = self.model(inputs, init_values)

        errors = targets - outputs
        loss = self.criterion(outputs, targets)
        r2 = get_r2(outputs, targets)
        acc = acc_score(outputs, targets)

        metrics = {'loss': loss, 'errors': errors, 'acc': acc}
        metrics.update(r2)
        return loss, metrics

    def post_val_step(self, batch_outputs):
        scalar = {
            'val/loss': np.mean([ba['loss'] for ba in batch_outputs]),
            'val/r2': np.mean([ba['r2'] for ba in batch_outputs]),
            'val/acc': np.mean([ba['acc'] for ba in batch_outputs]),
        }

        if self._bests.get('val/loss', np.inf) > scalar['val/loss']:
            self._bests['val/loss'] = scalar['val/loss']
        if self._bests.get('val/r2', -np.inf) < scalar['val/r2']:
            self._bests['val/r2'] = scalar['val/r2']
        if self._bests.get('val/acc', -np.inf) < scalar['val/acc']:
            self._bests['val/acc'] = scalar['val/acc']

        scalar.update({f'{k}_best': v for k, v in self._bests.items() if k.startswith('val')})

        num_steps = len(batch_outputs[0]['r2_sep'])
        scalar.update({
            f'val/r2_{i}': np.mean([ba['r2_sep'][i] for ba in batch_outputs])
            for i in range(num_steps)
        })
        histogram = {'val/errors': np.concatenate([ba['errors'] for ba in batch_outputs], axis=0)}
        return {'scalar': scalar, 'histogram': histogram}

    def inf_step(self, samples):
        inputs = samples[0]
        init_values = samples[1]
        targets = samples[2]

        outputs = self.model(inputs, init_values)

        loss = self.criterion(outputs, targets)
        r2 = get_r2(outputs, targets)

        metrics = {
            'loss': loss,
            'outputs': outputs,
            'targets': targets,
            'init_values': init_values
        }
        metrics.update(r2)
        return loss, metrics

    def post_inf_step(self, batch_outputs):
        scalar = {
            'inf/loss': np.mean([ba['loss'] for ba in batch_outputs]),
            'inf/r2': np.mean([ba['r2'] for ba in batch_outputs]),
        }

        num_steps = len(batch_outputs[0]['r2_sep'])
        scalar.update({
            f'inf/r2_{i}': np.mean([ba['r2_sep'][i] for ba in batch_outputs])
            for i in range(num_steps)
        })
        outputs = {
            'inf/outputs': np.concatenate([ba['outputs'] for ba in batch_outputs], axis=0),
            'inf/targets': np.concatenate([ba['targets'] for ba in batch_outputs], axis=0),
            'inf/init_values': np.concatenate([ba['init_values'] for ba in batch_outputs], axis=0)
        }
        return {'scalar': scalar, **outputs}

    def log_plot(self, x, y):
        pass
