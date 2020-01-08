import torch
from torch.optim import SGD
from detectron2.solver.lr_scheduler import WarmupMultiStepLR, _get_warmup_factor_at_iter
from bisect import bisect_right


class CaffeSGD(SGD):
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                d_p.mul_(group['lr'])
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-1, d_p)

        return loss

class CaffeLRScheduler(WarmupMultiStepLR):
    def _get_lr_ratio(self) -> float:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
    
    def step(self, epoch=None):
        super().step(epoch)

        #Adjust Momentum
        factor = 1. / self._get_lr_ratio()
        for param in self.optimizer.param_groups:
            p_keys = param['params']
            for p_key in p_keys:
                if 'momentum_buffer' in self.optimizer.state[p_key].keys():
                    self.optimizer.state[p_key]['momentum_buffer'] *= factor