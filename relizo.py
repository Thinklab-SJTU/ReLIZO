from functools import partial
import numpy as np
import torch
from torch.optim import Optimizer

def _backtracking(obj_func, obj_init, x_init, d, init_step=1.0, shrink_rate=0.2, c1=0.1, max_ls=10, record_num=None):
    ls_iter = 0
    step = init_step
    d_norm = d.norm()
    done = False
    while ls_iter < max_ls:
        new_obj = obj_func(x_init, step, d)
        if new_obj <= obj_init + c1*step*d_norm:
            done = True
            break
        else: step *= shrink_rate
        ls_iter += 1
    if record_num is not None: record_num.append(ls_iter)

    return step


class LIZO(Optimizer):
    ZO = True
    """
    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).
    """
    def __init__(self, params, lr=1e-3, weight_decay=0, num_sample_per_step=8, reuse_distance_bound=0., max_reuse_rate=1., sample_norm=1e-5, orthogonal_sample=True, sample_momentum=0., fast_alg=True, line_search_fn=None, strict_lr=False):

        defaults = dict(lr=lr,
                        weight_decay=weight_decay,
                        state={},
                        line_search_fn=line_search_fn
                        )
        super(LIZO, self).__init__(params, defaults)
        self._params = self.param_groups[0]['params']
        self.numel_params = sum([p.numel() for p in self._params])

        self.num_sample_per_step = min(num_sample_per_step, self.numel_params)
        if reuse_distance_bound is None:
            self.reuse_distance_bound = lr
        else: self.reuse_distance_bound = reuse_distance_bound
        self.max_reuse_rate = max_reuse_rate
        self.orthogonal_sample = orthogonal_sample
        self.sample_norm = sample_norm
        self.sample_momentum = sample_momentum
        self.fast_alg = fast_alg
        self.num_reuse = []
        self.num_line_search_query = []
        self.strict_lr = strict_lr

        #TODO: switch one-point/two-point difference

        #TODO: add momentum

    def _reset_state(self, state):
        device = self._params[0].device
        state['last_delta_samples'] = torch.zeros(self.num_sample_per_step, self.numel_params, device=device)
        state['sample_lr'] = torch.zeros(self.num_sample_per_step, device=device)
        state['sample_obj'] = torch.zeros(self.num_sample_per_step, device=device)
        state['last_obj'] = None
        state['last_lr'] = None
        state['last_grad'] = None
        state['dist_matrix'] = torch.zeros(self.num_sample_per_step, self.num_sample_per_step, device=device)


    def _flat_param(self, params):
        views = []
        for p in params:
            views.append(p.data.view(-1))
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        # print("step_size: ", step_size)
        # print("update: ", update.norm())
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self.numel_params

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d, weight_decay=0):
        self._add_grad(t, d)
        loss = float(closure())
        if weight_decay > 0:
            loss += weight_decay * float(self._flat_param(self._params).norm().pow(2))
        self._set_param(x)
        return loss

    #TODO: Euclidean distance
    def get_distance(self, delta_samples, distance_mode='euclidean'):
        return delta_samples.norm(p='fro', dim=-1)

    def _sample(self, num_to_samples, sample_dim, device='cpu', mean=0., std=1e-3):
        return torch.randn(num_to_samples, sample_dim, device=device)*std + mean
#        if mean is None:
#            # directly sample
#            new_delta_samples = torch.randn(num_to_samples, sample_dim, device=device)
#        else:
#            new_delta_samples = torch.normal(mean, std=std, size=(sample_dim,))
#        return new_delta_samples

    #sample points
    def sample(self, last_delta_samples, num_to_samples, sample_dim, orthogonal=True, device='cpu', mean=None, std=1e-3):
        if orthogonal:
            if last_delta_samples is None:
                num_all = num_to_samples
            else:
                num_all = last_delta_samples.shape[0] + num_to_samples
                last_delta_samples = last_delta_samples.t()
            _num = num_to_samples
            while _num > 0:
                new_delta_samples = self._sample(_num, sample_dim, device=device, mean=mean, std=std).t()
                if last_delta_samples is not None:
                    new_delta_samples = torch.cat([last_delta_samples, new_delta_samples], dim=1)
                last_delta_samples, _ = torch.linalg.qr(new_delta_samples)
                _num = num_all - last_delta_samples.shape[1]
            new_delta_samples = last_delta_samples[:,-num_to_samples:].t()
            new_lr = torch.ones(num_to_samples) * std
            new_delta_samples.div_(new_delta_samples.norm())
        else:
            # directly sample
#            new_delta_samples = torch.randn(num_to_samples, sample_dim, device=device)
            new_delta_samples = self._sample(num_to_samples, sample_dim, device=device, mean=mean, std=std)
            new_lr = (new_delta_samples.norm(dim=-1)+1e-8)
            new_delta_samples.div_(new_lr.view(-1,1))
        return new_delta_samples, new_lr

    @torch.no_grad()
    def step(self, closure):
        assert len(self.param_groups) == 1

        device = self._params[0].device
        current_obj = float(closure())
        group = self.param_groups[0]
        sample_norm = self.sample_norm
        lr = group['lr']
        weight_decay = group['weight_decay']
        line_search_fn = group['line_search_fn']
        x_init = self._clone_param()

        state = group['state']
        last_delta_samples = state.get('last_delta_samples', torch.zeros(self.num_sample_per_step, self.numel_params, device=device))
        sample_lr = state.get('sample_lr', torch.zeros(self.num_sample_per_step, device=device))
        sample_obj = state.get('sample_obj', torch.zeros(self.num_sample_per_step, device=device))
        last_obj = state.get('last_obj', None)
        last_lr = state.get('last_lr', None)
        last_grad = state.get('last_grad', None)
        dist_matrix = state.get('dist_matrix', torch.zeros(self.num_sample_per_step, self.num_sample_per_step, device=device))

        sample_obj = torch.cat([sample_obj, torch.tensor([last_obj], device=device) if last_obj is not None else torch.tensor([0], device=device)], dim=0)
        sample_lr = torch.cat([sample_lr, torch.tensor([0], device=device)], dim=0)
        last_delta_samples = torch.cat([last_delta_samples, torch.zeros(1, self.numel_params, device=device)], dim=0)

        reuse_last = last_grad is not None and self.reuse_distance_bound > 0
        # get reused samples from last samples
        if reuse_last:
            history_delta_samples = last_delta_samples
            last_delta_samples = last_delta_samples.mul(sample_lr.view(-1,1))+last_grad.mul(last_lr).view(1,-1)
            distances = self.get_distance(last_delta_samples)
            # norm last_delta_samples, which is not necessary but can be good to compute inverse
            history_sample_lr = sample_lr
            sample_lr = (last_delta_samples.norm(dim=-1) + 1e-8)
            last_delta_samples.div_(sample_lr.view(-1, 1))
            sample_idx = torch.where(distances < self.reuse_distance_bound)[0]
            # if all samples can be reused then remove the farthest, since last_sample is added to the reused samples 
            if len(sample_idx) > self.num_sample_per_step*self.max_reuse_rate:
                sample_idx = torch.argsort(distances, dim=0, descending=False)[:int(self.num_sample_per_step*self.max_reuse_rate)]
        else: 
            sample_idx = []
#        print('distance', distances)
#        print('sample_idx', sample_idx)
        self.num_reuse.append(len(sample_idx))
        if len(sample_idx) > 0: 
            print(f"Reuse {len(sample_idx)} samples")

        # random sample (orthogonal) points
        num_random = self.num_sample_per_step - len(sample_idx)
        if num_random > 0:
            if self.sample_momentum > 0 and last_grad is not None:
                mean = self.sample_momentum * last_lr * last_grad 
            else: mean = 0.
            new_delta_samples, new_lr = self.sample(last_delta_samples[sample_idx] if len(sample_idx)>0 else None, num_random, self.numel_params, orthogonal=self.orthogonal_sample, device=device, mean=mean, std=sample_norm)
#            new_lr.mul_(sample_norm)
            # print('new_lr', new_lr)

        if len(sample_idx) > 0:
            last_delta_samples[:len(sample_idx)] = last_delta_samples[sample_idx]
            sample_obj[:len(sample_idx)] = sample_obj[sample_idx]
            sample_lr[:len(sample_idx)] = sample_lr[sample_idx]
            history_sample_lr = history_sample_lr[sample_idx]
        if num_random > 0:
            last_delta_samples[-num_random-1:-1] = new_delta_samples
            sample_lr[-num_random-1:-1] = new_lr
            # get object of the new sampled points
            for idx in range(num_random, 0, -1):
                # print(new_lr[-idx])
                # print(new_delta_samples[-idx].norm())
                sample_obj[-idx-1] = self._directional_evaluate(closure, x_init, new_lr[-idx], new_delta_samples[-idx], 0.)
        last_delta_samples = last_delta_samples[:self.num_sample_per_step]
        sample_lr = sample_lr[:self.num_sample_per_step]
        sample_obj = sample_obj[:self.num_sample_per_step]

        # compute dist_matrix: \delta_w * \delta_w^\top
        if len(sample_idx) > 0 and self.fast_alg:
            # fast algorithm to compute dist_matrix
            tmp = history_delta_samples[sample_idx] @ last_grad
            if self.num_sample_per_step in sample_idx:
                # when the last sample is reused, then add new row/column to the last row/column of dist_matrix for fast algorithm to obtain the current dist_matrix
                new_vec = torch.zeros(self.num_sample_per_step+1, device=device).scatter_(dim=0, index=sample_idx, src=tmp)
                dist_matrix = torch.cat([dist_matrix, new_vec[1:].view(1,-1)], dim=0)
                dist_matrix = torch.cat([dist_matrix, new_vec.view(-1,1)], dim=1)

            tmp_lr = sample_lr[:len(sample_idx)]
            tmp.mul_(history_sample_lr).mul_(last_lr)
            dist_matrix[:len(sample_idx),:len(sample_idx)] = (dist_matrix[sample_idx][:,sample_idx].mul(history_sample_lr.view(-1,1)@history_sample_lr.view(1,-1)) + tmp.view(-1, 1) + tmp.view(1, -1) + last_lr*last_lr).div(tmp_lr.view(-1,1)@tmp_lr.view(1,-1)+1e-16)
            dist_matrix = dist_matrix[:self.num_sample_per_step][:,:self.num_sample_per_step]
            if num_random > 0:
                if self.orthogonal_sample:
                    dist_matrix[-num_random:] = 0
                    dist_matrix[:, -num_random:] = 0
                    # only need the diagonal items
                    vector = (new_delta_samples * new_delta_samples).sum(dim=-1)
                    dist_matrix.diagonal()[-num_random:] = vector
                else:
                    dist_matrix[-num_random:] = last_delta_samples[-num_random:] @ last_delta_samples.t()
                    dist_matrix[:, -num_random:] = dist_matrix[-num_random:].t()
#            gt_dist_matrix = last_delta_samples @ last_delta_samples.t()
#            print(gt_dist_matrix-dist_matrix)

        else:
            dist_matrix = last_delta_samples @ last_delta_samples.t()

        reset = False
        try:
#            last_grad = dist_matrix.inverse() @ (sample_obj-current_obj).t()
#            last_grad = torch.linalg.solve(dist_matrix, (sample_obj-current_obj).t().div_(sample_lr))
            last_grad = torch.linalg.lstsq(dist_matrix, (sample_obj-current_obj).t().div_(sample_lr)).solution # use pseudoinverse
            last_grad = last_delta_samples.t() @ last_grad
        except Exception as e: 
            raise(e)
            self._reset_state(state)
            return current_obj

        # add weight decay to grad
        if weight_decay != 0:
            last_grad = last_grad.add(self._flat_param(self._params), alpha=weight_decay)

        #TODO: line search for proper lr
        grad_norm = last_grad.norm()
        last_grad.div_(grad_norm)
        # print("grad_norm: ", grad_norm)
        # print("lr: ", lr)
        if not self.strict_lr:
            lr *= grad_norm
        # print(lr)
        # print(last_grad.norm())

        if line_search_fn is not None:
            def obj_func(x, t, d):
                return self._directional_evaluate(closure, x, t, d, 0.)
            lr = line_search_fn(obj_func, current_obj, x_init, last_grad.neg(), init_step=lr, record_num=self.num_line_search_query)
#        else:
#        if lr > 10: reset = True

#        new_obj = self._directional_evaluate(closure, x_init, lr, last_grad.neg(), 0.)
#        if np.isnan(new_obj):
        if torch.any(torch.isnan(last_grad)):
            print('Gradient has NaN, so the parameters will not be updated in this iteration.')
            reset = True

        if reset:
            self._reset_state(state)
        else:
            # print("lr: ", lr)
            # print("grad_norm: ", last_grad.norm())
            self._add_grad(lr, last_grad.neg())
            state['last_delta_samples'] = last_delta_samples
            state['sample_obj'] = sample_obj
            state['sample_lr'] = sample_lr
            state['last_obj'] = current_obj
            state['last_lr'] = lr
            state['last_grad'] = last_grad
            state['dist_matrix'] = dist_matrix
    
        return current_obj

    
if __name__ == '__main__':
    from src.benchmark.object import Benchmark_func
    num_var = 6
#    obj = Benchmark_func(function='rosenbrock', num_var=num_var, init_point=np.ones(num_var)*2.0)
    obj = Benchmark_func(function='sphere', num_var=num_var, init_point=np.random.randn(num_var)*2.0)
#    obj = Benchmark_func(function=lambda x: sum(x), num_var=num_var, init_point=np.random.randn(num_var)*2.0)
    lizo = LIZO(obj.parameters(), lr=0.1, num_sample_per_step=num_var, sample_norm=1e-5, reuse_distance_bound=5e-2, max_reuse_rate=0.5, orthogonal_sample=False, fast_alg=True, 
            line_search_fn=partial(_backtracking))
    for step in range(20):
        loss = obj()
        print(loss.item(), lizo._params)
        print('Compute gradient by backward')
        loss.backward()
        grad = torch.cat([p.grad.data.view(-1) for p in obj.parameters()], 0)
        #zero_grad
        for p in obj.parameters(): p.grad.zero_()
        print('Compute gradient by difference')
        diff_grad = torch.zeros_like(grad)
        sample_norm = 1e-5
        x_init = lizo._clone_param()
        delta = torch.eye(num_var)
        with torch.no_grad():
            for idx in range(num_var):
                sample_loss = lizo._directional_evaluate(obj, x_init, sample_norm, delta[idx])
                diff_grad[idx] = (sample_loss - loss) / sample_norm
        print('Compute gradient by LIZO')
        lizo.step(closure=obj)
        
        print('Compare gradient')
        lizo_grad = lizo.param_groups[0]['state']['last_grad']
        lr = lizo.param_groups[0]['state']['last_lr']
        print(grad, lizo_grad, diff_grad)
        print(grad.div(grad.norm(dim=-1, keepdim=True)), lizo_grad.div(lizo_grad.norm(dim=-1, keepdim=True)), diff_grad.div(diff_grad.norm(dim=-1, keepdim=True)))
        print(grad.norm(dim=-1), lr, diff_grad.norm(dim=-1))



