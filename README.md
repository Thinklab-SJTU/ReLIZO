# ReLIZO
Official Implementation of NeurIPS 2024 paper:
"ReLIZO: Sample Reusable Linear Interpolation-based Zeroth-order Optimization"

This repository provides the base code for EasyNAS. The complete code will be coming soon.

# Abstract
Gradient estimation is critical in zeroth-order optimization methods, which aims to obtain the descent direction by sampling update directions and querying function evaluations. Extensive research has been conducted including smoothing and linear interpolation. The former methods smooth the objective function, causing a biased gradient estimation, while the latter often enjoys more accurate estimates, at the cost of large amounts of samples and queries at each iteration to update variables. This paper resorts to the linear interpolation strategy and proposes to reduce the complexity of gradient estimation by reusing queries in the prior iterations while maintaining the sample size unchanged. Specifically, we model the gradient estimation as a quadratically constrained linear program problem and manage to derive the analytical solution. It innovatively decouples the required sample size from the variable dimension without extra conditions required, making it able to leverage the queries in the prior iterations. Moreover, part of the intermediate variables that contribute to the gradient estimation can be directly indexed, significantly reducing the computation complexity. Experiments on both simulation functions and real scenarios (black-box adversarial attacks neural architecture search, and parameter-efficient fine-tuning for large language models), show its efficacy and efficiency.

## Usage
Our implementation is based on the PyTorch optimizer base class.
```python
from functools import partial

from relizo import LIZO

def main():
  model = ...
  dataloader = ...
  criterion = ...
  parameters = model.parameters()
  optimizer = LIZO(
              parameters, lr,
              num_sample_per_step=8,
              reuse_distance_bound=2*lr,
              max_reuse_rate=0.5,
              orthogonal_sample=False,
  )
  # Training process
  def _closure(data, model=model, criterion=criterion):
      logits = model(data)
      loss = criterion(logits, label)
      return loss
  for data, label in dataloader:
      optimizer.zero_grad()
      optimizer.step(closure=partial(_closure, data))
```

# Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{geng2024predictive,
  title={ReLIZO: Sample Reusable Linear Interpolation-based Zeroth-order Optimization},
  author={Wang, Xiaoxing and Qin, Xiaohan and Yang, Xiaokang and Yan, Junchi},
  booktitle={NeurIPS 2024},
  year={2024}
}
