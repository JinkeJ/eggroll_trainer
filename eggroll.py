import math
import random

import torch
import torch.nn as nn
from typing import Callable

class EggrollLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "rank"]
    in_features: int
    out_features: int
    rank: int
    a_weight: torch.Tensor
    b_weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.a_weight = nn.Parameter(
            torch.empty((rank, in_features), **factory_kwargs)
        )
        self.b_weight = nn.Parameter(
            torch.empty((out_features, rank), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.a_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.b_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.b_weight.mul_(1.0 / math.sqrt(self.rank))

        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass.
        """
        x = nn.functional.linear(input, self.a_weight)   # in -> rank
        x = nn.functional.linear(x, self.b_weight, self.bias)  # rank -> out
        x = x / math.sqrt(self.rank)
        return x

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None}"

class ESOptimizer:
    def __init__(self, model: nn.Module, fitness_fn: Callable, population, sigma, lr, rank=1, device="cpu"):
        self.model = model
        self.fitness_fn = fitness_fn
        self.population = population
        self.sigma = sigma
        self.lr = lr
        self.rank = rank
        self.step_counter = 0

        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
            self.model = self.model.to(self.device)
        
        self.replace_linear()
        self.params = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]

    def replace_linear(self):
        pass

    def step(self, data, target): #data: 1 batch
        original_params = {
            name: p.data.clone()
            for name, p in self.params
        }
        
        step_seed = self.step_counter * 1000000
        self.step_counter += 1

        all_fitness = []
        eps_list = []

        for i in range(self.population):
            rng = torch.Generator(device=self.device)
            rng.manual_seed(step_seed + i)

            eps_i = {}
            for name, p in self.params:
                eps = torch.randn(p.shape, dtype=p.dtype, layout=p.layout, device=p.device, generator=rng)
                p.data = original_params[name] + self.sigma * eps
                eps_i[name] = eps
            eps_list.append(eps_i)

            with torch.no_grad():
                fitness = self.fitness_fn(self.model, data, target)
                all_fitness.append(fitness)

        fitness = torch.tensor(all_fitness, device=self.device)
        rewards = (fitness - fitness.mean()) / (fitness.std() + 1e-8)

        with torch.no_grad():
            for name, p in self.params:
                grad = torch.zeros_like(p)
                for i in range(self.population):
                    grad += rewards[i] * eps_list[i][name]

                p.data = original_params[name] + (self.lr / (self.population * self.sigma)) * grad

        return fitness.mean().item()

class EggrollOptimizer(ESOptimizer):
    def replace_linear(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                parent = self.model
                *path, layer_name = name.split(".")

                for p in path:
                    parent = getattr(parent, p)

                dtype = next(module.parameters()).dtype
                new_layer = EggrollLinear(
                    module.in_features,
                    module.out_features,
                    self.rank,
                    module.bias is not None,
                    self.device,
                    dtype,
                )

                if module.bias is not None:
                    new_layer.bias.data.copy_(module.bias.data)

                setattr(parent, layer_name, new_layer)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)