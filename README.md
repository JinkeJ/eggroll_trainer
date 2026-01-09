Eggroll and Evolution Strategy Implementation  

This implementation literally use (x @ A) @ B.T to avoid large matrix calculation
```
        x = nn.functional.linear(input, self.a_weight)   # in -> rank
        x = nn.functional.linear(x, self.b_weight, self.bias)  # rank -> out
```

[Reference](https://eshyperscale.github.io/)
