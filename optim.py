import torch


# custom optimizer
# that only uses the sign (+ or -) of the gradient, not its value
class GradSignOptimizer:
    def __init__(self, named_params, *, lr):
        self.named_params = [item for item in named_params]

        # initialize counts
        self.grad_counts = {}
        for n, p in self.named_params:
            self.grad_counts[n] = torch.randint(-64, 64, p.shape, dtype=torch.int8)

    def step(self):
        for n, p in self.named_params:
            new_count = 2 * (p.grad > 0) - 1
            self.grad_counts[n] -= (self.grad_counts[n] + 4) // 8
            self.grad_counts[n] += 8 * new_count  # max val will be +- 64 or so

            p.data -= lr * (self.grad_counts[n] / 64.0)

    def zero_grad(self):
        for n, p in self.named_params:
            p.grad = None
