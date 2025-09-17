import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hw1.fgsm import adv_x

# fix seed so that random initialization always performs the same
torch.manual_seed(13)


# create the model N as described in the question
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

# random input
x = torch.rand((1,10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # this is required so we can compute the gradient w.r.t x

t = 1 # target class

epsReal = 0.5  #depending on your data this might be large or small
eps = epsReal - 1e-7 # small constant to offset floating-point erros

# The network N classfies x as belonging to class 2
original_class = N(x).argmax(dim=1).item()  # TO LEARN: make sure you understand this expression
print("Original Class: ", original_class)
assert(original_class == 2)

# PGD initialization: random point within the eps-ball around x
adv_x = x.clone().detach() + torch.zeros_like(x).normal_(mean=0.0, std=1/np.sqrt(10)).clamp(-eps, eps)
assert( torch.norm((x-adv_x), p=float('inf')) <= epsReal)

# iterative FGSM attack initialized at x
# adv_x = x.clone().detach()

num_steps = 10000
alpha = 2
for step in range(num_steps):
    if torch.rand(1).item() < 0.001:
        alpha = alpha*0.9
    adv_x.requires_grad_()
    logits = N(adv_x)
    L = nn.CrossEntropyLoss()
    loss = L(logits, torch.tensor([t], dtype=torch.long))
    loss.backward()
    with torch.no_grad():
        assert (adv_x.grad is not None)
        adv_x = (adv_x - adv_x.grad.sign()*alpha).clamp(min=x-eps, max=x+eps)
        logits = N(adv_x)
        print("logits after clamping: ", logits)
        new_class = logits.argmax(dim=1).item()
        if new_class == t:
            print("Attack succeeded at step ", step)
            print("x: ", x)
            print("adv_x: ", adv_x)
            print("diff: ", adv_x - x)
            print("logits: ", logits)
            break

new_class = N(adv_x).argmax(dim=1).item()
print("New Class: ", new_class)
assert(new_class == t)
# it is not enough that adv_x is classified as t. We also need to make sure it is 'close' to the original x.
print(torch.norm((x-adv_x),  p=float('inf')).data)
assert( torch.norm((x-adv_x), p=float('inf')) <= epsReal)