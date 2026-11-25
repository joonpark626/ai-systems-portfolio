import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import random


a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
b = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)

e = a * b
print(e.item())