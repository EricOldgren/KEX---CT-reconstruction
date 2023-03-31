import torch
import torch.nn as nn


lin = nn.Linear(2,2)


X = torch.tensor([
    [[1,2], [1,2], [1,2]],

    [[1,2], [1,2], [1,2]]
    
    ], dtype=torch.float32)


print(lin(torch.tensor([1,2], dtype=torch.float32)[None]))

print(lin(X))