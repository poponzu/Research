from torch import nn

model = nn.Linear(3, 3)
print(list(model.parameters()))