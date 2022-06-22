import torch
from sequitur.models import LSTM_AE
from sequitur import quick_train

model = LSTM_AE(
  input_dim=2,
  encoding_dim=7,
  h_dims=[64],
  h_activ=None,
  out_activ=None
)

# 学習
train_set = [torch.randn(10,2) for _ in range(100)]
quick_train(LSTM_AE, train_set, epochs = 3, encoding_dim=7)

x = torch.randn(10, 2) # Sequence of 10 3D vectors
z = model.encoder(x) # z.shape = [7]
x_prime = model.decoder(z, seq_len=10) # x_prime.shape = [10, 3]


print(x_prime)