import torch
import lstm_ae
import quick_train
from torch.nn import MSELoss

import wandb

# wandb.login()


project_name = 'Research'
group_name = 'LSTM-AE'
experiment_name = 'AEの性能評価3'
wandb.init(
    project=project_name,
    group=group_name,
    name=experiment_name,
    config={
        "loss": "mean_loss",
        "metric": "MSE",
        "epoch": 100,
        "lr": 1e-3
    })
config = wandb.config

model = lstm_ae.LSTM_AE(
    input_dim=2,
    encoding_dim=7,
    h_dims=[64],
    h_activ=None,
    out_activ=None
)

# 学習データ生成
train_set = [torch.randn(4, 2) for _ in range(100)]
# print(train_set)
encoder, decoder, _, _ = quick_train.quick_train(model, train_set, verbose=True, epochs=100,
                                                 encoding_dim=4)

# データを作成してモデルに入れてみる
x = torch.randn(4, 2)  # Sequence of 10 3D vectors
z = model.encoder(x)  # z.shape = [7]
x_prime = model.decoder(z, seq_len=4)  # x_prime.shape = [10, 2]

# print(x)
# print(x_prime)

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))


print("性能は" + str(RMSELoss(x, x_prime)))
