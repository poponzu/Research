# Standard Library
from statistics import mean

# Third Party
import torch
from torch.nn import MSELoss
import tqdm
import wandb
###########
# UTILITIES
###########


def instantiate_model(model, train_set, encoding_dim, **kwargs):
    if model.__name__ in ("LINEAR_AE", "LSTM_AE"):
        return model(train_set[-1].shape[-1], encoding_dim, **kwargs)
    elif model.__name__ == "CONV_LSTM_AE":
        if len(train_set[-1].shape) == 3: # 2D elements
            return model(train_set[-1].shape[-2:], encoding_dim, **kwargs)
        elif len(train_set[-1].shape) == 4: # 3D elements
            return model(train_set[-1].shape[-3:], encoding_dim, **kwargs)


def train_model(model, train_set, verbose, lr, epochs, denoise):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # # 確かめるために追加
    # print(list(model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = MSELoss(size_average=False)
    criterion = MSELoss(reduction='mean')

    mean_losses = []
    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        model.train()

        # # Reduces learning rate every 50 epochs
        # if not epoch % 50:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = lr * (0.993 ** epoch)

        losses = []
        for x in train_set:
            # print("--for文の中のx-")
            # print(x)

            optimizer.zero_grad()

            # Forward pass
            x_prime = model(x)

            loss = criterion(x_prime, x)
            # print(loss)


            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        mean_loss = mean(losses)
        mean_losses.append(mean_loss)
        wandb.log({'epoch': epoch, 'loss': mean_loss})

        if verbose:
            print(f"Epoch: {epoch}, Loss: {mean_loss}")

        # wandb.save("mymodel.h5")

    return mean_losses


def get_encodings(model, train_set):
    model.eval()
    encodings = [model.encoder(x) for x in train_set]
    return encodings


######
# MAIN
######


def quick_train(model, train_set, encoding_dim, verbose=False, lr=1e-3,
                epochs=1, denoise=False, **kwargs):
    # 確かめるために追加
    # print(list(model.parameters()))

    # model = instantiate_model(model, train_set, encoding_dim, **kwargs)
    # print(model)
    losses = train_model(model, train_set, verbose, lr, epochs, denoise)
    encodings = get_encodings(model, train_set)

    return model.encoder, model.decoder, encodings, losses