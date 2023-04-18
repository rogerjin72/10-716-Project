import copy

import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_utils import get_mnist
from utils.config import DATA_PATH, MODEL_PATH
from utils.model_utils import ConvNet
from train import train_loop
from tqdm import tqdm


def train_loop(model, train_dataloader, test_dataloader, optimizer, epochs, early_stopping=0):
    if early_stopping:
        t = 0
        min_val_loss = torch.Tensor([float("inf")])
        best_model = None

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch += 1
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        model.train()
        for X, y in bar:
            optimizer.zero_grad()

            logits = model(X)
            loss = loss_func(logits, y)
            loss.backward()
            optimizer.step()
            bar.set_description(str(round(loss.item(), 4)))

        val_loss = 0
        val_acc = 0
        n = 0

        model.eval()
        for X, y in tqdm(test_dataloader, total=len(test_dataloader)):
            with torch.no_grad():
                logits = model(X)
                batch_loss = loss_func(logits, y)
                val_loss += batch_loss.item() * len(y)

                batch_acc = logits.argmax(axis=1) == y
                batch_acc = batch_acc.sum().item()
                val_acc += batch_acc

                n += len(y)

        val_loss = val_loss / n
        val_acc = val_acc / n
        print(f'Test loss: {val_loss}\n Test acc: {val_acc}')

        if early_stopping:
            if val_loss < min_val_loss:
                t = epoch
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)

            elif t + early_stopping <= epoch:
                print(f'early stopping, retrieving model from epoch {t + 1}')
                return best_model

    return model

if __name__ == '__main__':
    model = ConvNet()
    train_set = get_mnist(DATA_PATH)
    test_set = get_mnist(DATA_PATH, train=False)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.01,
        betas=[0.9, 0.999]
    )

    model = train_loop(model, train_set, test_set, optimizer, epochs=100, early_stopping=3)
    torch.save(model.state_dict(), MODEL_PATH / 'cnn.pt')