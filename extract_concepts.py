import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils.config import DATA_PATH, MODEL_PATH
from utils.data_utils import get_mnist
from utils.model_utils import ConvNet, ConceptModel
from utils.losses import top_k_sim, concept_sim
from tqdm import tqdm


def train_concepts(model, concept_model, train_dataloader, test_dataloader, optimizer, lambda1=0.1, lambda2=0.1, K=5, epochs=100, early_stopping=0):
    if early_stopping:
        t = 0
        min_val_loss = torch.Tensor([float("inf")])
        best_model = None

    classifier = model.linear
    loss_func = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        epoch +=1 

        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for x, y in bar:
            # intermediate layers
            with torch.no_grad():
                inter = model.get_intermediate(x)
                inter = inter.reshape(inter.shape[:2] + (-1,))

            # get predictions
            concepts = concept_model.get_concepts()
            concept_prod = concept_model(inter.transpose(-1, -2))
            logits = classifier(concept_prod)

            # losses
            ce_loss = loss_func(logits, y)
            sim_loss = top_k_sim(concepts, inter, K)
            dissim_loss = concept_sim(concepts)
            loss = ce_loss - lambda1 * sim_loss + lambda2 * dissim_loss

            # GD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.set_description(str(round(loss.item(), 4)))


        val_loss = 0
        val_acc = 0
        n = 0

        model.eval()
        for x, y in tqdm(test_dataloader, total=len(test_dataloader)):
            with torch.no_grad():
                inter = model.get_intermediate(x)
                inter = inter.reshape(inter.shape[:2] + (-1,))

                concept_prod = concept_model(inter.transpose(-1, -2))
                logits = classifier(concept_prod)

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
                print(f'early stopping, retrieving model from epoch {t}')
                return best_model
    return model


if __name__ == '__main__':
    model = ConvNet()
    model.load_state_dict(torch.load(MODEL_PATH / 'cnn.pt'))

    train_set = get_mnist(DATA_PATH)
    test_set = get_mnist(DATA_PATH, train=False)

    for C in range(2, 11):
        concept_model = ConceptModel(C)

        optimizer = optim.Adam(
            concept_model.parameters(),
            lr=0.01,
            betas=[0.9, 0.999]
        )

        concept_model = train_concepts(model, concept_model, train_set, test_set, optimizer, epochs=100, early_stopping=2)
        torch.save(model.state_dict(), MODEL_PATH / f'{C}_concept_model.pt')