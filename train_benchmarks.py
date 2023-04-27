import os
import copy
import torch

from torch import nn
from torch import optim

from tqdm import tqdm
from utils.config import DATA_PATH, MODEL_PATH
from utils.data_utils import get_intermediate, get_mnist
from utils.model_utils import ConceptModel, ConvNet
from utils.data_utils import get_intermediate
from utils.benchmarks import preprocess_intermediate, get_pca, get_k_means


def train_fixed_concepts(model, concept_model, train_dataloader, test_dataloader, optimizer, epochs=100, early_stopping=0):
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
            concept_prod = concept_model(inter.transpose(-1, -2))
            logits = classifier(concept_prod)

            # loss
            loss = loss_func(logits, y) 

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
        print(f'Test loss: {val_loss}\nTest acc: {val_acc}')

        if early_stopping:
            if val_loss < min_val_loss:
                t = epoch
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)

            elif t + early_stopping <= epoch:
                print(f'early stopping, retrieving model from epoch {t}')
                return best_model
    return model


if __name__ == "__main__":
    try:
        os.mkdir(MODEL_PATH / 'kmeans')
        os.mkdir(MODEL_PATH / 'pca')
    except FileExistsError:
        pass

    k_max = 10

    model = ConvNet()
    model.load_state_dict(torch.load(MODEL_PATH / 'cnn.pt'))

    data = get_intermediate(DATA_PATH, model)
    data = preprocess_intermediate(data)

    pca = get_pca(data)

    kmeans = {}

    for k in range(2, k_max+1):
        kmeans[k] = get_k_means(data, k)


    train_set = get_mnist(DATA_PATH)
    test_set = get_mnist(DATA_PATH, train=False)

    for C in range(2, 11):
        concept_model = ConceptModel(C)

        pca_concepts = pca[:C]
        concept_model.load_concepts(pca_concepts)
        concept_model.freeze_concepts()

        optimizer = optim.Adam(
            concept_model.parameters(),
            lr=0.01,
            betas=[0.9, 0.999]
        )

        concept_model = train_fixed_concepts(model, concept_model, train_set, test_set, optimizer, epochs=100, early_stopping=1)
        torch.save(model.state_dict(), MODEL_PATH / 'pca' / f'{C}_concept_model.pt')

        concept_model = ConceptModel(C)

        kmeans_concepts = kmeans[C]
        concept_model.load_concepts(kmeans_concepts)
        concept_model.freeze_concepts()

        optimizer = optim.Adam(
            concept_model.parameters(),
            lr=0.01,
            betas=[0.9, 0.999]
        )

        concept_model = train_fixed_concepts(model, concept_model, train_set, test_set, optimizer, epochs=100, early_stopping=1)
        torch.save(model.state_dict(), MODEL_PATH / 'kmeans' / f'{C}_concept_model.pt')

    os.remove(DATA_PATH / 'intermediate.pt')

