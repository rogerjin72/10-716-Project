import os
import json
import numpy as np
import torch
import torch.nn as nn

from utils.config import MODEL_PATH, DATA_PATH, PATH
from utils.model_utils import ConceptModel, ConvNet
from utils.data_utils import get_mnist, get_intermediate, get_mnist_image
from utils.benchmarks import preprocess_intermediate
from tqdm import tqdm
import matplotlib.patches as patches
from matplotlib import pyplot as plt


def get_val_accuracy(model, test_dataloader):
    val_loss = 0
    val_acc = 0
    n = 0
    loss_func = nn.CrossEntropyLoss()

    model.eval()
    for x, y in tqdm(test_dataloader, total=len(test_dataloader)):
        with torch.no_grad():
            logits = model(x)
            batch_loss = loss_func(logits, y)
            val_loss += batch_loss.item() * len(y)

            batch_acc = logits.argmax(axis=1) == y
            batch_acc = batch_acc.sum().item()
            val_acc += batch_acc

            n += len(y)

    val_loss = val_loss / n
    val_acc = val_acc / n
    print(f'Test loss: {val_loss}\nTest acc: {val_acc}')
    return val_acc


def plot_completeness_scores(model, test_set):
    test_acc = get_val_accuracy(model, test_set)
    r = 0.1

    comp_aware = json.load(open(MODEL_PATH / 'concept_aware' / 'test_accuracies.json'))    
    kmeans = json.load(open(MODEL_PATH / 'kmeans' / 'test_accuracies.json'))    
    pca = json.load(open(MODEL_PATH / 'pca' / 'test_accuracies.json'))    

    comp_aware = np.array(comp_aware)
    kmeans = np.array(kmeans)
    pca = np.array(pca)

    fig, ax = plt.subplots()

    dims = np.arange(2, 11)
    ax.plot(dims, (comp_aware - r)/ (test_acc - r), label='Completeness Aware', color='#2A9D8F')
    ax.plot(dims, (pca - r)/ (test_acc - r), label='PCA', color='#E9C46A')
    ax.plot(dims, (kmeans - r)/ (test_acc - r), label='K-Means', color='#E76F51')
    ax.legend()
    ax.set_xlabel('Number of Concepts')
    ax.set_ylabel('Completeness Score')
    fig.tight_layout()
    plt.savefig(PATH / 'figures'/ 'Completeness Scores.png', dpi=250)


def get_concept_nns(intermediate, concept_model, name, k=3):
    concepts = concept_model.get_concepts()

    similarities = intermediate @ concepts.T
    d = similarities.shape[1]

    fig, axs = plt.subplots(d, k, figsize=(k*6, d*8))

    for col in range(similarities.shape[1]):
        top_k = torch.topk(similarities[:, col], k)
        for i, idx in enumerate(top_k.indices):
            ax = axs[col, i]
            im_id = idx.item() // 49
            im_loc = idx.item() % 49

            x =  4 * (im_loc // 7)
            y =  4 * (im_loc % 7)
            image = get_mnist_image(im_id, DATA_PATH)[0]
            
            if k - 2 * i == 1 or k - 2 * i == 1:
                ax.set_title('Concept {}'.format(col), fontdict={'fontsize': 40})

            im_max = 28
            im_min = 0

            x_min = max(im_min, x - 8)
            y_min = max(im_min, y - 8)
            
            x_max = min(im_max, x + 7)
            y_max = min(im_max, y + 7)

            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=3, edgecolor='r', facecolor='none')
            ax.imshow(image)
            ax.add_patch(rect)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    plt.savefig(PATH / 'figures' / f'{name}.png')


if __name__ == '__main__':
    concepts = 3

    model = ConvNet()
    model.load_state_dict(torch.load(MODEL_PATH / 'cnn.pt'))
    test_set = get_mnist(DATA_PATH, train=False)
    # plot_completeness_scores(model, test_set)

    inter = get_intermediate(DATA_PATH, model, save=True)
    pre = torch.Tensor(preprocess_intermediate(inter))
    
    concept_aware = ConceptModel(concepts)  
    concept_aware.load_state_dict(torch.load(MODEL_PATH / 'concept_aware' / f'{concepts}_concept_model.pt'))
    
    get_concept_nns(pre, concept_aware, f'{concepts}_concepts')
