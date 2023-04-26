import torch
import os

from utils.config import DATA_PATH, MODEL_PATH
from utils.data_utils import get_intermediate
from utils.model_utils import ConceptModel, ConvNet
from utils.data_utils import get_intermediate
from utils.benchmarks import preprocess_intermediate, get_pca, get_k_means


if __name__ == "__main__":
    k_max = 10

    model = ConvNet()
    model.load_state_dict(torch.load(MODEL_PATH / 'cnn.pt'))

    data = get_intermediate(DATA_PATH, model)
    data = preprocess_intermediate(data)

    pca = get_pca(data)

    k_means = {}
    for k in range(2, k_max+1):
        k_means[k] = get_k_means(data, k)

    # os.remove(DATA_PATH / 'intermediate.pt')

    