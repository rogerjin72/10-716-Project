import torch
import torch.nn as nn
import torch.nn.functional as F


def concept_sim(concepts):
    concepts = F.normalize(concepts)
    m, _ = concepts.shape
    
    # get similarities
    sims = concepts @ concepts.T
    # sum cross-terms
    sims.fill_diagonal_(0)
    return sims.mean()


def top_k_sim(concepts, inter, K):
    concepts = F.normalize(concepts)
    m, _ = concepts.shape

    # get per concept similarities
    prod = torch.matmul(concepts, inter)
    prod = prod.transpose(0, 1).flatten(1)

    # get top k
    top_k, _ = prod.topk(K)
    return top_k.sum() / (m * K)
