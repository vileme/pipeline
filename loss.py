import numpy as np
import torch
from torch import nn

class LossBinary:
    """
    Loss defined as BCE - log(soft_jaccard)
    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        # print('outputs.type', outputs.type())
        # print('targets.type', targets.type())
        # print('outputs.shape', outputs.shape)
        # print('targets.shape', targets.shape)
        loss = self.nll_loss(outputs, targets)
        # print('loss.type', loss.type())
        # print('loss.shape', loss.shape)
        # print(loss)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = torch.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss



class ContrastiveLoss:
    def __init__(self, temperature, device):
        self.temperature = temperature
        self.device = device

    def count_labels(self, class_of_p, mask_transformed):
        torch.set_printoptions(threshold=10000)
        mask = (torch.tensor(mask_transformed == class_of_p.item(), device=self.device)).type(torch.int)
        n_labels = torch.sum(mask)
        return mask, n_labels

    def __call__(self, original_result, transformed_result, mask_original, mask_transformed):
        H = original_result.size()[2]
        W = original_result.size()[3]
        batches = original_result.size()[0]
        loss = []
        total_pixels = W * H
        for b in range(batches):
            results = torch.zeros((64,64), device=self.device)
            for p_i in range(H):
                for p_j in range(W):
                    class_of_p_in_original = mask_original[b, :, p_i, p_j]
                    mask, n_labels = self.count_labels(class_of_p_in_original, mask_transformed[b])
                    if n_labels == 0:
                        results[p_i][p_j] = 0
                        continue
                    preprod = torch.matmul(transformed_result[b].permute(1, 2, 0), original_result[b, :, p_i, p_j])
                    preprod = torch.exp(preprod)
                    denominator = torch.sum(preprod)
                    preprod = torch.log(torch.div(preprod, denominator))
                    res = torch.div(torch.sum(mask * preprod), n_labels)
                    results[p_i][p_j] = res
            loss.append(torch.div(torch.sum(results), (-total_pixels)).item())
        return torch.tensor(np.mean(loss), requires_grad=True)
