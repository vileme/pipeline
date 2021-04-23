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


def count_labels(class_of_p, mask_transformed):
    return torch.sum(mask_transformed == class_of_p.item()).item()


class ContrastiveLoss:
    def __init__(self, temperature, device):
        self.temperature = temperature
        self.device = device

    def __call__(self, original_result, transformed_result, mask_original,
                 mask_transformed):
        H = original_result.size()[2]
        W = original_result.size()[3]
        batches = original_result.size()[0]
        loss = []
        total_pixels = W * H
        denominators = torch.zeros(size = (batches, H, W), dtype = torch.float, device = self.device)
        for b in range(batches):
            for p_i in range(H):
                for p_j in range(W):
                    feature_product = torch.dot(original_result[b, :, p_i, p_j],transformed_result[b].sum(axis=1).sum(axis=1))
                    denominators[b, p_i, p_j] = torch.div(feature_product, self.temperature)
        print("end")
        for b in range(batches):
            sum1 = 0
            for p_i in range(H):
                for p_j in range(W):
                    print(p_i, p_j)
                    class_of_p_in_original = mask_original[b, :, p_i, p_j]
                    class_of_p_in_transformed = count_labels(class_of_p_in_original, mask_transformed)
                    sum2 = 0
                    for q_i in range(H):
                        for q_j in range(W):
                            if torch.equal(class_of_p_in_original, mask_transformed[b, :, q_i, q_j]):
                                numerator = torch.div(torch.dot(original_result[b, :, p_i, p_j],
                                                               transformed_result[b, :, q_i, q_j]),
                                                      self.temperature)
                                denominator = denominators[b, p_i, p_j]
                                sum2 += torch.log(torch.div(numerator, denominator))
                    sum1 += torch.div(sum2, class_of_p_in_transformed)
            loss.append(torch.div(sum1, (-total_pixels)))
        return np.mean(loss)
