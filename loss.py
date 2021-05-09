import math
import time

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


    def __call__(self, original_result, transformed_result, mask_original, mask_transformed):
        #start_time = time.time()
        # print(mask_original)
        # print(mask_transformed)
        # if(torch.all(mask_transformed == 100)):
        #     print(f"gg : {transformed_result}")
        H = original_result.size()[2]
        W = original_result.size()[3]
        batches = original_result.size()[0]
        total_pixels = W * H
        loss = torch.zeros(batches, device=self.device)
        for b in range(batches):
            mask_or_tiles = mask_original[b].flatten().view(total_pixels, 1).repeat(1, total_pixels)
            mask_tr_tiles = mask_transformed[b].repeat(total_pixels, 1, 1).view(total_pixels, total_pixels)
            mask = (mask_or_tiles == mask_tr_tiles).type(torch.uint8)
            n_labels = mask.sum(1)
            channels = original_result.shape[1]
            features_dot = torch.einsum('ik, jk -> ij', original_result[b].permute(1, 2, 0).view(total_pixels, channels),
                                        transformed_result[b].permute(1, 2, 0).view(total_pixels, channels))
            features_dot = torch.div(features_dot, self.temperature)
            features_dot = torch.exp(features_dot)
            sum = torch.sum(features_dot, 1).view(total_pixels, 1).repeat(1, total_pixels)
            div = torch.div(features_dot, sum)
            log = torch.log(div)
            prod = mask * log
            sum_q = prod.sum(1)
            sum_p = torch.zeros(total_pixels, device=self.device)
            sum_p[n_labels != 0] = torch.div(sum_q[n_labels != 0], n_labels[n_labels != 0])
            loss[b] = torch.div(sum_p.sum(), -total_pixels)
        #print(f"loss time:{time.time() - start_time}")
        #print(torch.mean(loss))
        return torch.mean(loss)
