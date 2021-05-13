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
        # start_time = time.time()
        H = original_result.size()[2]
        W = original_result.size()[3]
        batches = original_result.size()[0]
        total_pixels = W * H
        mask_or_tiles = mask_original.flatten().view(batches, total_pixels, 1).repeat(1, 1, total_pixels)
        mask_tr_tiles = mask_transformed.repeat(1, total_pixels, 1, 1).view(batches, total_pixels, total_pixels)
        mask = (mask_or_tiles == mask_tr_tiles).type(torch.uint8)
        n_labels = mask.sum(2)
        channels = original_result.shape[1]
        features_dot = torch.einsum('bik, bjk -> bij',
                                    original_result.permute(0, 2, 3, 1).view(batches, total_pixels, channels),
                                    transformed_result.permute(0, 2, 3, 1).view(batches, total_pixels, channels))
        features_dot = torch.div(features_dot, self.temperature)
        features_dot = torch.exp(features_dot)
        sum = torch.sum(features_dot, 2).view(batches, total_pixels, 1).repeat(1, 1, total_pixels)
        div = torch.div(features_dot, sum)
        log = torch.log(div)
        prod = mask * log
        sum_q = prod.sum(2)
        sum_p = torch.zeros(batches, total_pixels, device=self.device)
        sum_p[n_labels != 0] = torch.div(sum_q[n_labels != 0], n_labels[n_labels != 0])
        loss = torch.div(sum_p.sum(1), -total_pixels)
        # print(f"loss time:{time.time() - start_time}")
        # print(torch.mean(loss))
        return torch.mean(loss)
