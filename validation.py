import numpy as np

from torch import nn
import torch
import torch.nn.functional as F
from metrics import AllInOneMeter
import time
import torchvision.transforms as transforms


def validation_binary(model: nn.Module, criterion, valid_loader, device):
    with torch.no_grad():
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        meter = AllInOneMeter(device)
        start_time = time.time()
        w1 = 1.0
        w2 = 0.5
        w3 = 0.5
        for valid_image, valid_mask, valid_mask_ind in valid_loader:
            valid_image = valid_image.to(device)  # [N, 1, H, W]
            valid_mask  = valid_mask.to(device).type(torch.cuda.FloatTensor)
            valid_image = valid_image.permute(0, 3, 1, 2)
            valid_mask  = valid_mask.permute(0, 3, 1, 2)
            valid_mask_ind = valid_mask_ind.to(device).type(torch.cuda.FloatTensor)

            outputs, outputs_mask_ind1, outputs_mask_ind2 = model(valid_image)
            valid_prob = F.sigmoid(outputs)
            valid_mask_ind_prob1 = F.sigmoid(outputs_mask_ind1)
            valid_mask_ind_prob2 = F.sigmoid(outputs_mask_ind1)
            loss1 = criterion(outputs, valid_mask)

            loss2 = F.binary_cross_entropy_with_logits(outputs_mask_ind1, valid_mask_ind)
            loss3 = F.binary_cross_entropy_with_logits(outputs_mask_ind2, valid_mask_ind)
            loss = loss1 * w1 + loss2 * w2 + loss3 * w3

            meter.add(valid_prob, valid_mask, valid_mask_ind_prob1, valid_mask_ind_prob2, valid_mask_ind,
                      loss1.item(), loss2.item(), loss3.item(), loss.item())

        valid_metrics = meter.value()
        epoch_time = time.time() - start_time
        valid_metrics['epoch_time'] = epoch_time

        valid_metrics['image'] = valid_image.data
        valid_metrics['mask'] = valid_mask.data
        valid_metrics['prob'] = valid_prob.data
    return valid_metrics