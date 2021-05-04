import argparse
import pickle
import time
import pandas as pd
import torch
import wandb
from torch import nn
from torch.backends import cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from dataset import make_pretrain_loader, make_loader
# from loss import LossBinary, ContrastiveLoss
from loss import LossBinary, ContrastiveLoss
from metrics import AllInOneMeter
from models import UNet16
from validation import validation_binary
import numpy as np
def save_weights(model, model_path, ep, step, train_metrics, valid_metrics):
    torch.save({'model': model.state_dict(), 'epoch': ep, 'step': step, 'valid_loss': valid_metrics['loss1'], 'train_loss': train_metrics['loss1']},
               str(model_path)
               )

def get_split(train_test_split_file='./data/train_test_id.pickle'):
    with open(train_test_split_file, 'rb') as f:
        train_test_id = pickle.load(f)

        train_test_id['total'] = train_test_id[['pigment_network',
                                                'negative_network',
                                                'streaks',
                                                'milia_like_cyst',
                                                'globules']].sum(axis=1)
        valid = train_test_id[train_test_id.Split != 'train'].copy()
        valid['Split'] = 'train'
        train_test_id = pd.concat([train_test_id, valid], axis=0)
    return train_test_id


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    server = True
    path_default = "/mnt/tank/scratch/vkozyrev/" if server else "e:/diploma/"
    arg('--jaccard-weight', type=float, default=1)
    arg('--t', type=float, default=0.07)
    arg('--pretrain-epochs', type=int, default=100)
    arg('--train-epochs', type=int, default=100)
    arg('--train-test-split-file', type=str, default='./data/train_test_id.pickle', help='train test split file path')
    arg('--pretrain-image-path', type=str, default= f'{path_default}ham10000/', help='train test split file path')
    arg('--pretrain-mask-image-path', type=str, default=f'{path_default}/ham_clusters_20/lab/20/', help="images path for pretraining")
    arg('--image-path', type=str, default=f'{path_default}task2_h5/', help="h5 images path for training")
    arg('--batch-size', type=int, default=8, help="n batches")
    arg('--workers', type=int, default=4, help="n workers")
    arg('--cuda-driver', type=int, default=1, help="cuda driver")
    arg('--lr', type=float, default=0.001, help="lr")
    args = parser.parse_args()

    # wandb.init(project="pipeline")
    # wandb.run.name = f"pipeline lr = {args.lr}\n pretrain epochs = {args.pretrain_epochs}\ntrain epochs = {args.train_epochs}"
    # wandb.run.save()


    cudnn.benchmark = True
    device = torch.device(f'cuda:{args.cuda_driver}' if torch.cuda.is_available() else 'cpu')

    num_classes = 5
    args.num_classes = 5
    model = UNet16(num_classes= num_classes, pretrained="vgg")
    model = nn.DataParallel(model, device_ids=[args.cuda_driver])
    model.to(device)


    center_layer = model.module.center_Conv2d
    for p in center_layer.parameters():
        p.requires_grad = False


    pretrain_mask_image_path = args.pretrain_mask_image_path
    pretrain_image_path = args.pretrain_image_path
    pretrain_loader = make_pretrain_loader(pretrain_image_path, pretrain_mask_image_path, args, shuffle=True)
    epoch = 1
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True)
    print(model)
    print('Start pretraining')
    # wandb.watch(model)
    for epoch in range(epoch, args.pretrain_epochs + 1):
        model.train()
        start_time = time.time()
        losses = []
        for ind, (id, image_original, image_transformed, mask_original, mask_transformed) in enumerate(pretrain_loader):
            criterion = ContrastiveLoss(args.t, device)
            train_image_original = image_original.permute(0, 3, 1, 2)
            train_image_transformed = image_transformed.permute(0, 3, 1, 2)
            train_image_original.to(device)
            train_image_transformed.to(device)
            mask_original.to(device).type(
                torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
            mask_transformed.to(device).type(
                torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
            original_result, _ = model(train_image_original)
            transformed_result, _ = model(train_image_transformed)
            loss = (criterion(original_result, transformed_result, mask_original, mask_transformed))
            losses.append(loss.item())
            print(
                f'epoch={epoch:3d},iter={ind:3d}, loss={loss.item():.4g}')
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        print(losses)
        avg_loss = np.mean(losses)
        wandb.log({"pretrain/loss": avg_loss})
        epoch_time = time.time() - start_time
        print(f"epoch time:{epoch_time}")
        scheduler.step(avg_loss)
    print("Pretraining ended")
    epoch = 1
    ## get train_test_id
    train_test_id = get_split(args.train_test_split_file)

    ## train vs. val
    print('--' * 10)
    print('num train = {}, num_val = {}'.format((train_test_id['Split'] == 'train').sum(),
                                                (train_test_id['Split'] != 'train').sum()
                                                ))
    print('--' * 10)
    image_path = args.image_path
    train_loader = make_loader(train_test_id, image_path, args, train=True, shuffle=True,
                               train_test_split_file=args.train_test_split_file)
    valid_loader = make_loader(train_test_id, image_path, args, train=False, shuffle=True,
                               train_test_split_file=args.train_test_split_file)
    if True:
        print('--' * 10)
        print('check data')
        train_image, train_mask, train_mask_ind = next(iter(train_loader))
        print('train_image.shape', train_image.shape)
        print('train_mask.shape', train_mask.shape)
        print('train_mask_ind.shape', train_mask_ind.shape)
        print('train_image.min', train_image.min().item())
        print('train_image.max', train_image.max().item())
        print('train_mask.min', train_mask.min().item())
        print('train_mask.max', train_mask.max().item())
        print('train_mask_ind.min', train_mask_ind.min().item())
        print('train_mask_ind.max', train_mask_ind.max().item())
        print('--' * 10)
    valid_fn = validation_binary
    criterion = LossBinary(jaccard_weight=args.jaccard_weight)
    meter = AllInOneMeter(device)


    model.module.projection_head = nn.Conv2d(32, num_classes, 1)
    center_layer = model.module.center_Conv2d
    for p in center_layer.parameters():
        p.requires_grad = True

    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True)
    print("Start fine tuning")
    for epoch in range(epoch, args.train_epochs + 1):
        model.train()
        start_time = time.time()
        meter.reset()
        w1 = 1.0
        w2 = 0.5
        w3 = 0.5
        for i, (train_image, train_mask, train_mask_ind) in enumerate(train_loader):
            train_image = train_image.permute(0, 3, 1, 2)
            train_mask = train_mask.permute(0, 3, 1, 2)
            train_image = train_image.to(device)
            train_mask = train_mask.to(device).type(
                torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
            train_mask_ind = train_mask_ind.to(device).type(
                torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
            outputs, outputs_mask_ind1 = model(train_image)
            outputs_mask_ind2 = nn.MaxPool2d(kernel_size=outputs.size()[2:])(outputs)
            outputs_mask_ind2 = torch.squeeze(outputs_mask_ind2, 2)
            outputs_mask_ind2 = torch.squeeze(outputs_mask_ind2, 2)
            train_prob = torch.sigmoid(outputs)
            train_mask_ind_prob1 = torch.sigmoid(outputs_mask_ind1)
            train_mask_ind_prob2 = torch.sigmoid(outputs_mask_ind2)
            loss1 = criterion(outputs, train_mask)
            loss2 = F.binary_cross_entropy_with_logits(outputs_mask_ind1, train_mask_ind)
            loss3 = F.binary_cross_entropy_with_logits(outputs_mask_ind2, train_mask_ind)
            loss = loss1 * w1 + loss2 * w2 + loss3 * w3
            print(f'epoch={epoch:3d},iter={i:3d}, loss1={loss1.item():.4g}, loss2={loss2.item():.4g}, loss={loss.item():.4g}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            meter.add(train_prob, train_mask, train_mask_ind_prob1, train_mask_ind_prob2, train_mask_ind,
                        loss1.item(), loss2.item(), loss3.item(), loss.item())
        epoch_time = time.time() - start_time
        train_metrics = meter.value()
        train_metrics['epoch_time'] = epoch_time
        train_metrics['image'] = train_image.data
        train_metrics['mask'] = train_mask.data
        train_metrics['prob'] = train_prob.data
        valid_metrics = valid_fn(model, criterion, valid_loader, device)

        wandb.log({"loss/loss": valid_metrics["loss"], "loss/loss1": valid_metrics["loss1"],
                   "loss/loss2": valid_metrics["loss2"],
                   "jaccard_mean/jaccard_mean": valid_metrics["jaccard"],
                   "jaccard_class/jaccard_pigment_network": valid_metrics["jaccard1"],
                   "jaccard_class/jaccard_negative_network": valid_metrics["jaccard2"],
                   "jaccard_class/jaccard_streaks": valid_metrics["jaccard3"],
                   "jaccard_class/jaccard_milia_like_cyst": valid_metrics["jaccard4"],
                   "jaccard_class/jaccard_globules": valid_metrics["jaccard5"]})
        scheduler.step(valid_metrics['loss1'])

if __name__ == '__main__':
    main()
