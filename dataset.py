import os
import pickle
import random
from collections import defaultdict
import numpy as np
import torch
import torchvision.transforms.functional as TF
from keras_preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from wandb.old.summary import h5py


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort



class PretrainDataset(Dataset):
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path
        self.images = os.listdir(image_path)
        self.masks = os.listdir(mask_path)
        self.n = len(self.masks)

    def __len__(self):
        return self.n

    def load_image(self, image_file):
        img = load_img(image_file, color_mode='rgb', target_size=(512, 512))
        img_np = img_to_array(img)
        img_np = img_np.astype(np.uint8)
        return img_np

    def load_mask(self, mask_file):
        f = h5py.File(mask_file, 'r')
        img_np = f['img'][()]
        mask = torch.tensor(img_np).permute(2, 0, 1)
        return mask

    def transform_fn(self, img_np, mask):
        image = array_to_img(img_np, data_format="channels_last")
        image = get_color_distortion()(image)
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        angle = random.randint(0, 90)
        translate = (random.uniform(0, 100), random.uniform(0, 100))
        scale = random.uniform(0.5, 2)
        shear = random.uniform(0, 0)

        image = TF.affine(image, angle, translate, scale, shear)
        mask = TF.affine(mask, angle, translate, scale, shear)

        image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))

        image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))

        image = img_to_array(image, data_format="channels_last")
        return image, mask

    def __getitem__(self, item):
        image_id = self.images[item]
        mask_id = image_id.replace(".jpg", ".h5")
        image_file = f"{self.image_path}/{image_id}"
        mask_file = f"{self.mask_path}/{mask_id}"
        image_original = self.load_image(image_file)
        mask_original = self.load_mask(mask_file)
        image_transformed, mask_transformed = self.transform_fn(image_original, mask_original)
        image_original = (image_original / 255.0).astype('float32')
        image_transformed = (image_transformed / 255.0).astype('float32')
        return image_id, image_original, image_transformed, mask_original, mask_transformed


def make_pretrain_loader(image_path, mask_path, args, shuffle=True):
    data_set = PretrainDataset(image_path, mask_path)
    data_loader = DataLoader(data_set,
                             batch_size=args.batch_size,
                             shuffle=shuffle,
                             num_workers=args.workers,
                             pin_memory=torch.cuda.is_available())
    return data_loader


class SkinDataset(Dataset):
    def __init__(self, train_test_id, image_path, train_test_split_file='./data/train_test_id.pickle',
                 train=True, attribute='all', transform=None, num_classes=None):

        self.train_test_id = train_test_id
        self.image_path = image_path
        self.train = train
        self.attr_types = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
        self.attribute = attribute

        self.transform = transform
        self.num_classes = num_classes

        with open(train_test_split_file, 'rb') as f:
            self.mask_ind = pickle.load(f)

        ## subset the data by mask type
        if self.attribute is not None and self.attribute != 'all':
            ## if no mask, this sample will be filtered out
            # ind = (self.train_test_id[self.mask_attr] == 1)
            # self.train_test_id = self.train_test_id[ind]
            print('mask type: ', self.mask_attr, 'train_test_id.shape: ', self.train_test_id.shape)
        ## subset the data by train test split
        if self.train:
            self.train_test_id = self.train_test_id[self.train_test_id['Split'] == 'train'].ID.values
            print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)
        else:
            self.train_test_id = self.train_test_id[self.train_test_id['Split'] != 'train'].ID.values
            print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)
        self.n = self.train_test_id.shape[0]

    def __len__(self):
        return self.n

    def transform_fn(self, image, mask):
        if self.num_classes == 1:
            ### Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL Image while preserving the value range.
            image = array_to_img(image, data_format="channels_last")
            mask = array_to_img(mask, data_format="channels_last")
            ## Input type float32 is not supported

            ##!!!
            ## the preprocess funcions from Keras are very convenient
            ##!!!

            # Resize
            # resize = transforms.Resize(size=(520, 520))
            # image = resize(image)
            # mask = resize(mask)

            # Random crop
            # i, j, h, w = transforms.RandomCrop.get_params(
            #    image, output_size=(512, 512))
            # image = TF.crop(image, i, j, h, w)
            # mask = TF.crop(mask, i, j, h, w)

            ## https://pytorch.org/docs/stable/torchvision/transforms.html
            ## https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random to_grayscale
            # if random.random() > 0.6:
            #     image = TF.to_grayscale(image, num_output_channels=3)

            angle = random.randint(0, 90)
            translate = (random.uniform(0, 100), random.uniform(0, 100))
            scale = random.uniform(0.5, 2)
            shear = random.uniform(-10, 10)
            image = TF.affine(image, angle, translate, scale, shear)
            mask = TF.affine(mask, angle, translate, scale, shear)

            # Random adjust_brightness
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))

            # Random adjust_saturation
            image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))

            # Random adjust_hue
            # `hue_factor` is the amount of shift in H channel and must be in the
            #     interval `[-0.5, 0.5]`.
            # image = TF.adjust_hue(image, hue_factor=random.uniform(-0.2, 0.2))

            # image = TF.adjust_gamma(image, gamma=random.uniform(0.8, 1.5), gain=1)

            angle = random.randint(0, 90)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            # Transform to tensor
            image = img_to_array(image, data_format="channels_last")
            mask = img_to_array(mask, data_format="channels_last")

        else:
            ### Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL Image while preserving the value range.
            image = array_to_img(image, data_format="channels_last")
            mask_pil_array = [None] * mask.shape[-1]
            for i in range(mask.shape[-1]):
                mask_pil_array[i] = array_to_img(mask[:, :, i, np.newaxis], data_format="channels_last")

            ## https://pytorch.org/docs/stable/torchvision/transforms.html
            ## https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                for i in range(mask.shape[-1]):
                    mask_pil_array[i] = TF.hflip(mask_pil_array[i])

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                for i in range(mask.shape[-1]):
                    mask_pil_array[i] = TF.vflip(mask_pil_array[i])

            # Random to_grayscale
            # if random.random() > 0.6:
            #     image = TF.to_grayscale(image, num_output_channels=3)

            angle = random.randint(0, 90)
            translate = (random.uniform(0, 100), random.uniform(0, 100))
            scale = random.uniform(0.5, 2)
            shear = random.uniform(0, 0)
            image = TF.affine(image, angle, translate, scale, shear)
            for i in range(mask.shape[-1]):
                mask_pil_array[i] = TF.affine(mask_pil_array[i], angle, translate, scale, shear)

            # Random adjust_brightness
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))

            # Random adjust_saturation
            image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))

            # Random adjust_hue
            # `hue_factor` is the amount of shift in H channel and must be in the
            #     interval `[-0.5, 0.5]`.
            # image = TF.adjust_hue(image, hue_factor=random.uniform(-0.2, 0.2))

            # image = TF.adjust_gamma(image, gamma=random.uniform(0.8, 1.5), gain=1)

            # angle = random.randint(0, 90)
            # image = TF.rotate(image, angle)
            # for i in range(mask.shape[-1]):
            #    mask_pil_array[i] = TF.rotate(mask_pil_array[i], angle)

            # Transform to tensor
            image = img_to_array(image, data_format="channels_last")
            for i in range(mask.shape[-1]):
                # img_to_array(mask_pil_array[i], data_format="channels_last"): 512, 512, 1
                mask[:, :, i] = img_to_array(mask_pil_array[i], data_format="channels_last")[:, :, 0].astype('uint8')

        ### img_to_array will scale the image to (0,255)
        ### when use img_to_array, the image and mask will in (0,255)
        image = (image / 255.0).astype('float32')
        mask = (mask / 255.0).astype('uint8')
        # print(11)
        return image, mask

    def __getitem__(self, index):
        img_id = self.train_test_id[index]

        ### load image
        image_file = self.image_path + '%s.h5' % img_id
        img_np = load_image(image_file)
        ### load masks
        mask_np = load_mask(self.image_path, img_id, self.attribute)

        ###
        # print(img_id,img_np.shape,mask_np.shape)

        if self.train:
            img_np, mask_np = self.transform_fn(img_np, mask_np)

        # mean = np.array([0.485, 0.456, 0.406])
        # std  = np.array([0.229, 0.224, 0.225])
        # img_np = (img_np - mean) / std
        img_np = img_np.astype('float32')
        ind = self.mask_ind.loc[index, self.attr_types].values.astype('uint8')
        # ind = np.array(ind)
        # print(ind)
        # print(ind.shape)

        ###########################################
        # img_np = self.transform(img_np)
        # mask_np = self.transform(mask_np)
        ######
        return img_np, mask_np, ind


def load_image(image_file):
    f = h5py.File(image_file, 'r')
    img_np = f['img'][()]
    img_np = (img_np / 255.0).astype('float32')
    return img_np


def load_mask(image_path, img_id, attribute='pigment_network'):
    mask_file = image_path + '%s_attribute_all.h5' % (img_id)
    f = h5py.File(mask_file, 'r')
    mask_np = f['img'][()]
    mask_np = mask_np.astype('uint8')
    return mask_np


def make_loader(train_test_id, image_path, args, train=True, shuffle=True,
                train_test_split_file='./data/train_test_id.pickle', ):
    data_set = SkinDataset(train_test_id=train_test_id,
                           image_path=image_path,
                           train=train,
                           num_classes=args.num_classes,
                           train_test_split_file=train_test_split_file)
    data_loader = DataLoader(data_set,
                             batch_size=args.batch_size,
                             shuffle=shuffle,
                             num_workers=args.workers,
                             pin_memory=torch.cuda.is_available())
    return data_loader
