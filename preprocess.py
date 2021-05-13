import os

import cv2
from keras_preprocessing.image import load_img, img_to_array


def main():
    path = "e:/diploma/ham10000"
    save = "e:/diploma/ham10000_resized"
    images = os.listdir(path)
    for ind, i in enumerate(images):
        print(f"{ind} / {len(images)}")
        image_path = os.path.join(path, i)
        img = load_img(image_path, target_size = (512, 512))
        img_np = img_to_array(img)
        save_image_path = os.path.join(save, i)
        cv2.imwrite(save_image_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()