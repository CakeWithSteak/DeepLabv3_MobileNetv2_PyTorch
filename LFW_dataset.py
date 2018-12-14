# Adapted from https://github.com/akirasosa/mobile-semantic-segmentation/blob/master/dataset.py

import re
import os
from glob import glob

import cv2
import numpy as np
from torch.utils.data import Dataset


def _mask_to_img(mask_file, img_dir):
    img_file = re.sub('^{}/masks'.format(img_dir), '{}/images'.format(img_dir), mask_file)
    img_file = re.sub('\.ppm$', '.jpg', img_file)
    return img_file


def _img_to_mask(img_file, img_dir):
    mask_file = re.sub('^{}/images'.format(img_dir), '{}/masks'.format(img_dir), img_file)
    mask_file = re.sub('\.jpg$', '.ppm', mask_file)
    return mask_file


def get_img_files(img_dir):
    mask_files = sorted(glob('{}/masks/*.ppm'.format(img_dir)))
    return np.array([_mask_to_img(f, img_dir) for f in mask_files])


class LFWDataset(Dataset):
    def __init__(self, dataset_dir, mode, transforms=None):
        if mode not in ['train', 'test', 'val']:
            raise ValueError('Unsupported mode %s' % mode)

        if not os.path.exists(dataset_dir + "/images") or not os.path.exists(dataset_dir + "/masks"):
            raise ValueError('Dataset doesn\'t exist at %s' % dataset_dir)

        self.img_files = get_img_files(dataset_dir)
        self.mask_files = [_img_to_mask(f, dataset_dir) for f in self.img_files]
        self.transform = transforms

        if mode == "train":
            self.img_files = self.img_files[: int(2 / 3 * len(self.img_files))]
            self.mask_files = self.mask_files[: int(2 / 3 * len(self.mask_files))]
        else:
            self.img_files = self.img_files[int(2 / 3 * len(self.img_files)):]
            self.mask_files = self.mask_files[int(2 / 3 * len(self.mask_files)):]

        assert len(self.img_files) == len(self.mask_files)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_name = os.path.basename(self.img_files[idx])

        mask = cv2.imread(self.mask_files[idx])
        mask = np.argmax(mask, axis=2)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if mask.min() == -1:
            raise ValueError

        mask_name = os.path.basename(self.mask_files[idx])

        sample = {'image': img, 'label': mask,
                  'image_name': image_name, 'label_name': mask_name}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    #pass
    path = input("Dataset test path: ")
    train_dataset = LFWDataset(path, "train")
    print("-------------Train-------------")
    print(train_dataset.img_files)
    print("\n-------------Test-------------")
    test_dataset = LFWDataset(path, "test")
    print(test_dataset.img_files)
    print("\nTesting some samples...")
    sample = test_dataset[3]
    cv2.imshow("Image", sample["image"])
    cv2.imshow("Mask", sample["label"])
    print(sample["image_name"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()