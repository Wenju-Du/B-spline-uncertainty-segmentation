import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # 1. Read image
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        # 2. Read mask
        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                                                img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        # 3. Apply augmentation
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        if mask.ndim == 3 and mask.shape[-1] == self.num_classes:
            mask = mask.permute(2, 0, 1)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # 4. Normalize values
        mask = mask.float() / 255.0

        # Binarize mask. Adjust this threshold if needed.
        mask[mask > 0.01] = 1.0    # adjust here
        mask[mask <= 0.01] = 0.0   # adjust here

        return img, mask, {'img_id': img_id}
