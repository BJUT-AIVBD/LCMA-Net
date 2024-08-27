#!usr/bin/env python
# -*- coding:utf-8 _*-
import torch
import numpy as np
import torchvision.utils
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sim_matrix, labels, img_path = torch.load("/home/yjc/PythonProject/PLIntegration/train_outputs/mma_reid/"
                                              "MMClassification_XY_YZ1/2022-03-04_11-59-58/"
                                              "checkpoints/epoch=239-train_loss=0.35-EER=2.57.visual_results")
    index = np.random.randint(0, len(img_path), 10)
    select_sim = sim_matrix[index, :10]
    selected_imgs = []
    for i in select_sim:
        for j in i:
            img = Image.open(img_path[j])
            selected_imgs.append((ToTensor()(img)))
    imgs = torch.stack(selected_imgs)
    grid_img = make_grid(imgs,nrow=10,padding=10,pad_value=1)
    torchvision.utils.save_image(grid_img, "visual_results.png")
    plt.imshow(np.transpose(grid_img.numpy(),(1,2,0)))
    plt.show()
    print(imgs.shape)

