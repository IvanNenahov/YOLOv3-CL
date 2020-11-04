import os
import pandas as pd
import torch
from PIL import Image
import numpy as np
import cv2 as cv
from common.coco_dataset import COCODataset

class CORE50s(torch.utils.data.Dataset):
    def __init__(self, root='G:/projects/core50_350_1f', n_batch=0, bboxes='x1y1x2y2'):
        with open(root + '/train.txt', 'r') as f:
            self.train_paths = f.readlines()

        self.n_batch = n_batch
        # reproduce CORE50 NIC experiment
        if n_batch == 0:
            self.train_paths = self.train_paths[:3000]
        else:
            start = 3000 + (n_batch - 1) * 300
            end = 3000 + n_batch * 300

            self.train_paths = self.train_paths[start: end]

        self.bboxes = bboxes

    def __len__(self):
        return len(self.train_paths)

    def __getitem__(self, item):

        labels_path = self.train_paths[item].replace('images', 'labels').replace('png', 'txt').replace('\n', '')
        image_path = self.train_paths[item].replace("\n", "")
        #image = np.array(Image.open(f'{image_path}'), dtype=int)
        image = cv.imread(f'{image_path}', cv.IMREAD_COLOR)

        #image = np.swapaxes(image, 0, 2)
        #image = np.swapaxes(image, 1, 2)



        labels = np.loadtxt(labels_path)

        if self.bboxes == 'xywh':
            h = labels[4] - labels[3]
            w = labels[2] - labels[1]

            x = (labels[1] + labels[2]) / 2
            y = (labels[3] + labels[4]) / 2
            labels[1:] = x, y, w, h
        else:
            labels[2, 3] = labels[3, 2]
        labels = labels.reshape(-1, 5)

        image = torch.from_numpy(image).type(torch.FloatTensor)
        labels = torch.from_numpy(labels).type(torch.FloatTensor)

        return {'image': image, 'label': labels}


def main():
    dset = CORE50s(root='G:/projects/core50_350_1f', bboxes='xywh')
    # dataloader = torch.utils.data.DataLoader(
    #     dset,
    #     batch_size=24,
    # #    shuffle=True,
    #     #num_workers=opt.n_cpu,
    # #    pin_memory=True,
    #     # collate_fn=dataset.collate_fn,
    # )
    # for i, (x, y) in enumerate(dataloader):
    #     print(x.shape)
    #     break

    dataloader = torch.utils.data.DataLoader(dset, batch_size=16)
    for step, sample in enumerate(dataloader):
        for i, (image, label) in enumerate(zip(sample['image'], sample['label'])):
            image = image.numpy()
            cv.imshow('im2', image)
            h, w = image.shape[:2]

            for l in label:
                if l.sum() == 0:
                    continue
                x1 = int((l[1] - l[3] / 2) * w)
                y1 = int((l[2] - l[4] / 2) * h)
                x2 = int((l[1] + l[3] / 2) * w)
                y2 = int((l[2] + l[4] / 2) * h)
                cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            #cv.imwrite("step{}_{}.jpg".format(step, i), image)
            cv.imshow('im', image)
            cv.waitKey()
        # only one batch
        break
    cv.destroyAllWindows()

    # dataloader = torch.utils.data.DataLoader(COCODataset("G:/projects/core50_350_1f/train.txt",
    #                                                      (350, 350), True, is_debug=True),
    #                                          batch_size=16,
    #                                          shuffle=True, num_workers=1, pin_memory=False)
    # for step, sample in enumerate(dataloader):
    #     for i, (image, label) in enumerate(zip(sample['image'], sample['label'])):
    #         image = image.numpy()
    #         h, w = image.shape[:2]
    #         for l in label:
    #             if l.sum() == 0:
    #                 continue
    #             x1 = int((l[1] - l[3] / 2) * w)
    #             y1 = int((l[2] - l[4] / 2) * h)
    #             x2 = int((l[1] + l[3] / 2) * w)
    #             y2 = int((l[2] + l[4] / 2) * h)
    #
    #             x1 = l[1] * w
    #             x2 = l[2] * w
    #             y1 = l[3] * h
    #             y2 = l[4] * h
    #
    #             cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    #         image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    #         cv.imshow('z', image)
    #         cv.waitKey()
    #     # only one batch
    #     break


if __name__ == '__main__':
    main()

