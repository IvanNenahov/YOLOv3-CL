# coding='utf-8'
import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
# sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'evaluate'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.coco_dataset import COCODataset
from data_loader import CORE50s
import torchvision
from torchvision import transforms


def train(config):
    config["global_step"] = config.get("start_step", 0)
    is_training = False if config.get("export_onnx") else True

    # Load and initialize network
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)

    # Optimizer and learning rate
    optimizer = _get_optimizer(config, net)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr"]["decay_step"],
        gamma=config["lr"]["decay_gamma"])

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("Load pretrained weights from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)


    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 416))
    ])

    # DataLoader
    #dataset = torchvision.datasets.VOCDetection(root='G:/projects/pascalVOC/VOCtrainval_11-May-2012',
    #                                            year='2012', transforms=trans)
    dataset = CORE50s(root='G:/projects/core50_350_1f', n_batch=0, bboxes='xywh')
    #dataset = COCODataset('G:/projects/core50_350_1f/train.txt', (416, 416), True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=4,
                                             shuffle=True, num_workers=4, pin_memory=True)

    # Start the training loop
    logging.info("Start training.")
    for epoch in range(config["epochs"]):
        for step, samples in enumerate(dataloader):
            images, labels = samples['image'], samples['label']
            start_time = time.time()
            config["global_step"] += 1

            # Forward and backward
            optimizer.zero_grad()
            outputs = net(images.permute(0, 3, 1, 2))

            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], labels)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]
            loss.backward()
            optimizer.step()

            if step > 0 and step % 10 == 0:
                _loss = loss.item()
                duration = float(time.time() - start_time)
                example_per_second = config["batch_size"] / duration
                lr = optimizer.param_groups[0]['lr']
                logging.info(
                    "epoch [%.3d] iter = %d loss = %.2f example/sec = %.3f lr = %.5f "%
                    (epoch, step, _loss, example_per_second, lr)
                )
                config["tensorboard_writer"].add_scalar("lr",
                                                        lr,
                                                        config["global_step"])
                config["tensorboard_writer"].add_scalar("example/sec",
                                                        example_per_second,
                                                        config["global_step"])
                for i, name in enumerate(losses_name):
                    value = _loss if i == 0 else losses[i]
                    config["tensorboard_writer"].add_scalar(name,
                                                            value,
                                                            config["global_step"])

            if step > 0 and step % 500 == 0:
                # net.train(False)
                _save_checkpoint(net.state_dict(), config)
                # net.train(True)

        _save_checkpoint(net.state_dict(), config)
        lr_scheduler.step()

    # net.train(False)
    _save_checkpoint(net.state_dict(), config)
    # net.train(True)
    logging.info("Bye~")

# best_eval_result = 0.0
def _save_checkpoint(state_dict, config, evaluate_func=None):
    # global best_eval_result
    checkpoint_path = os.path.join(config["sub_working_dir"], "model.pth")
    torch.save(state_dict, checkpoint_path)
    logging.info("Model checkpoint saved to %s" % checkpoint_path)
    # eval_result = evaluate_func(config)
    # if eval_result > best_eval_result:
        # best_eval_result = eval_result
        # logging.info("New best result: {}".format(best_eval_result))
        # best_checkpoint_path = os.path.join(config["sub_working_dir"], 'model_best.pth')
        # shutil.copyfile(checkpoint_path, best_checkpoint_path)
        # logging.info("Best checkpoint saved to {}".format(best_checkpoint_path))
    # else:
        # logging.info("Best result: {}".format(best_eval_result))


def _get_optimizer(config, net):
    optimizer = None

    # Assign different lr for each layer
    params = None
    base_params = list(
        map(id, net.backbone.parameters())
    )
    logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

    if not config["lr"]["freeze_backbone"]:
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
            {"params": net.backbone.parameters(), "lr": config["lr"]["backbone_lr"]},
        ]
    else:
        logging.info("freeze backbone's parameters.")
        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
        ]

    # Initialize optimizer class
    if config["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])
    elif config["optimizer"]["type"] == "amsgrad":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"],
                               amsgrad=True)
    elif config["optimizer"]["type"] == "rmsprop":
        optimizer = optim.RMSprop(params, weight_decay=config["optimizer"]["weight_decay"])
    else:
        # Default to sgd
        logging.info("Using SGD optimizer.")
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["optimizer"]["weight_decay"],
                              nesterov=(config["optimizer"]["type"] == "nesterov"))

    return optimizer

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    params_path = 'params.py'
    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()

    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Create sub_working_dir
    sub_working_dir = '{}/{}/size{}x{}_try{}/{}'.format(
        config['working_dir'], config['model_params']['backbone_name'], 
        config['img_w'], config['img_h'], config['try'],
        time.strftime("%Y%m%d%H%M%S", time.localtime()))
    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    config["sub_working_dir"] = sub_working_dir
    logging.info("sub working dir: %s" % sub_working_dir)

    # Creat tf_summary writer
    config["tensorboard_writer"] = SummaryWriter(sub_working_dir)
    logging.info("Please using 'python -m tensorboard.main --logdir={}'".format(sub_working_dir))

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    train(config)

if __name__ == "__main__":
    main()
