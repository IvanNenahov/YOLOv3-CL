TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "G:/projects/YOLOv4-CL/YOLOv3_PyTorch/weights/darknet53_weights_pytorch.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 20,
    },
    "lr": {
        "backbone_lr": 0.0001,
        "other_lr": 0.0001,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.1,
        "decay_step": 2,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 4e-05,
    },
    "batch_size": 16,
    "train_path": "G:\projects\core50_350_1f/train.txt",
    "epochs": 5,
    "img_h": 416,
    "img_w": 416,
    "parallels": [0,1,2,3],                         #  config GPU device
    "working_dir": "G:/projects/training_yolov3",              #  replace with your working dir
    "pretrain_snapshot": '',#"G:/projects/YOLOv4-CL/YOLOv3_PyTorch/weights/yolov3_weights_pytorch.pth",                        #  load checkpoint
    "evaluate_type": "", 
    "try": 1,
    "export_onnx": False,
}
