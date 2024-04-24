# Copyright (c) owenxing1994@gmail.com
import torch
from torchvision import transforms
import torchvision

import argparse
import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from kornia.color import rgb_to_ycbcr, bgr_to_rgb, rgb_to_bgr, ycbcr_to_rgb
from kornia import image_to_tensor, tensor_to_image
import cv2
import numpy as np
import time

from module.fuse.generator import Generator
from module.fuse.discriminator import Discriminator
from module.detect.models.yolo import Model

class IVIFDataset(Dataset):
    def __init__(self, dataset_folder):
        super().__init__()
        self.ir_folder = os.path.join(dataset_folder, "ir")
        self.ir_weight = os.path.join(dataset_folder, 'iqa','ir')
        self.vis_folder = os.path.join(dataset_folder, 'vi')
        self.vis_weight = os.path.join(dataset_folder, "iqa" , 'vi')
        self.mask_folder = os.path.join(dataset_folder, 'mask')
        self.label_folder = os.path.join(dataset_folder, 'labels')
        assert len(os.listdir(self.ir_folder)) == len(os.listdir(self.vis_folder)), "The number of images in the two folders must be the same."
        self.image_names = os.listdir(self.ir_folder)
        
        self.ir_transform = transforms.Compose([ 
            # transforms.Resize((320, 320)), 
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])
        self.vis_transform = transforms.Compose([   
            # transforms.Resize((320, 320)), 
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        ir_image_path = os.path.join(self.ir_folder, img_name)
        ir_weight_path = os.path.join(self.ir_weight, img_name)
        vis_image_path = os.path.join(self.vis_folder, img_name)
        vis_weight_path = os.path.join(self.vis_weight, img_name)
        img_mask_path = os.path.join(self.mask_folder, img_name)
        label_path = os.path.join(self.label_folder, img_name.split('.')[0]+'.txt')

        # load and transform images
        data_IR = self.ir_transform(Image.open(ir_image_path))
        weight_IR = self.ir_transform(Image.open(ir_weight_path))
        data_VIS = self.vis_transform(Image.open(vis_image_path))
        weight_VIS = self.vis_transform(Image.open(vis_weight_path))

        img_mask = self.ir_transform(Image.open(img_mask_path))

        target = np.loadtxt(label_path, dtype=np.float32)
        target = torch.from_numpy(target).view(-1, 5)
        # add 1 column at the beginning for class label
        target = torch.cat((torch.zeros(target.size(0), 1), target), dim=1)
        # target = yolo_to_xyxy_normalized(target)
        
        # load cbcr for vis image for batch_size = 1
        img_n = cv2.imread(str(vis_image_path), cv2.IMREAD_COLOR)
        # img_n = cv2.resize(img_n, (320, 320), interpolation=cv2.INTER_CUBIC)    
        img_t = image_to_tensor(img_n).float() / 255
        img_t = rgb_to_ycbcr(bgr_to_rgb(img_t))
        vis_image_y, vis_image_cbcr = torch.split(img_t, [1, 2], dim=0)

        return img_name, data_IR, weight_IR, data_VIS, weight_VIS, vis_image_y, vis_image_cbcr, img_mask, target

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.3 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output

def main(args):
    # create output directories
    if not (Path(args.model).parent / ("images_"+ Path(args.model).stem )).exists(): 
        (Path(args.model).parent / ("images_"+ Path(args.model).stem )).mkdir(parents=True)
    if not (Path(args.model).parent / ("labels_"+ Path(args.model).stem )).exists():
        (Path(args.model).parent / ("labels_"+ Path(args.model).stem )).mkdir(parents=True)

    # buil dataset
    dataset_path = args.dataset
    dataset = IVIFDataset(dataset_path)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build fusion model
    generator = Generator(dim=32, depth=3).to(device)
    dis_t = Discriminator(dim=32, size=[224, 224]).to(device)
    dis_d = Discriminator(dim=32, size=[224, 224]).to(device)

    # build detection model
    config_p = "/home/deep/Owen_ssd/code/TarDAL/module/detect/models/yolov5s.yaml"
    net = Model(cfg=config_p, ch=3, nc=args.classes).to(device)
    

    # load pre-trained model
    checkpoint = torch.load(args.model, map_location=torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')
    # generator.load_state_dict(checkpoint["generator"])
    # net.load_state_dict(checkpoint["net"])
    generator.load_state_dict(checkpoint["fuse"])
    net.load_state_dict(checkpoint["detect"])
    

    print("Start inference...")
    with torch.no_grad():
        for i, (img_name, data_IR, weight_IR, data_VIS, weight_VIS, vis_image_y, vis_image_cbcr, img_mask, target)  in enumerate(test_loader):
            data_IR, weight_IR, data_VIS, weight_VIS, vis_image_y, vis_image_cbcr, img_mask, target = data_IR.to(device), weight_IR.to(device), data_VIS.to(device), weight_VIS.to(device), vis_image_y.to(device), vis_image_cbcr.to(device), img_mask.to(device), target.to(device)
            
            generator.eval()
            net.eval()

            # fusion generator
            data_Fuse = generator(data_IR, data_VIS)

            # recolor data_Fuse
            data_Fuse = torch.cat([data_Fuse, vis_image_cbcr], dim=1)
            data_Fuse = ycbcr_to_rgb(data_Fuse)

            # save images
            img_fused =tensor_to_image(data_Fuse.squeeze().cpu())*255
            cv2.imwrite(str(Path(args.model).parent / ("images_"+ Path(args.model).stem ) / img_name[0]), img_fused)

            # detection
            prediction, _ = net(data_Fuse)
            batch_size, _, height, width = data_Fuse.shape
            preds = non_max_suppression(prediction, conf_thres=args.conf_thres, iou_thres=args.iou_thres, multi_label=args.multi_label) 
            
            import pdb; pdb.set_trace()
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer DenseFuse')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--classes', type=int, default=6, help='Number of classes in the dataset')

    parser.add_argument('--conf_thres', type=float, default=0.2, help='Object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--multi_label', action='store_true', help='Use multi-label detection')
    args = parser.parse_args()
    main(args)