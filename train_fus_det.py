# Copyright (c) owenxing1994@gmail.com
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import argparse
import logging
import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from kornia.color import rgb_to_ycbcr, bgr_to_rgb, rgb_to_bgr, ycbcr_to_rgb
from kornia import image_to_tensor, tensor_to_image
from kornia.losses import MS_SSIMLoss, ssim_loss
from kornia.filters import spatial_gradient
import cv2
import numpy as np
import random
import math

from module.fuse.generator import Generator
from module.fuse.discriminator import Discriminator
from module.detect.models.yolo import Model

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        # self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        # self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        
        # store the kernels as buffers, not parameters and don't assign them to devices.
        # buffers are part of the tensors that hold state. They don't require gradients and should be updated by back-propagation.
        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)
    def forward(self,x):
        # Move the kernels to the device of the input
        weightx = self.weightx.to(x.device)
        weighty = self.weighty.to(x.device)
        # sobelx=F.conv2d(x, self.weightx, padding=1)
        # sobely=F.conv2d(x, self.weighty, padding=1)
        sobelx = F.conv2d(x, weightx, padding=1)
        sobely = F.conv2d(x, weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        # change fused image into gray scale
        gray_transform = transforms.Grayscale(num_output_channels=1)
        generate_img = torch.stack([gray_transform(image) for image in generate_img])
        # image_y=image_vis[:,:1,:,:] # [b,1,224,224]
        image_y=torch.stack([gray_transform(image) for image in image_vis])# modify original code to gray scale
        x_in_max=torch.max(image_y,image_ir) # [b,1,224,224]
        loss_in=F.l1_loss(x_in_max,generate_img) # scalar
        y_grad=self.sobelconv(image_y) # [b,1,224,224]
        ir_grad=self.sobelconv(image_ir) # [b,1,224,224]
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

def yolo_to_xyxy_normalized(yolo_labels):
    """
    Convert a batch of YOLO labels to normalized (x0, y0, x1, y1) format.

    Parameters:
    - yolo_labels: A numpy array of shape [n, 5] containing YOLO labels

    Returns:
    - A numpy array of shape [n, 4] containing converted and normalized labels
    """
    # Extract class_ids and coordinates
    x_centers = yolo_labels[:, 1]
    y_centers = yolo_labels[:, 2]
    widths = yolo_labels[:, 3]
    heights = yolo_labels[:, 4]

    # Compute (x0, y0, x1, y1), still normalized between 0 and 1
    x0 = x_centers - (widths / 2)
    y0 = y_centers - (heights / 2)
    x1 = x_centers + (widths / 2)
    y1 = y_centers + (heights / 2)

    # Stack to form [n, 4] tensor and ensure bounding box coordinates stay within [0, 1]
    xyxy_labels = torch.stack([x0, y0, x1, y1], dim=1)
    xyxy_labels = torch.clamp(xyxy_labels, min=0, max=1)

    return xyxy_labels

def collate_fn(batch):
    img_names, data_IRs, weight_IRs, data_VISs, weight_VISs, vis_image_ys, vis_image_cbcrs, img_masks, targets = zip(*batch)
    
    # Convert lists of tensors to a single tensor per item
    data_IRs = torch.stack(data_IRs, dim=0)
    weight_IRs = torch.stack(weight_IRs, dim=0)
    data_VISs = torch.stack(data_VISs, dim=0)
    weight_VISs = torch.stack(weight_VISs, dim=0)
    vis_image_ys = torch.stack(vis_image_ys, dim=0)
    vis_image_cbcrs = torch.stack(vis_image_cbcrs, dim=0)
    img_masks = torch.stack(img_masks, dim=0)
    
    # Concatenate all targets
    targets = torch.cat(targets, dim=0)
    
    return img_names, data_IRs, weight_IRs, data_VISs, weight_VISs, vis_image_ys, vis_image_cbcrs, img_masks, targets

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
            transforms.Resize((320, 320)), 
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])
        self.vis_transform = transforms.Compose([   
            transforms.Resize((320, 320)), 
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
        img_n = cv2.resize(img_n, (320, 320), interpolation=cv2.INTER_CUBIC)    
        img_t = image_to_tensor(img_n).float() / 255
        img_t = rgb_to_ycbcr(bgr_to_rgb(img_t))
        vis_image_y, vis_image_cbcr = torch.split(img_t, [1, 2], dim=0)

        return img_name, data_IR, weight_IR, data_VIS, weight_VIS, vis_image_y, vis_image_cbcr, img_mask, target
    
def src_loss(x, y):
    return 0.01 * ssim_loss(x, y, window_size=11) + 0.99 * torch.nn.functional.l1_loss(x, y)

def adv_loss(data_Fuse, img_mask, tar_w, det_w, dis_t, dis_d):
    # tar_w, det_w = args.t_weight, args.d_weight
    dis_t.eval()
    
    tar_l = -dis_t(data_Fuse * img_mask)
    

    dis_d.eval()
    img_mask = img_mask if args.d_mask else 0
    det_l = -dis_d(get_gradient(data_Fuse) * (1 - img_mask))
    return tar_w * tar_l + det_w * det_l, tar_l.mean().item(), det_l.mean().item()

def get_gradient(x, eps=1e-8):
    s = spatial_gradient(x, 'sobel')
    dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
    u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2) + eps)  # sqrt backwork x range: (0, n]
    return u

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        # h = model.hyp  # hyperparameters
        # manually override hyperparameters
        h  ={'box': 0.05, 'cls': 0.0225, 'cls_pw': 1.0, 'obj': 0.175, 'obj_pw': 1.0, 'iou_t': 0.2, 'anchor_t': 4.0, 'fl_gamma': 0.0, 'label_smoothing': False}

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

def div_loss(disc, real_x, fake_x, wp: int = 6, eps: float = 1e-6):
    alpha = torch.rand((real_x.shape[0], 1, 1, 1)).cuda()
    tmp_x = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)
    tmp_y = disc(tmp_x)
    grad = autograd.grad(
        outputs=tmp_y,
        inputs=tmp_x,
        grad_outputs=torch.ones_like(tmp_y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(tmp_x.shape[0], -1) + eps
    div = (grad.norm(2, dim=1) ** wp).mean()
    return div

def main(args):
     # create logger
    if not os.path.exists("./log/"+args.name):
        os.makedirs("./log/"+args.name)
    else:
        # stop the program if the log folder already exists
        raise Exception(f"Log folder {args.name} already exists. Please use another name.")
    log_filename = f'./log/{args.name}/training_log.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    logging.info(f"Create logger at {log_filename}")
    logging.info(f"Arguments: {args}")

    # fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Fix the seed {seed}")

    # buil dataset
    dataset_path = args.dataset
    dataset = IVIFDataset(dataset_path)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    logging.info(f"Load dataset from {dataset_path} with {len(dataset)} images")

    # device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # build fusion model
    generator = Generator(dim=32, depth=3).to(device)
    dis_t = Discriminator(dim=32, size=[320, 320]).to(device)
    dis_d = Discriminator(dim=32, size=[320, 320]).to(device)
    logging.info(f"Build fusion model with generator, dis_t, dis_d")

    # build detection model
    config_p = "/home/deep/Owen_ssd/code/TarDAL/module/detect/models/yolov5s.yaml"
    net = Model(cfg=config_p, ch=3, nc=args.classes).to(device)
    logging.info(f"Build detection model with config file {config_p}")

    # model freeze

    # fusion criterion
    criteria_fusion = Fusionloss()
    logging.info(f"Build fusion criterion")
    
    # detection criterion
    det_loss = ComputeLoss(net)
    logging.info(f"Build detection criterion")

    # optimizer and lr scheduler
    optim_G = torch.optim.SGD(generator.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optim_D_t = torch.optim.SGD(dis_t.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optim_D_d = torch.optim.SGD(dis_d.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optim_det = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    sche_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=args.step_size, gamma=args.gamma)
    sche_D_t = torch.optim.lr_scheduler.StepLR(optim_D_t, step_size=args.step_size, gamma=args.gamma)
    sche_D_d = torch.optim.lr_scheduler.StepLR(optim_D_d, step_size=args.step_size, gamma=args.gamma)
    sche_det = torch.optim.lr_scheduler.StepLR(optim_det, step_size=args.step_size, gamma=args.gamma)
    logging.info(f"Build optimizer and lr scheduler")

    logging.info("Start training ...")
    for epoch in range(1, args.epochs + 1):
        acc_generator, acc_discriminator, acc_detection, acc_all = 0, 0, 0, 0
        for i, (img_name, data_IR, weight_IR, data_VIS, weight_VIS, vis_image_y, vis_image_cbcr, img_mask, target)  in enumerate(train_loader):
            data_IR, weight_IR, data_VIS, weight_VIS, vis_image_y, vis_image_cbcr, img_mask, target = data_IR.to(device), weight_IR.to(device), data_VIS.to(device), weight_VIS.to(device), vis_image_y.to(device), vis_image_cbcr.to(device), img_mask.to(device), target.to(device)

            generator.train()
            dis_t.train()
            dis_d.train()
            net.train()

            generator.zero_grad()
            dis_t.zero_grad()
            dis_d.zero_grad()
            net.zero_grad()

            optim_G.zero_grad()
            optim_D_t.zero_grad()
            optim_D_d.zero_grad()
            optim_det.zero_grad()

            # fusion generator
            data_Fuse = generator(data_IR, data_VIS)
            
            # fusion generator loss
            src_l = weight_IR * src_loss(data_Fuse, data_IR) + weight_VIS * src_loss(data_Fuse, data_VIS)
            adv_l, tar_l, det_l = adv_loss(data_Fuse, img_mask, args.t_weight, args.d_weight, dis_t, dis_d)
            g_loss = args.source_weight * src_l.mean() + args.adv_weight * adv_l.mean()
            acc_generator += g_loss.item()

            # if epoch < warmup_epoch: detatch the data_Fuse
            if epoch < args.warmup_epoch:
                data_Fuse = data_Fuse.detach()

            # recolor data_Fuse
            data_Fuse = torch.cat([data_Fuse, vis_image_cbcr], dim=1)
            data_Fuse = ycbcr_to_rgb(data_Fuse)

            # detection
            prediction = net(data_Fuse)

            # detection loss for each img in the batch
            detection_loss, [box_l, obj_l, cls_l]  = det_loss(prediction, target)
            acc_detection += detection_loss.item()

            # generator and detection loss
            fd_loss = g_loss * args.fw + detection_loss * args.dw

            # backward
            fd_loss.backward()
            optim_G.step()
            optim_det.step()

            # fusion discriminator 
            # dis_t
            real_s = data_IR * img_mask
            generator.eval()
            fake_s = generator(data_IR, data_VIS) * img_mask
            fake_s.detach_()
            # judge value towards real & fake
            real_v = torch.squeeze(dis_t(real_s))
            fake_v = torch.squeeze(dis_t(fake_s))
            real_l, fake_l = -real_v.mean(), fake_v.mean()
            div = div_loss(dis_t, real_s, fake_s, 6)
            loss_dis_t = real_l + fake_l + 2 * div
            loss_dis_t.backward()
            optim_D_t.step()
            acc_discriminator += loss_dis_t.item()

            # dis_d
            # sample real & fake
            img_mask = img_mask if args.d_mask else 0  # use mask or not
            real_s = get_gradient(data_VIS) * (1 - img_mask)
            generator.eval()
            fake_s = get_gradient(generator(data_IR, data_VIS)) * (1 - img_mask)
            fake_s.detach_()
            # judge value towards real & fake
            real_v = torch.squeeze(dis_d(real_s))
            fake_v = torch.squeeze(dis_d(fake_s))
            real_l, fake_l = -real_v.mean(), fake_v.mean()
            div = div_loss(dis_d, real_s, fake_s, 6)
            loss_dis_d = real_l + fake_l + 2 * div
            loss_dis_d.backward()
            optim_D_d.step()
            acc_discriminator += loss_dis_d.item()

        acc_generator /= len(train_loader)
        acc_discriminator /= len(train_loader)
        acc_detection /= len(train_loader)
        acc_all = acc_generator + acc_discriminator + acc_detection

        logging.info(f"EPOCH: {epoch}, Generator Loss: {acc_generator:2f}, Discriminator Loss: {acc_discriminator:2f}, Detection Loss: {acc_detection:2f}, All Loss: {acc_all:2f}")
            
        if epoch % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'generator': generator.state_dict(),
                'dis_t': dis_t.state_dict(),
                'dis_d': dis_d.state_dict(),
                'net': net.state_dict(),
                'optim_G': optim_G.state_dict(),
                'optim_D_t': optim_D_t.state_dict(),
                'optim_D_d': optim_D_d.state_dict(),
                'optim_det': optim_det.state_dict(),
            }
            torch.save(checkpoint, f"./log/{args.name}/checkpoint_{epoch}.pth")
            logging.info(f"Saving checkpoint at epoch {epoch}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer DenseFuse')
    parser.add_argument('--name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--classes", type=int, default=6, help="number of classes")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.937, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--step_size", type=int, default=20, help="step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="gamma")
    parser.add_argument("--source_weight", type=float, default=1, help="source weight")
    parser.add_argument("--adv_weight", type=float, default=0, help="adv weight")
    parser.add_argument("--t_weight", type=float, default=1, help="target loss weight")
    parser.add_argument("--d_weight", type=float, default=1, help="detail loss weight")
    parser.add_argument("--d_mask", default=False, action="store_true", help="use mask for detail loss")
    parser.add_argument("--warmup_epoch", type=int, default=2, help="warmup epoch")
    parser.add_argument("--fw", type=str, default=0.5, help="fusion loss weight")
    parser.add_argument("--dw", type=str, default=0.5, help="detection loss weight")
    parser.add_argument("--save_freq", type=int, default=10, help="save frequency")
    args = parser.parse_args()
    main(args)