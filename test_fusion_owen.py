# Copyright (c) owenxing1994@gmail.com
import torch
from torchvision import transforms

import argparse
import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from kornia.color import rgb_to_ycbcr, bgr_to_rgb, rgb_to_bgr, ycbcr_to_rgb
from kornia import image_to_tensor, tensor_to_image
import cv2

from module.fuse.generator import Generator
from module.fuse.discriminator import Discriminator

class IVIFDataset(Dataset):
    def __init__(self, dataset_folder):
        super().__init__()
        self.ir_folder = os.path.join(dataset_folder, 'ir')
        self.vis_folder = os.path.join(dataset_folder, 'vi')
        assert len(os.listdir(self.ir_folder)) == len(os.listdir(self.vis_folder)), "The number of images in the two folders must be the same."
        self.image_names = os.listdir(self.ir_folder)
        
        self.ir_transform = transforms.Compose([ 
            # transforms.Resize((224, 224)), 
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])
        self.vis_transform = transforms.Compose([   
            # transforms.Resize((224, 224)), 
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        ir_image_path = os.path.join(self.ir_folder, img_name)
        vis_image_path = os.path.join(self.vis_folder, img_name)

        # load and transform images
        ir_image = self.ir_transform(Image.open(ir_image_path))
        vis_image = self.vis_transform(Image.open(vis_image_path))
        
        # load cbcr for vis image for batch_size = 1
        img_n = cv2.imread(str(vis_image_path), cv2.IMREAD_COLOR)
        img_t = image_to_tensor(img_n).float() / 255
        img_t = rgb_to_ycbcr(bgr_to_rgb(img_t))
        vis_image_y, vis_image_cbcr = torch.split(img_t, [1, 2], dim=0)

        return img_name, ir_image, vis_image, vis_image_y, vis_image_cbcr 

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

    # build model
    generator = Generator(dim=32, depth=3).to(device)
    dis_t = Discriminator(dim=32, size=[224, 224]).to(device)
    dis_d = Discriminator(dim=32, size=[224, 224]).to(device)
    # load pre-trained model
    checkpoint = torch.load(args.model, map_location=torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')
    # use generator only in inference
    generator.load_state_dict(checkpoint)

    print("Start inference...")
    with torch.no_grad():
        for i, (img_name, data_IR, data_VIS, vis_image_y, vis_image_cbcr)  in enumerate(test_loader):
            data_IR, data_VIS, vis_image_y, vis_image_cbcr = data_IR.to(device), data_VIS.to(device), vis_image_y.to(device), vis_image_cbcr.to(device)
            
            generator.eval()

            # inference
            fusion = generator(data_IR, data_VIS)
            # recolor
            fusion = torch.cat([fusion, vis_image_cbcr], dim=1)
            fusion = ycbcr_to_rgb(fusion)

            # save images
            fusion =tensor_to_image(fusion.squeeze().cpu())*255
            cv2.imwrite(str(Path(args.model).parent / ("images_"+ Path(args.model).stem ) / img_name[0]), fusion)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer DenseFuse')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    args = parser.parse_args()
    main(args)