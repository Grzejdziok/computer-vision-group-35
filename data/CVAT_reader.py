import os
from black import out
import cv2
import shutil
import numpy as np
from lxml import etree
from tqdm import tqdm
import os
from PIL import Image
from typing import List, Tuple
import torchvision
import torch


def dir_create(path:str):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)

def parse_anno_file(cvat_xml:str, image_name:str):
    root = etree.parse(cvat_xml).getroot()
    anno = []
    image_name_attr = ".//image[@name='{}']".format(image_name)
    for image_tag in root.iterfind(image_name_attr):
            image = {}
            for key, value in image_tag.items():
                image[key] = value
            image['shapes'] = []
            for poly_tag in image_tag.iter('polygon'):
                polygon = {'type': 'polygon'}
                for key, value in poly_tag.items():
                    polygon[key] = value
                image['shapes'].append(polygon)
            for box_tag in image_tag.iter('box'):
                box = {'type': 'box'}
                for key, value in box_tag.items():
                    box[key] = value
                box['points'] = "{0},{1};{2},{1};{2},{3};{0},{3}".format(
                    box['xtl'], box['ytl'], box['xbr'], box['ybr'])
                image['shapes'].append(box)
            image['shapes'].sort(key=lambda x: int(x.get('z_order', 0)))
            anno.append(image)
    return anno


def create_mask_file(width:int, height:int, shapes:List[dict], scale_factor):
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    if len(shapes)!=0:
        shape = shapes[-1]
        points = [tuple(map(float, p.split(','))) for p in shape['points'].split(';')]
        points = np.array([(int(p[0]), int(p[1])) for p in points])
        points = points*scale_factor
        points = points.astype(int)
        mask = cv2.fillPoly(mask, [points], color=(255, 255, 255))
    return mask[:, :, 0].astype(np.bool_)
    


def create_masks(masks_dir:str, image_dir:str, cvat_xml:str, scale_factor:float = 1):
    dir_create(masks_dir)
    img_list = []
    for root, dirs, files in os.walk(image_dir, topdown=True):    
        for image in files:
            if ".png" in image:
                parent, folder, name = os.path.join(root, image).split("/")
                img_list.append(os.path.join(folder, name))
                dir_create(os.path.join(masks_dir, folder))
    for img in tqdm(img_list, desc='Writing masks'):
        img_path = os.path.join(image_dir, img)
        anno = parse_anno_file(cvat_xml, img)
        background = []
        is_first_image = True
        for image in anno:
            if is_first_image:
                current_image = cv2.imread(img_path)
                height, width, _ = current_image.shape
                background = np.zeros((height, width, 3), np.uint8)
                is_first_image = False
            output_path = os.path.join(masks_dir, img)
            background = create_mask_file(width,
                                          height,
                                          image['shapes'],
                                          scale_factor)
            background = Image.fromarray(background)
            background.save(output_path)
    
def create_images_1(masks_dir:str, image_dir:str, image_1_dir:str, image_size:Tuple[int,int]):
    dir_create(image_1_dir)
    for root, dirs, files in os.walk(image_dir, topdown=True):
        for box in dirs:
            if "box" in box:
                dir_create(os.path.join(image_1_dir, box))
                image_folder = os.path.join(root, box)
                mask_folder = os.path.join(masks_dir, box)
                #identify box image
                for mask_file in os.listdir(mask_folder):
                    mask = Image.open(os.path.join(mask_folder, mask_file))
                    mask_tensor = torchvision.transforms.ToTensor()(mask)
                    if torch.all((mask_tensor == 0)):
                        box_image = Image.open(os.path.join(image_folder, mask_file))
                        box_tensor = torchvision.transforms.ToTensor()(box_image)
                        break
                
                for mask_file in tqdm(os.listdir(mask_folder), desc=f"Writing single-object images - {box}"):
                    mask = Image.open(os.path.join(mask_folder, mask_file))
                    mask_tensor = torchvision.transforms.ToTensor()(mask)
                    image = Image.open(os.path.join(image_folder, mask_file))
                    image_tensor = torchvision.transforms.ToTensor()(image)
                    image_1 = torchvision.transforms.ToPILImage()(mask_tensor*image_tensor+(1-mask_tensor)*box_tensor)
                    output_path = os.path.join(image_1_dir, box, mask_file)
                    image_1.save(output_path)


