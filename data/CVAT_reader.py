import os
import cv2
import argparse
import shutil
import numpy as np
from lxml import etree
from tqdm import tqdm
import os

def dir_create(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)

def parse_anno_file(cvat_xml, image_name):
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


def create_mask_file(width, height, bitness, background, shapes, scale_factor):
    mask = np.full((height, width, bitness // 8), background, dtype=np.uint8)
    for shape in shapes:
        points = [tuple(map(float, p.split(','))) for p in shape['points'].split(';')]
        points = np.array([(int(p[0]), int(p[1])) for p in points])
        points = points*scale_factor
        points = points.astype(int)
        # mask = cv2.drawContours(mask, [points], -1, color=(255, 255, 255), thickness=5)
        mask = cv2.fillPoly(mask, [points], color=(255, 255, 255))
    return mask

def create_masks():
    output_dir = "masks"
    image_dir = "images"
    cvat_xml = "images/annotations.xml"
    scale_factor=1
    dir_create(output_dir)
    img_list = []
    for root, dirs, files in os.walk(image_dir, topdown=True):    
        for image in files:
            if ".png" in image:
                parent, folder, name = os.path.join(root, image).split("/")
                img_list.append(os.path.join(folder, name))
                dir_create(os.path.join(output_dir, folder))
    mask_bitness = 24
    for img in tqdm(img_list, desc='Writing contours:'):
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
            output_path = os.path.join(output_dir, img)
            background = create_mask_file(width,
                                          height,
                                          mask_bitness,
                                          background,
                                          image['shapes'],
                                          scale_factor)
            cv2.imwrite(output_path, background)