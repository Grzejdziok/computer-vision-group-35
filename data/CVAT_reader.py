import cv2
import shutil
import numpy as np
from lxml import etree
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
from PIL import Image
from typing import List, Dict
import torchvision.transforms.functional as TF
import torch


def dir_create(path:str):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)


def find_image_annotation(cvat_xml_root: etree._ElementTree, image_name:str):
    anno = []
    image_name_attr = ".//image[@name='{}']".format(image_name)
    for image_tag in cvat_xml_root.iterfind(image_name_attr):
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
    

def create_masks(masks_dir: str, image_dir: str, cvat_xml: str, scale_factor: float = 1):
    assert os.path.exists(image_dir), f"Requested image directory {image_dir} does not exist!"
    dir_create(masks_dir)
    img_list = []
    for root, dirs, files in os.walk(image_dir, topdown=True):    
        for image in files:
            if ".png" in image:
                folder = os.path.split(root)[-1]
                img_list.append(os.path.join(folder, image))
                dir_create(os.path.join(masks_dir, folder))

    cvat_xml_root = etree.parse(cvat_xml).getroot()
    for img in tqdm(img_list, desc='Writing masks'):
        img_path = os.path.join(image_dir, img).replace(os.path.sep, '/')  # this is to make sure it uses UNIX convention
        anno = find_image_annotation(cvat_xml_root, img_path)
        assert len(anno) == 1, f"Expected exactly one annotation for image path {img_path}, got {len(anno)}"
        annotation = anno[0]
        height = int(annotation.get("height"))
        width = int(annotation.get("width"))
        shapes = annotation['shapes']

        output_path = os.path.join(masks_dir, img)
        background = create_mask_file(width, height, shapes, scale_factor)
        background = Image.fromarray(background)
        background.save(output_path)


def create_images_1(masks_dir: str, image_dir: str, image_1_dir: str):
    assert os.path.exists(image_dir), f"Requested image directory {image_dir} does not exist!"
    dir_create(image_1_dir)
    for root, dirs, files in os.walk(image_dir, topdown=True):
        for box in dirs:
            assert "box" in box
            dir_create(os.path.join(image_1_dir, box))
            image_folder = os.path.join(root, box)
            mask_folder = os.path.join(masks_dir, box)
            #identify box image
            for mask_file in os.listdir(mask_folder):
                mask = Image.open(os.path.join(mask_folder, mask_file))
                mask_tensor = TF.to_tensor(mask)
                if torch.all((mask_tensor == 0)):
                    box_image = Image.open(os.path.join(image_folder, mask_file))
                    box_tensor = TF.to_tensor(box_image)
                    break
                
            for mask_file in tqdm(os.listdir(mask_folder), desc=f"Writing single-object images - {box}"):
                mask = Image.open(os.path.join(mask_folder, mask_file))
                mask_tensor = TF.to_tensor(mask)
                image = Image.open(os.path.join(image_folder, mask_file))
                image_tensor = TF.to_tensor(image)
                image_1 = TF.to_pil_image(mask_tensor*image_tensor+(1-mask_tensor)*box_tensor)
                output_path = os.path.join(image_1_dir, box, mask_file)
                image_1.save(output_path)


def read_mask_for_image(cvat_xml_root: ET, image_filename: str, subset_name: str) -> np.ndarray:
    images = cvat_xml_root.findall(f"image[@name='{image_filename}'][@subset='{subset_name}']")
    assert len(images) == 1
    image = images[0]
    all_polygons = image.findall("polygon")
    assert len(all_polygons) == 0 or len(all_polygons) == 1, f"Cannot parse more than a single polygon, " \
                                                             f"image {image_filename} contains {len(all_polygons)}"

    height = int(image.get("height"))
    width = int(image.get("width"))
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    if len(all_polygons) > 0:
        last_added_mask_polygon = image.findall("polygon")[-1]
        points_str = last_added_mask_polygon.get("points")
        assert points_str is not None
        points = [tuple(map(float, p.split(','))) for p in points_str.split(';')]
        points = np.array([(int(p[0]), int(p[1])) for p in points])
        points = points
        points = points.astype(int)
        mask = cv2.fillPoly(mask, [points], color=(255, 255, 255))
    return mask[:, :, 0].astype(np.bool_)

"""
def parse_cvat_xml_to_image_path_to_item_mask_dict(cvat_xml_root: etree._ElementTree) -> Dict[str, np.ndarray]:
    all_images = cvat_xml_root.findall("image")

    image_path_to_last_added_mask = {}
    for image in all_images:
        name = image.get("name")
        height = int(image.get("height"))
        width = int(image.get("width"))
        assert name is not None and width is not None and height is not None

        last_added_mask = np.zeros((height, width, 3), dtype=np.uint8)

        all_polygons = image.findall("polygon")
        assert len(all_polygons) == 0 or len(all_polygons) == 1, f"Cannot parse more than a single polygon, image {name} contains {len(all_polygons)}"

        if len(all_polygons) > 0:
            last_added_mask_polygon = image.findall("polygon")[-1]
            points_str = last_added_mask_polygon.get("points")
            assert points_str is not None
            points = [tuple(map(float, p.split(','))) for p in points_str.split(';')]
            points = np.array([(int(p[0]), int(p[1])) for p in points])
            points = points
            points = points.astype(int)
            last_added_mask = cv2.fillPoly(last_added_mask, [points], color=(255, 255, 255))

        image_path_to_last_added_mask[name] = last_added_mask[:, :, 0].astype(np.bool_)
    return image_path_to_last_added_mask
"""