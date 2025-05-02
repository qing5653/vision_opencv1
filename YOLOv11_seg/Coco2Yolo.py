import os
import json
from tqdm import tqdm
from pycocotools.coco import COCO

def convert(coco_json, output_dir, img_dir):
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    coco = COCO(coco_json)
    
    for img_id in tqdm(coco.imgs):
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        with open(f"{output_dir}/labels/{img_info['file_name'].replace('.jpg', '.txt')}", "w") as f:
            for ann in anns:
                # 转换为 YOLO 格式：class_id x_center y_center width height
                x, y, w, h = ann['bbox']
                x_center = (x + w/2) / img_info['width']
                y_center = (y + h/2) / img_info['height']
                w /= img_info['width']
                h /= img_info['height']
                f.write(f"{ann['category_id']} {x_center} {y_center} {w} {h}\n")

if __name__ == "__main__":
    convert(
        #coco_json="basket_seg/train/annotations/coco_instance_segmentation.json",
        coco_json="basket_seg/val/annotations/coco_instance_segmentation.json",
        #output_dir="basket_yolo/train",
        output_dir="basket_yolo/val",
        #img_dir="basket_seg/train/image"
        img_dir="basket_seg/val/image"
    )