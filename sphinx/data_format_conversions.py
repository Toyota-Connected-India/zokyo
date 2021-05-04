import sys
import os
from pathlib import Path
import json
import xml.etree.ElementTree as ET
 
def coco_std_2_voc(coco_ann_path, database=None, save_folder=None):
    coco_ann_path = Path(coco_ann_path)
    
    if not database:
        database = 'Unspecified'

    if not save_folder:
        save_folder = coco_ann_path.parent
    else:
        save_folder = Path(save_folder)
    
    if not save_folder.is_dir():
        raise Exception('Save path should be a directory')

    with open(coco_ann_path, 'r') as f:
        coco_ann = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in coco_ann['categories']}
    categories = dict(sorted(categories.items()))
    
    voc_names = ''
    for i, cat in categories.items():
        voc_names += cat + '\n'
    with open(f'{save_folder}/voc.names', 'w') as f:
        f.write(voc_names)

    for i, img in enumerate(coco_ann['images']):
        root = ET.Element('annotation')
        
        img_path = Path(img['file_name'])

        fold = ET.Element('folder')
        fold.text = str(img_path.parent)
        root.append(fold)

        file_name = ET.Element('filename')
        file_name.text = img_path.name
        root.append(file_name)

        path = ET.Element('path')
        path.text = str(img_path)
        root.append(path)

        source = ET.Element('source')
        db = ET.SubElement(source, 'database')
        db.text = database
        root.append(source)

        size = ET.Element('size')
        width = ET.SubElement(size, 'width')
        width.text = str(img['width'])
        height = ET.SubElement(size, 'height')
        height.text = str(img['height'])
        depth = ET.SubElement(size, 'depth')
        depth.text = str(3)
        root.append(size)

        for ann in coco_ann['annotations']:
            if img['id'] == ann['image_id']:
                obj = ET.Element('object')

                name = ET.Element('name')
                name.text = categories[ann['category_id']]
                obj.append(name)

                pose = ET.Element('pose')
                if 'pose' in ann:
                    pose.text = str(ann['pose'])
                else:
                    pose.text = 'Unspecified'
                obj.append(pose)

                truncated = ET.Element('truncated')
                if 'truncated' in ann:
                    truncated.text = str(ann['truncated'])
                else:
                    truncated.text = 'Unspecified'
                obj.append(truncated)

                bndbox = ET.Element('bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(ann['bbox'][0])
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(ann['bbox'][1])
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(ann['bbox'][0] + ann['bbox'][2])
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(ann['bbox'][1] + ann['bbox'][3])
                obj.append(bndbox)

                root.append(obj)

        tree = ET.ElementTree(root)

        with open(f'{save_folder}/{i+1}.xml', 'wb') as f:
            tree.write(f)


def voc_2_coco_std(voc_folder_path, info=None, save_folder=None):
    voc_folder_path = Path(voc_folder_path)
    
    if not info:
        info = {}

    if not save_folder:
        save_folder = voc_folder_path
    else:
        save_folder = Path(save_folder)

    if not save_folder.is_dir():
        raise Exception('Save path should be a directory')

    coco_ann = {}

    voc_names_path = list(voc_folder_path.glob('*.names'))[0]
    with open(voc_names_path, 'r') as f:
        voc_names = [n.rstrip('\n') for n in f.readlines()]
    categories = [{"id": i, "name": cat} for i, cat in enumerate(voc_names, 1)]

    coco_ann['info'] = info
    coco_ann['categories'] = categories

    voc_ann_paths = list(voc_folder_path.glob('*.xml'))

    if len(voc_ann_paths) == 0:
        raise Exception('Empty Voc directory')
    
    imgs = []
    anns = []
    j = 0
    for i, voc_ann_path in enumerate(voc_ann_paths):
        root = ET.parse(voc_ann_path).getroot()

        img = {}
        img['id'] = i
        size = root.find('size')
        img['width'] = int(size.find('width').text)
        img['height'] = int(size.find('height').text)
        img['filename'] = root.find('path').text

        imgs.append(img)

        for obj in root.findall('object'):
            ann = {}
            ann['id'] = j
            ann['image_id'] = i
            ann['category_id'] = voc_names.index(obj.find('name').text)
            bbox = obj.find('bndbox')
            x = float(bbox.find('xmin').text)
            y = float(bbox.find('ymin').text)
            w = float(bbox.find('xmax').text) - x
            h = float(bbox.find('ymax').text) - y
            ann['area'] = w * h
            ann['bbox'] = [x, y, w, h]
            j += 1
            anns.append(ann)

    coco_ann['images'] = imgs
    coco_ann['annotations'] = anns

    with open(f'{save_folder}/coco_ann.json', 'w') as f:
        json.dump(coco_ann, f)


def coco_std_2_coco_toyo(coco_ann_path, save_folder=None):
    coco_ann_path = Path(coco_ann_path)
    
    if not save_folder:
        save_folder = coco_ann_path.parent
    else:
        save_folder = Path(save_folder)
    
    if not save_folder.is_dir():
        raise Exception('Save path should be a directory')
    
    with open(coco_ann_path, 'r') as f:
        coco_ann = json.load(f)
    
    coco_toyo_ann = {}
    coco_toyo_ann['info'] = coco_ann['info']
    coco_toyo_ann['categories'] = coco_ann['categories']
    
    for i, img in enumerate(coco_ann['images']):
        coco_toyo_ann['images'] = [img]
        
        anns = []
        for j, ann in enumerate(coco_ann['annotations']):
            if img['id'] == ann['image_id']:
                anns.append(ann)
        
        coco_toyo_ann['annotations'] = anns

        with open(f'{save_folder}/{i}.json', 'w') as f:
            json.dump(coco_toyo_ann, f)


def coco_toyo_2_coco_std(coco_toyo_ann_folder, save_folder=None):
    coco_toyo_ann_folder = Path(coco_toyo_ann_folder)
    
    if not save_folder:
        save_folder = coco_toyo_ann_folder.parent
    else:
        save_folder = Path(save_folder)
    
    if not save_folder.is_dir():
        raise Exception('Save path should be a directory')
    
    coco_toyo_ann_paths = sorted(list(coco_toyo_ann_folder.glob('*.json')))
    
    coco_ann = {}
    imgs = []
    anns = []
    
    for _, coco_toyo_ann_path in enumerate(coco_toyo_ann_paths):
        with open(coco_toyo_ann_path, 'r') as f:
            coco_toyo_ann = json.load(f)

        coco_ann['info'] = coco_toyo_ann['info']
        coco_ann['categories'] = coco_toyo_ann['categories']
        imgs.append(coco_toyo_ann['images'][0])
        anns.extend(coco_toyo_ann['annotations'])

    coco_ann['images'] = sorted(imgs, key=lambda x: x['id'])
    coco_ann['annotations'] = sorted(anns, key=lambda x: x['id'])
    
    with open(f'{save_folder}/coco_ann.json', 'w') as f:
        json.dump(coco_ann, f)


def coco_toyo_2_voc(coco_toyo_ann_folder, database=None, save_folder=None):
    coco_toyo_ann_folder = Path(coco_toyo_ann_folder)
    
    if not database:
        database = 'Unspecified'

    if not save_folder:
        save_folder = coco_toyo_ann_folder.parent
    else:
        save_folder = Path(save_folder)
    
    if not save_folder.is_dir():
        raise Exception('Save path should be a directory')

    coco_toyo_ann_paths = sorted(list(coco_toyo_ann_folder.glob('*.json')))
    
    with open(coco_toyo_ann_paths[0], 'r') as f:
        coco_toyo_ann = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in coco_toyo_ann['categories']}
    categories = dict(sorted(categories.items()))
    
    voc_names = ''
    for i, cat in categories.items():
        voc_names += cat + '\n'
    with open(f'{save_folder}/voc.names', 'w') as f:
        f.write(voc_names)

    for i, coco_toyo_ann_path in enumerate(coco_toyo_ann_paths):

        with open(coco_toyo_ann_path, 'r') as f:
            coco_toyo_ann = json.load(f)
        img = coco_toyo_ann['images'][0]

        root = ET.Element('annotation')
        
        img_path = Path(img['file_name'])

        fold = ET.Element('folder')
        fold.text = str(img_path.parent)
        root.append(fold)

        file_name = ET.Element('filename')
        file_name.text = img_path.name
        root.append(file_name)

        path = ET.Element('path')
        path.text = str(img_path)
        root.append(path)

        source = ET.Element('source')
        db = ET.SubElement(source, 'database')
        db.text = database
        root.append(source)

        size = ET.Element('size')
        width = ET.SubElement(size, 'width')
        width.text = str(img['width'])
        height = ET.SubElement(size, 'height')
        height.text = str(img['height'])
        depth = ET.SubElement(size, 'depth')
        depth.text = str(3)
        root.append(size)

        for ann in coco_toyo_ann['annotations']:
            if img['id'] == ann['image_id']:
                obj = ET.Element('object')

                name = ET.Element('name')
                name.text = categories[ann['category_id']]
                obj.append(name)

                pose = ET.Element('pose')
                if 'pose' in ann:
                    pose.text = str(ann['pose'])
                else:
                    pose.text = 'Unspecified'
                obj.append(pose)

                truncated = ET.Element('truncated')
                if 'truncated' in ann:
                    truncated.text = str(ann['truncated'])
                else:
                    truncated.text = 'Unspecified'
                obj.append(truncated)

                bndbox = ET.Element('bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(ann['bbox'][0])
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(ann['bbox'][1])
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(ann['bbox'][0] + ann['bbox'][2])
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(ann['bbox'][1] + ann['bbox'][3])
                obj.append(bndbox)

                root.append(obj)

        tree = ET.ElementTree(root)

        with open(f'{save_folder}/{i+1}.xml', 'wb') as f:
            tree.write(f)


def voc_2_coco_toyo(voc_folder_path, info=None, save_folder=None):
    voc_folder_path = Path(voc_folder_path)
    
    if not info:
        info = {}

    if not save_folder:
        save_folder = voc_folder_path
    else:
        save_folder = Path(save_folder)

    if not save_folder.is_dir():
        raise Exception('Save path should be a directory')

    coco_toyo_ann = {}

    voc_names_path = list(voc_folder_path.glob('*.names'))[0]
    with open(voc_names_path, 'r') as f:
        voc_names = [n.rstrip('\n') for n in f.readlines()]
    categories = [{"id": i, "name": cat} for i, cat in enumerate(voc_names, 1)]

    coco_toyo_ann['info'] = info
    coco_toyo_ann['categories'] = categories

    voc_ann_paths = list(voc_folder_path.glob('*.xml'))

    if len(voc_ann_paths) == 0:
        raise Exception('Empty Voc directory')
    
    imgs = []
    anns = []
    j = 0
    for i, voc_ann_path in enumerate(voc_ann_paths):
        root = ET.parse(voc_ann_path).getroot()

        img = {}
        img['id'] = i
        size = root.find('size')
        img['width'] = int(size.find('width').text)
        img['height'] = int(size.find('height').text)
        img['filename'] = root.find('path').text

        imgs.append(img)

        for obj in root.findall('object'):
            ann = {}
            ann['id'] = j
            ann['image_id'] = i
            ann['category_id'] = voc_names.index(obj.find('name').text)
            bbox = obj.find('bndbox')
            x = float(bbox.find('xmin').text)
            y = float(bbox.find('ymin').text)
            w = float(bbox.find('xmax').text) - x
            h = float(bbox.find('ymax').text) - y
            ann['area'] = w * h
            ann['bbox'] = [x, y, w, h]
            j += 1
            anns.append(ann)

        coco_toyo_ann['images'] = imgs
        coco_toyo_ann['annotations'] = anns
        imgs = []
        anns = []

        with open(f'{save_folder}/{i}.json', 'w') as f:
            json.dump(coco_toyo_ann, f)


if __name__ == '__main__':
    #coco_std_2_voc('../uatg_sample_gt_coco.json', save_folder='../voc')
    #voc_2_coco_std('../voc')
    #coco_std_2_coco_toyo('../uatg_sample_gt_coco.json', save_folder='../coco_toyo')
    #coco_toyo_2_coco_std('../coco_toyo')
    #coco_toyo_2_voc('../coco_toyo', save_folder='../voc')
    voc_2_coco_toyo('../voc', save_folder='../coco_toyo')
