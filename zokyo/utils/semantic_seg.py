from pathlib import Path
import json
import numpy as np
import cv2
from matplotlib import cm


def generate_semantic_map(img_path, ann_path, out_dir=None,
                          opacity=0.4, gt=True, color_map=None, fill_poly=False, verbose=True):
    """
    Generate semantic segmentation map for the given image and its annotations and
    save it to given output directory if given. If gt is true then ground truth mask is also returned.
    If color map list is not given then Spectral color map from matplotlib is used.
    (Currently accepts coco toyo annotations)
    """

    if opacity > 1 or opacity < 0:
        raise Exception('Opacity factor should be between 0 and 1')

    img_path = Path(img_path)
    ann_path = Path(ann_path)
    if out_dir:
        out_dir = Path(out_dir)

    with open(ann_path, 'r') as anns_json:
        anns = json.load(anns_json)

    h = anns['images'][0]['height']
    w = anns['images'][0]['width']
    sem_seg = np.zeros((h, w, 3))
    if gt:
        gt_seg = np.zeros((h, w))

    n_classes = len(anns['categories']) + 1

    if color_map is None:
        colours = cm.get_cmap('Spectral', n_classes)
        color_map = colours(np.linspace(0, 1, n_classes))
        color_map = color_map[:, :-1]
        color_map = color_map * 255

    color_map = np.array(color_map)
    if len(color_map.shape) == 1:
        color_map = np.expand_dims(color_map, axis=1)

    for seg_ann in anns['annotations']:
        ann_corners = [int(val)
                       for val in seg_ann['segmentation'][2:-2].split(', ')]
        ann_corners = np.reshape(np.array(ann_corners), (-1, 2))
        colour = [int(c)
                  for c in color_map[seg_ann['category_id']]]
        sem_seg = cv2.polylines(sem_seg, [ann_corners], True, colour, 5)
        if fill_poly:
            sem_seg = cv2.fillPoly(sem_seg, [ann_corners], color=tuple(colour))
        if gt:
            gt_seg = cv2.fillPoly(
                gt_seg, [ann_corners], color=seg_ann['category_id'])

    sem_seg = sem_seg.astype('uint8')

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sem_seg = cv2.addWeighted(sem_seg, opacity, img, 1 - opacity, 0)

    sem_seg = cv2.cvtColor(sem_seg, cv2.COLOR_RGB2BGR)

    if out_dir:
        cv2.imwrite(str(out_dir / img_path.stem) + '.jpg', sem_seg)

        if gt:
            np.save(str(out_dir / img_path.stem) + '_gt.npy', gt_seg)

    if verbose and out_dir:
        print(f'Saved at {out_dir}')

    if gt:
        return sem_seg, gt_seg
    return sem_seg
