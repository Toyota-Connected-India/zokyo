import glob
import os
from pathlib import Path
import time
import json
import argparse
import numpy as np
import cv2
from matplotlib import cm


def generate_semantic_map(img_path, ann_path, out_dir,
                          opacity=0.4, gt=False, cmap='Spectral', verbose=True):
    '''
    Generate semantic segmentation map for the given image and its annotations and save it to given output directory.
    '''

    if opacity > 1 or opacity < 0:
        raise Exception('Transperancy factor should be between 0 and 1')

    img_path = Path(img_path)
    ann_path = Path(ann_path)
    out_dir = Path(out_dir)

    with open(ann_path, 'r') as anns_json:
        anns = json.load(anns_json)
        h = anns['images'][0]['height']
        w = anns['images'][0]['width']
        sem_seg = np.zeros((h, w, 3))
        if gt:
            gt_seg = np.zeros((h, w))

        n_classes = len(anns['categories']) + 1
        colours = cm.get_cmap(cmap, n_classes)
        color_map = colours(np.linspace(0, 1, n_classes))
        color_map = color_map[:, :-1]

        for seg_ann in anns['annotations']:
            ann_corners = [int(val)
                           for val in seg_ann['segmentation'][2:-2].split(', ')]
            ann_corners = np.reshape(np.array(ann_corners), (-1, 2))
            colour = [int(c * 255)
                      for c in color_map[seg_ann['category_id'], :]]
            sem_seg = cv2.fillPoly(sem_seg, [ann_corners], color=tuple(colour))
            if gt:
                gt_seg = cv2.fillPoly(
                    gt_seg, [ann_corners], color=seg_ann['category_id'])

        sem_seg = sem_seg.astype('uint8')

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sem_seg = cv2.addWeighted(sem_seg, opacity, img, 1 - opacity, 0)

        sem_seg = cv2.cvtColor(sem_seg, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(out_dir / img_path.stem) + '.jpg', sem_seg)

        if gt:
            np.save(str(out_dir / img_path.stem) + '_gt.npy', gt_seg)

        if verbose:
            print(f'Saved at {out_dir}')


# def arg_parser():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('-img', '--img_path', type=str,
#                         help='Image path')
#     parser.add_argument('-ann', '--ann_path', type=str,
#                         help='Annotations path')
#     parser.add_argument('-out', '--out_dir', type=str,
#                         help='Directory to store the output')
#     parser.add_argument('-op', '--opacity', type=float, default=0.4,
#                         help='Transperancy factor')
#     parser.add_argument('-gt', '--ground_truth', action='store_true', default=False,
#                         help='Ground truth of semantic segmentation')
#     parser.add_argument('-cm', '--cmap', type=str, default='Spectral',
#                         help='Matplotlib color map')
#     parser.add_argument('-v', '--verbose', action='store_true', default=False,
#                         help='verbose')
#     args = parser.parse_args()

#     return args


# if __name__ == '__main__':
#     args = arg_parser()
#     semantic_map(args.img_path, args.ann_path, args.out_dir, args.opacity,
#                  args.ground_truth, args.cmap, args.verbose)
