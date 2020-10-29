# -*- coding: utf-8 -*-

import Augmentor
import os

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/uatg"

def do_augmentation(dir_path=dir_path, no_of_sample=10, cache=True):
    """do sample augmentation"""
    p = Augmentor.Pipeline(dir_path)
    p.rotate90(probability=0.5)
    p.rotate270(probability=0.5)
    p.flip_left_right(probability=0.8)
    p.flip_top_bottom(probability=0.3)
    p.crop_random(probability=1, percentage_area=0.5)
    p.resize(probability=1.0, width=120, height=120)
    p.sample(no_of_sample)
    return len([name for name in os.listdir(dir_path + "/output")])
