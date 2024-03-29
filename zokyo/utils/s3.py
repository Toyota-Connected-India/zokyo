# -*- coding: utf-8 -*-
# Contributors : [srinivas.v@toyotaconnected.co.in,srivathsan.govindarajan@toyotaconnected.co.in,
# harshavardhan.thirupathi@toyotaconnected.co.in,
# ashok.ramadass@toyotaconnected.com ]

import os


def _find_latest_dir(s3_bucket, dir_path):
    """
    Finds the latest model directory by using the directory name. Since
    saved models use timestamp as the folder name, takes the maximum timestamp
    value as the latest directory
    """
    obj_list = []
    for obj_summary in s3_bucket.objects.filter(Prefix=dir_path):
        # get the version number for each file
        obj_list.append(int(obj_summary.key.lstrip(dir_path).split('/')[0]))
    latest_dir = str(max(obj_list)) + '/'
    return latest_dir


def _maybe_strip_from_path(download_path, strip_from_path=None):
    if strip_from_path:
        save_path = download_path.replace(strip_from_path, '')
    else:
        save_path = download_path
    return save_path


def download_s3_dir(s3_bucket, dir_path, strip_from_path=None):
    """
    Downloads a directory from s3
    """
    for obj_summary in s3_bucket.objects.filter(Prefix=dir_path):
        download_path = obj_summary.key

        save_path = _maybe_strip_from_path(download_path, strip_from_path)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        s3_bucket.download_file(download_path, save_path)
