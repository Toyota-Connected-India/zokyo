import mock

from ...utils.s3 import _find_latest_dir
from ...utils.s3 import _maybe_strip_from_path


def test_find_latest_dir():
    class s3_return:
        key = 'models/latest/1589/'

    s3_bucket = mock.MagicMock()
    s3_bucket.objects.filter = mock.MagicMock(
        return_value=[s3_return])

    latest_dir = _find_latest_dir(s3_bucket, dir_path='models/latest')
    assert latest_dir == '1589/'


def test_maybe_strip_from_path():
    download_path = 'models/latest/1589/vocab'

    save_path1 = _maybe_strip_from_path(download_path, strip_from_path=None)
    save_path2 = _maybe_strip_from_path(download_path, strip_from_path='1589/')
    assert save_path1 == download_path
    assert save_path2 == 'models/latest/vocab'
