import pytest
import shutil
from pathlib import Path
from sphinx.utils.CustomExceptions import CoefficientNotinRangeError, OperationNotFoundOrImplemented, \
    CrucialValueNotFoundError
from sphinx.augmentation import Builder

test_configs = [
    "tests/coeff_not_in_range_exception_bright_test_config.json",
    "tests/coeff_not_in_range_exception_dark_test_config.json",
    "tests/operation_not_found_exception_test_config.json",
    "tests/crucial_value_not_found_exception_operations_test_config.json",
    "tests/crucial_value_not_found_exception_input_dir_test_config.json",
    "tests/crucial_value_not_found_exception_ann_format_test_config.json",
    "tests/crucial_value_not_found_exception_dark_test_config.json",
    "tests/crucial_value_not_found_exception_bright_test_config.json",
    "tests/crucial_value_not_found_exception_rand_bright_test_config.json"
]

exception_classes = [
    CoefficientNotinRangeError,
    CoefficientNotinRangeError,
    OperationNotFoundOrImplemented,
    CrucialValueNotFoundError,
    CrucialValueNotFoundError,
    CrucialValueNotFoundError,
    CrucialValueNotFoundError,
    CrucialValueNotFoundError,
    CrucialValueNotFoundError
]

err_msgs = [
    '''\"BrightnessCoefficient\" coefficient of value 2 is not in range 0 and 1''',
    '''\"darkness\" coefficient of value -2 is not in range 0 and 1''',
    '"XYZ" not found or implemented in the module "sphinx.augmentation"',
    "\"operations\" value not found for the \"augmentation configurations\" mentioned",
    "\"input_dir\" value not found for the \"configuration json file\" mentioned",
    "\"annotation_format\" value not found for the \"annotation data\" mentioned",
    "\"darkness\" value not found for the \"DarkenScene\" mentioned",
    "\"brightness\" value not found for the \"BrightenScene\" mentioned",
    "\"distribution\" value not found for the \"RandomBrightness\" mentioned"
]


@pytest.mark.parametrize(
    "test_config,exception_class,err_msg",
    [pytest.param(conf, cl, err_msg) for conf, cl, err_msg in zip(
        test_configs, exception_classes, err_msgs)],
    ids=[Path(conf).stem for conf in test_configs]
)
class TestException:

    @pytest.fixture(autouse=True)
    def builder_teardown(self, request):
        # request.cls.builder = Builder("tests/builder_test_config.json")
        def teardown():
            shutil.rmtree('tests/output/images', ignore_errors=True)
            shutil.rmtree('tests/output/masks', ignore_errors=True)
            shutil.rmtree('tests/output/annotations', ignore_errors=True)

        request.addfinalizer(teardown)

    def test_exception(self, test_config, exception_class, err_msg):
        with pytest.raises(exception_class) as e:
            builder = Builder(test_config)
            builder.process_and_save()

        assert str(e.value) == err_msg
