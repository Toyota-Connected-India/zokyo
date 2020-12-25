from sphinx.augmentation import Builder

pipeline = Builder(config_json="tests/sun_flare_test_config.json")
pipeline.calculate_and_set_generator_params(batch_size=1)
gen = pipeline.process_and_generate()

while True:
    try:
        next(gen)
    except StopIteration:
        break
