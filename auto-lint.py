import os
from tqdm import tqdm

SPHINX_DIR = "sphinx"

for root, dirs, files in os.walk(SPHINX_DIR):
    modules = root.split(os.sep)
    print("Linting module {}..".format(modules[-1]))
    for file in tqdm(files):
        if file.split(".")[-1] == "py" and "__pycache__" not in modules:
            os.system("autopep8 -i {0} --in-place --aggressive".format(os.path.join(root, file)))


