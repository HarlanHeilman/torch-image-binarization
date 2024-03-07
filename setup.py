from setuptools import find_packages, setup


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="torch_image_binarization",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "main = torch_image_binarization:main",
        ]
    },
)
