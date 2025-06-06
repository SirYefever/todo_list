from setuptools import setup, find_packages

setup(
    name="tsu_nlp",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "transformers",
        "torch",
        "tqdm",
        "sentencepiece",
        "protobuf"
    ]
) 