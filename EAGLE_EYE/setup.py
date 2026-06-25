from setuptools import setup, find_packages

setup(
    name='eagle-eye',
    version='1.0.0',
    description='Accelerating LVMs ',
    packages=find_packages(),
    install_requires=[
        "accelerate == 1.6.0",
        "transformers == 4.51.1",
        "safetensors >= 0.4.5",
        "numpy >= 1.24, < 2.0",
        "tqdm >= 4.66",
        "einops >= 0.7",
        "pyyaml >= 6.0",
        "matplotlib >= 3.7",
        "fschat == 0.2.31",
        "gradio == 3.50.2",
        "openai == 0.28.0",
        "anthropic == 0.5.0",
        "sentencepiece == 0.1.99",
        "protobuf >= 3.20.3, < 5",
        "wandb >= 0.16"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
