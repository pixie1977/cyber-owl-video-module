from setuptools import setup, find_packages

setup(
    name="cyber-owl-camera",
    version="0.1.0",
    description="Video module for Cyber Owl",
    packages=find_packages(where="app"),
    package_dir={"": "app"},
    install_requires=[
        "torch",
        "torchaudio",
        "omegaconf",
        "pygame",
        "python-dotenv",
        "fastapi>=0.129.0",
        "uvicorn>=0.39.0",
        "python-multipart",
        "opencv-python-headless>=4.5.5.64",
        "numpy>=1.23.5",
    ],
    entry_points={
        "console_scripts": [
            "cyber-owl-camera=cyber_owl_video.app:main",
        ],
    },
)