from setuptools import setup, find_packages

setup(
    name="pat-cv-lifecycle",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # 你的依赖项
    ],
)