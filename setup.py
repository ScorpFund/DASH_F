from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("github_requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="market-pattern-explorer",
    version="1.0.0",
    author="Market Pattern Explorer Contributors",
    author_email="your-email@example.com",
    description="A sophisticated stock analysis tool with machine learning pattern clustering and visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/market-pattern-explorer",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "market-pattern-explorer=app:main",
        ],
    },
)