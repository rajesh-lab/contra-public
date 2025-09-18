import setuptools

package_dependencies = [
    ('numpy', '1.14.0'),
    ('matplotlib', '2.0.0'),
    ('mkl', '2019.0.0'),
    ('scipy', '1.2.1'),
    ('scikit-learn', '0.19.0'),
    ('tqdm', '4.0.1')
]

dependencies = [f'{p}>={v}' for p, v in package_dependencies]
requires = [f'{p} (>={v})' for p, v in package_dependencies]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="amicrt",
    version="0.1.1",
    author="Mukund Sudarshan",
    author_email="ms7490+amicrt-github@nyu.edu",
    description="Model-based conditional independence tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rajesh-lab/ami-crt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
    install_requires=dependencies
)

