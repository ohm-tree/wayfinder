from setuptools import find_packages, setup

setup(
    name='wayfinder',
    packages=find_packages(
        include=['wayfinder', 'wayfinder.*']
    ),
    author='rowechen',
    author_email='rowechen@mit.edu',
    install_requires=[
        'numpy',
    ]
)
