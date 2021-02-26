from setuptools import setup, find_packages, Extension

"""
Setup file for k-Means implementation in C
"""

setup(
    name='kmeanspp',
    version='1.0.1',
    description='K-Means Implementation In C',
    install_requires=['invoke'],
    packages=find_packages(),
    ext_modules=[Extension('kmeanspp', sources=['kmeans.c'])]
)

