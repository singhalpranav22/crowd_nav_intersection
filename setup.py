"""
Script to install the dependencies to run the project.
Use the command pip install -e . to install the required dependencies for the project
"""

from setuptools import setup

setup(
    name='crowdnav',
    version='0.0.3',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.policy',
        'crowd_nav.utils',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.utils',
        'crowd_sim.envs.policy',
    ],
    install_requires=[
        'gym==0.17.0',
        'gitpython',
        'matplotlib',
        'numpy',
        'scipy',
        'torch',
        'torchvision',
        'pandas',
        'opencv-python',
        'PySimpleGUI'
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
