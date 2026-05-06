from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'laser_weeder'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='LaserWeeder Dev',
    maintainer_email='dev@example.com',
    description='LaserWeeder ROS2 nodes',
    license='Proprietary',
    entry_points={
        'console_scripts': [
            'camera_node      = laser_weeder.camera_node:main',
            'recorder_node    = laser_weeder.recorder_node:main',
            'cv_node          = laser_weeder.cv_node:main',
            'triangulation_node = laser_weeder.triangulation_node:main',
            'gantry_node      = laser_weeder.gantry_node:main',
            'fine_align_node  = laser_weeder.fine_align_node:main',
        ],
    },
)
