from setuptools import find_packages, setup

package_name = 'vmc_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='visitorsproject',
    maintainer_email='visitorsproject@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vision_node = vmc_vision.vision_node:main',
            'vision_kalman  = vmc_vision.vision_node_kalman:main',
            'vision_pcl = vmc_vision.vision_node_pcl:main',
        ],
    },
)
