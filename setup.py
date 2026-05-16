from setuptools import find_packages, setup
from glob import glob

package_name = 'vslam_semantic'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/scripts', glob('scripts/*.sh')),
    ],
    install_requires=['setuptools', 'redis', 'numpy'],
    zip_safe=True,
    maintainer='Sharvesh',
    maintainer_email='sharvesh@example.com',
    description='cuVSLAM + NanoOWL + Pixhawk + Redis semantic graph for D415.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'px4_imu_bridge_node = vslam_semantic.px4_imu_bridge_node:main',
            'nanoowl_inference_node = '
                'vslam_semantic.nanoowl_inference_node:main',
            'semantic_graph_combiner_node = '
                'vslam_semantic.semantic_graph_combiner_node:main',
            'redis_writer_node = vslam_semantic.redis_writer_node:main',
        ],
    },
)
