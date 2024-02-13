from setuptools import find_packages, setup

package_name = 'camera_controller'
submodule= 'camera_controller/submodulos'
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
    maintainer='rhobtor',
    maintainer_email='rhobtor@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node=camera_controller.camera_node:main',
            'distance_move=camera_controller.distance_move:main',
            'distance_cells=camera_controller.distnce_move:main',
            'Zed_node=camera_controller.Zed_node:main',
        ],
    },
)
