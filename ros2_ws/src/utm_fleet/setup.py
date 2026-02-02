from setuptools import setup

package_name = 'utm_fleet'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'networkx'],
    zip_safe=True,
    maintainer='mploures',
    maintainer_email='matheuspaiva1024@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'utm_fleet_demo = utm_fleet.fleet_controller:main',
        ],
    },
)
