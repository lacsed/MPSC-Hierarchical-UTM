from setuptools import setup

package_name = 'utm_graph'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'networkx'],
    zip_safe=True,
    maintainer='mploures',
    maintainer_email='matheuspaiva1024@gmail.com',
    description='UTM graph loader (graph_nodes.csv + graph_edges.csv) with vehicle spawn separation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'utm_graph_dump = utm_graph.cli_dump:main',
        ],
    },
)
