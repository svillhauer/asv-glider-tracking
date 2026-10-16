from setuptools import find_packages, setup

package_name = 'asv_bridge_stable'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/stable_nondim.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='svillhauer',
    maintainer_email='svillhauer@g.ucla.edu',
    description='ROS 2 bridge for the stable reference policy (nglider_knowndepth_flat_fixed2_stable_s0).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'stable_bridge_node     = asv_bridge_stable.bridge_node:main',
            'stable_policy_node     = asv_bridge_stable.policy_node:main',
            'stable_glider_viz_node = asv_bridge_stable.glider_viz_node:main',
            'stable_attach_current  = asv_bridge_stable.attach_current_node:main',
        ],
    },
)
