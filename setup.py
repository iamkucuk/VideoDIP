from setuptools import setup, find_packages

setup(
    name='video_dip',
    version='0.1.0',
    author='Furkan & Alper',
    # author_email='your.email@example.com',
    description='A project for video deep image prior processing',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://github.com/yourusername/video_dip',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'numpy',
        # Add other dependencies required by your project
    ],
    entry_points={
        'console_scripts': [
            'video_dip=video_dip.module:main',
        ],
    },
)