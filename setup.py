from setuptools import setup, find_packages

setup(
    name='elevate3d',
    version='0.5.0',
    author='Ömer Can Karadağ',
    author_email='krdg.omercan@hotmail.com',
    description='3D terrain and structure reconstruction from single RGB images',
    long_description=open('README_pypi.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/krdgomer/elevate3d',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'open3d',
        'numpy',
        'pillow',
        'opencv-python',
        'torch',
        'albumentations',
        'deepforest',
        'scikit-image',
        'huggingface_hub',
        'flask',
        'trimesh',
        'matplotlib',
        'scipy'
            ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'elevate3d-run=elevate3d.app:run_app',
        ],
    },
    python_requires='>=3.8')