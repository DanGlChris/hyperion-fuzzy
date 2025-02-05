"""
    Setup file for hyperion-fuzzy.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.6.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""

from setuptools import setup

if __name__ == "__main__":
    try:
        setup(
            name='fuzzy_classifier',
            version='0.1.0',
            author='Your Name',
            author_email='danglchris.manage@gmail.com',
            description='A Conformal transformation twin-hypersphere multi-cluster with fuzzy membership.',
            long_description=open('README.md').read(),
            long_description_content_type='text/markdown',
            url='https://github.com/DanGlChris/Hyperion-Fuzzy.git',  # Update with your repo URL
            packages=find_packages(),
            install_requires=[
                'numpy',
                'pandas',
                'scipy',
                'memory_profiler',
            ],
            classifiers=[
                'Programming Language :: Python :: 3',
                'License :: OSI Approved :: MIT License',
                'Operating System :: OS Independent',
            ],
            python_requires='>=3.6',
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
