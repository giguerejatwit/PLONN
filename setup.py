from setuptools import setup, find_packages

setup(
    name='PLONN',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
        # Example: 'numpy', 'pandas'
    ],
    entry_points={
        'console_scripts': [
            # Define command-line scripts here
            # Example: 'script_name = module:function'
        ],
    },
    author='Jake Giguere',
    author_email='giguere.jake@gmail.com',
    description='PLONN NBA',
    url='https://github.com/jakegiguere/assignment2',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
