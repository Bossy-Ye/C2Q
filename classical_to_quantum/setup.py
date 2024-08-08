from setuptools import setup, find_packages

setup(
    name="Quantum Translation Framework",
    version="0.1.0",
    description="A brief description of your project",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Boshuai Ye",
    author_email="boshuaiye@gmail.com",
    url="https://github.com/yourusername/yourproject",
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        "required_package1",
        "required_package2>=1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'your_command=your_module:main_function',
        ],
    },
    include_package_data=True,
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
        # Include any *.msg files found in the 'hello' package, too:
        'hello': ['*.msg'],
    },
)
