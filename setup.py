from setuptools import setup, find_packages

setup(
    name='model_comparison',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'librosa',
        'numpy',
        'keras',
        'scikit-learn',
        'xgboost'
    ],
    author='FourLayer',
    author_email='fourlayerpbl@gmail.com',
    description='A library for comparing four machine learning models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/FourLayer/FinalCode/fourmodels.py',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
