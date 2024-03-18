import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modelcompare", # 모듈 이름
    version="0.1.0", # 버전
    author="Layer4", # 제작자
    author_email="layerfourpbl@gmail.com", # contact
    description="model-for-easy-comparing", # 모듈 설명
    long_description=open('README.md').read(), 
    long_description_content_type="text/markdown",
    url="https://github.com/FourLayer/FinalCode.git",
    install_requires=[ 
    "matplotlib==3.5.2", 
    "numpy==1.21.5", 
    "pandas==1.4.4", 
    "scikit_learn==1.2.0", 
    "scipy==1.9.1", 
    "seaborn==0.11.2", 
    "setuptools==63.4.1", 
    ],
    package_data={'': ['LICENSE.txt', 'requirements.txt']}, # 원하는 파일 포함, 제대로 작동되지 않았음
    include_package_data=True,
    packages = setuptools.find_packages(), # 모듈을 자동으로 찾아줌
    python_requires=">=3.9.13", # 파이썬 최소 요구 버전
)
