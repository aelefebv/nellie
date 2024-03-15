from setuptools import setup, find_packages

setup(
    name='nellie',
    version='0.0.1',
    packages=find_packages(),
    description='woaaaah',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Austin E. Y. T. Lefebvre',
    author_email='austin.e.lefebvre@gmail.com',
    url='https://github.com/aelefebv/nellie',
    install_requires=[
    ],
    python_requires='>=3.12',
)