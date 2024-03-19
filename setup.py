from setuptools import setup, find_packages

setup(
    name='nellie',
    version='0.0.1',
    packages=find_packages(),
    package_data={
        "nellie_napari": ["napari.yaml", "logo.png"],
    },
    description='woaaaah',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Austin E. Y. T. Lefebvre',
    author_email='austin.e.lefebvre@gmail.com',
    url='https://github.com/aelefebv/nellie',
    install_requires=[
        'numpy==1.26.4',
        'scipy==1.12.0',
        'scikit-image==0.22.0',
        'nd2==0.9.0',
        'ome-types==0.5.0',
        'pandas==2.2.1',
        'matplotlib==3.8.3',
        'napari[all]',
    ],
    python_requires='>=3.12',
)
