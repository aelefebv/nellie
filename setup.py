from setuptools import setup, find_packages

setup(
    name='nelly-organelles',
    version='0.1.0',
    packages=find_packages(),
    description='woaaaah',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Austin E. Y. T. Lefebvre',
    author_email='austin.e.lefebvre@gmail.com',
    url='https://github.com/aelefebv/nelly',
    install_requires=[
        'numpy',  # List all dependencies here
        'tifffile',
        'nd2',
        'ome-types',
        'scipy',
        'scikit-learn',
        'pandas',
        'umap-learn',
        'scikit-learn',
        'matplotlib',
        'napari[all]',
        'plotly',
        'seaborn',
        'scikit-image',
        # 'cupy-cuda11x',
    ],
    python_requires='>=3.9',
)