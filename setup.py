from setuptools import setup

version = '0.0.1'
packages = ['dask_gdf']

setup(name='dask_gdf',
      version=version,
      description='A Partitioned GPU DataFrame',
      url='https://github.com/gpuopenanalytics/dask_gdf',
      packages=packages,
      zip_safe=False)
