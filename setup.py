from setuptools import setup
import versioneer

packages = ['dask_gdf']

setup(name='dask_gdf',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='A Partitioned GPU DataFrame',
      url='https://github.com/gpuopenanalytics/dask_gdf',
      packages=packages,
      zip_safe=False)
