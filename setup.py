from os.path import exists
from setuptools import setup
import versioneer

packages = ['dask_gdf',
            'dask_gdf.tests']

setup(name='dask_gdf',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='A Partitioned GPU DataFrame',
      license="Apache Software License 2.0",
      url='https://github.com/gpuopenanalytics/dask_gdf',
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      packages=packages,
      zip_safe=False)
