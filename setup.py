from os.path import exists
from setuptools import setup
import versioneer

packages = ["dask_cudf", "dask_cudf.io"]

packages = packages + [p + ".tests" for p in packages]

setup(
    name="dask_cudf",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A Partitioned GPU DataFrame",
    license="Apache Software License 2.0",
    url="https://github.com/rapidsai/dask_cudf",
    long_description=(open("README.rst").read() if exists("README.rst") else ""),
    packages=packages,
    zip_safe=False,
)
