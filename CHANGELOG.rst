dask-cudf 0.7.0 (10 May 2019)
-----------------------------

-  Remove dependency on libgdf_cffi (#228) `Keith Kraus`_
-  Update build process `Rick Ratzel`_
-  Convert query to use standard dask query and update GPUCI to cudf 0.7 (#196) `Nick Becker`_
-  Update GPU CI to use cudf 0.7 (#204) `Nick Becker`_
-  Route single-partition merge cases through dask.dataframe (#194) `Matthew Rocklin`_
-  Avoid compression warning in read_csv if chunksize=None (#192) `Matthew Rocklin`_
-  Fix classifier (#182) `Ray Douglass`_
-  Fix gpuCI build script (#173) `Dillon Cullinan`_


0.6.1 - 2019-04-09
------------------

-  Add cudf.DataFrame.mean = None (#205) `Matthew Rocklin`_


dask-cudf 0.6.0 (22 Mar 2019)
-----------------------------

In this release we aligned Dask cuDF to the mainline Dask Dataframe
codebase.  This was made possible by an alignment of cuDF to Pandas, and
resulted in us maintaining much less code in this repository.  Dask cuDF
dataframes are now just Dask DataFrames that contain cuDF dataframes, and have
a few extra methods.

-  Bump cudf to 0.6 (#162) `Keith Kraus`_
-  Fix upload-anaconda to find the right package (#159) `Ray Douglass`_
-  Add gpuCI (#151) `Mike Wendt`_
-  Skip s3fs tests before importing dask.bytes.s3 (#153) `Matthew Rocklin`_
-  Raise FileNotFoundError if no files found (#145) `Benjamin Zaitlen`_
-  Add tests for repartition and indexed joins (#91) `Matthew Rocklin`_
-  URLs for CSVs (#122) `Benjamin Zaitlen`_
-  Rely on mainline concat function (#126) `Matthew Rocklin`_
-  add test for generic idx test using loc (#121) `Benjamin Zaitlen`_
-  Fix gzip `Benjamin Zaitlen`_
-  Replace custom make_meta with mainline make_meta (#105) `Matthew Rocklin`_
-  Cleanup dead code (#99) `Matthew Rocklin`_
-  Remove from_cudf and from_dask_dataframe functions (#98) `Matthew Rocklin`_
-  Increase default chunk size in read_csv (#95) `Matthew Rocklin`_
-  Remove assertions outlawing inner joins (#89) `Matthew Rocklin`_
-  Fix reset_index(drop=) keyword handling (#94) `Matthew Rocklin`_
-  Add index= keyword to make_meta dispatch functions `Matthew Rocklin`_
-  Use mainline groupby aggregation codebase (#69) `Matthew Rocklin`_
-  remove dtype inference on chunks of data when parsing csv (#86) `Matthew Rocklin`_
-  Avoid format strings to support Python 3.5 `Matthew Rocklin`_
-  use byte_range when reading CSVs (#78) `Benjamin Zaitlen`_
-  Move cudf dask backends code to backends file here (#75) `Matthew Rocklin`_
-  Clean up join code (#70) `Matthew Rocklin`_
-  Replace pygdf with cudf in README (#65) `Matthew Rocklin`_
-  Add dask_cudf.io to setup.py packages (#60) `Matthew Rocklin`_
-  Add basic read_csv implementation (#58) `Matthew Rocklin`_
-  Add tests for repr (#56) `Matthew Rocklin`_
-  Rename gd to cudf in tests `Matthew Rocklin`_
-  add style instructions to README `Matthew Rocklin`_
-  Apply isort to code `Matthew Rocklin`_
-  Add pre-commit-config.yaml including black and flake8 `Matthew Rocklin`_
-  Inherit from Dask Dataframe and respond to cudf update (#48) `Matthew Rocklin`_
-  updating for new cuDF API `Matthew Jones`_
-  add orc reader (#220) `Benjamin Zaitlen`_

.. _`Matthew Jones`: https://github.com/mt-jones
.. _`Keith Kraus`: https://github.com/kkraus14
.. _`Ray Douglass`: https://github.com/raydouglass
.. _`Matthew Rocklin`: https://github.com/mrocklin
.. _`Benjamin Zaitlen`: https://github.com/quasiben
.. _`Mike Wendt`: https://github.com/mike-wendt
.. _`Dillon Cullinan`: https://github.com/dillon-cullinan
.. _`Nick Becker`: https://github.com/beckernick
