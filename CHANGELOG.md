# Changelog

<!--next-version-placeholder-->

## v0.2.1 (16/03/2026)

### Fixed
- Fix CUDA mode: `cuda_context` initialization, CuPy scalar conversion, parent selection compatibility

### Changed
- Replace 3 custom CUDA kernels (weights, reduction, update) with native CuPy operations
- Remove Kahan summation algorithm from reduction (unnecessary in float64)
- Retain only `calculateDistancesKernel` as custom kernel (shared memory optimization)
- Remove unused modules `metrics.py`, `funcs.py`, and `dataviz`
- Update API documentation to reflect current module structure

## v0.2.0 (2026)

### Changed
- Configure TestPyPI and PyPI publishing indexes

## v0.1.0 (23/09/2024)

- First release of `pyecsago`!