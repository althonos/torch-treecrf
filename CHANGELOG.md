# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
[Unreleased]: https://github.com/althonos/torch-treecrf/compare/v0.2.0...HEAD


## [v0.2.0] - 2024-02-16
[v0.2.0]: https://github.com/althonos/torch-treecrf/compare/v0.1.1...v0.2.0

### Changed
- Make `TreeCRFLayer` zero-initialize transition weights.
- Use vectorization and broadcasting to compute message passing.
- Bump Torch dependency to `v2.0`.
- Swapped last two dimensions of tensors returned from `TreeCRFLayer` for `CrossEntropyLoss` compatibility.

### Removed
- Dedicated `TreeMatrix` type for storing a hierarchy.


## [v0.1.1] - 2023-03-19
[v0.1.1]: https://github.com/althonos/torch-treecrf/compare/v0.1.0...v0.1.1

### Fixed
- Extraction of final probabilities in `TreeCRF.forward`.


## [v0.1.0] - 2023-01-17
[v0.1.0]: https://github.com/althonos/torch-treecrf/compare/e8ae1847...v0.1.0

Initial release.
