# Changelog

## 1.0.0 (2025-09-21)

### Breaking changes

* The `Rater` now accepts mutable references to `Rating`s and updates them
  in-place instead of returning new nested values that had to be manually
  unpacked.

### Other changes

* Replaced custom serialization logic with Serde's derive macros.
* Rankings can now be specified more flexibly (`AsRef<[usize]>`), so arrays, `Vec`s, and slices will work.
* `Rater` and `Rating` now implement `Copy`.
* `update_ratings` now returns a proper `BBTError` type instead of a string.
* New function: `Rating::conservative_estimate`.

## [0.2.0] (2018-08-25)

* Added optional dependency on `serde` to make `Rating` serializable.

## [0.1.0]

* Initial release
