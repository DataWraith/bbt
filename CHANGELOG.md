# Changelog

## [0.3.0] (2020-09-26)

* Changed the type signature of `update_ratings` to return an error 
  enum on failure and `()` (unit) on success.
* Changed `update_ratings` to take a mutable self reference and
  mutable slices for performance reasons. It also updates the ratings
  *in place*. Head-on duels should be more than twice as fast now.
* Implemented `Copy` for `Rating`.
* Added a very basic dueling performance benchmark.

## [0.2.0] (2018-08-25)

* Added optional dependency on `serde` to make `Rating` serializable.

## [0.1.0]

* Initial release
