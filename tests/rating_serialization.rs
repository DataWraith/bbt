#![cfg(feature="serde")]
/// Using a system test for serialization feature, as `serde_json` should not be included as a crate for builds, only for testing.

extern crate serde;
extern crate serde_json;
extern crate bbt;

use bbt::Rating;

#[test]
fn end_to_end() {
    let original = Rating::default();

    let serialized = serde_json::to_string(&original).unwrap_or_else(|_| panic!("Failed to serialize {:?}", original));
    let deserialized: Rating = serde_json::from_str(&serialized).unwrap_or_else(|_| panic!("Failed to deserialize {}", &serialized));

    assert_eq!(original, deserialized);
}