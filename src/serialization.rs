use serde::{Serialize, Serializer};
use Rating;
use serde::ser::SerializeStruct;
use serde::de::{self, Deserialize, Deserializer, Visitor, SeqAccess, MapAccess};

impl Serialize for Rating {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Rating", 2)?;
        state.serialize_field("mu", &self.mu)?;
        state.serialize_field("sigma", &self.sigma)?;
        state.end()
    }
}

