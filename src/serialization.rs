use std::fmt;

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

use crate::Rating;

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

impl<'de> Deserialize<'de> for Rating {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            Mu,
            Sigma,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("`mu` or `sigma`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "mu" => Ok(Field::Mu),
                            "sigma" => Ok(Field::Sigma),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct RatingVisitor;

        impl<'de> Visitor<'de> for RatingVisitor {
            type Value = Rating;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Rating")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Rating, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let mu = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let sigma = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                Ok(Rating::new(mu, sigma))
            }

            fn visit_map<V>(self, mut map: V) -> Result<Rating, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut mu = None;
                let mut sigma = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Mu => {
                            if mu.is_some() {
                                return Err(de::Error::duplicate_field("mu"));
                            } else {
                                mu = Some(map.next_value()?);
                            }
                        }
                        Field::Sigma => {
                            if sigma.is_some() {
                                return Err(de::Error::duplicate_field("sigma"));
                            } else {
                                sigma = Some(map.next_value()?);
                            }
                        }
                    }
                }
                let mu = mu.ok_or_else(|| de::Error::missing_field("mu"))?;
                let sigma = sigma.ok_or_else(|| de::Error::missing_field("sigma"))?;
                Ok(Rating::new(mu, sigma))
            }
        }

        const FIELDS: &[&str] = &["mu", "sigma"];
        deserializer.deserialize_struct("Rating", FIELDS, RatingVisitor)
    }
}
