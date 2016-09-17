pub struct Rater {
    beta: f64
}

#[derive(PartialEq)]
pub struct Rating {
    mu: f64,
    sigma: f64
}

impl Default for Rating {
    fn default() -> Rating {
        Rating{ mu: 25.0, sigma: 25.0/3.0 }
    }
}

impl PartialOrd for Rating {
    fn partial_cmp(&self, other: &Rating) -> Option<std::cmp::Ordering> {
        (self.mu - 3.0 * self.sigma).partial_cmp(&(other.mu - 3.0 * other.sigma))
    }
}

impl Rating {
    fn new(mu: f64, sigma: f64) -> Rating {
        Rating { mu: mu, sigma: sigma }
    }
}

#[cfg(test)]
mod test {

    #[test]
    fn can_instantiate_ratings() {
        let default_rating = ::Rating::default();
        let new_rating = ::Rating::new(25.0, 25.0/3.0);
        assert!(default_rating == new_rating)
    }
}
