use std::collections::HashMap;

pub struct Rater {
    beta_sq: f64
}

impl Rater {
    fn new(beta: f64) -> Rater {
        Rater { beta_sq: beta * beta }
    }
}

impl Default for Rater {
    fn default() -> Rater {
        Rater::new(25.0/6.0)
    }
}

impl Rater {
    fn update_ratings(&self,
                      players: Vec<Rating>,
                      team_nr: Vec<usize>,
                      rank: HashMap<usize, usize>) -> Result<Vec<Rating>, &'static str> {

        if players.len() != team_nr.len() {
            return Err("`players` and `team_nr` vectors must be of the same length")
        }

        let mut team_mu : HashMap<usize, f64> = HashMap::new();
        let mut team_sigma_sq : HashMap<usize, f64> = HashMap::new();
        let mut team_omega : HashMap<usize, f64> = HashMap::new();
        let mut team_delta : HashMap<usize, f64> = HashMap::new();

        ////////////////////////////////////////////////////////////////////////
        // Step 1 - Collect Team skill and variance ////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        for it in players.iter().zip(team_nr.iter()) {
            let (player, team_nr) = it;

            if team_mu.contains_key(team_nr) {
                *team_mu.get_mut(team_nr).unwrap() += player.mu;
            } else {
                team_mu.insert(*team_nr, player.mu);
            }

            if team_sigma_sq.contains_key(team_nr) {
                *team_sigma_sq.get_mut(team_nr).unwrap() += player.sigma_sq;
            } else {
                team_sigma_sq.insert(*team_nr, player.sigma_sq);
            }
        }

        for team in team_mu.keys() {
            if rank.get(team).is_none() {
                return Err("Missing rank information for at least one team");
            }
        }

        ////////////////////////////////////////////////////////////////////////
        // Step 2 - Compute Team Omega and Delta ///////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        for team in team_mu.keys() {
            for team2 in team_mu.keys().filter(|t| **t != *team) {
                let c = (team_sigma_sq.get(team).unwrap() + team_sigma_sq.get(team2).unwrap() + 2.0 * self.beta_sq).sqrt();
                let e1 = (team_mu.get(team).unwrap() / c).exp();
                let e2 = (team_mu.get(team2).unwrap() / c).exp();
                let piq = e1 / (e1 + e2);
                let pqi = e2 / (e1 + e2);
                let ri = rank.get(team).unwrap();
                let rq = rank.get(team2).unwrap();
                let s = if rq > ri {
                    1.0
                } else if rq == ri {
                    0.5
                } else {
                    0.0
                };
                let delta = team_sigma_sq.get(team).unwrap() / c * (s - piq);
                let y = team_sigma_sq.get(team).unwrap().sqrt() / c;
                let eta = y * (team_sigma_sq.get(team).unwrap() / (c * c)) * piq * pqi;


                if team_omega.contains_key(team) {
                    *team_omega.get_mut(team).unwrap() += delta;
                } else {
                    team_omega.insert(*team, delta);
                }

                if team_delta.contains_key(team) {
                    *team_delta.get_mut(team).unwrap() += eta;
                } else {
                    team_delta.insert(*team, eta);
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////
        // Step 3 - Individual skill update ////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        let mut result = Vec::with_capacity(players.len());
        for it in players.iter().zip(team_nr.iter()) {
            let (player, team_nr) = it;

            let new_mu = player.mu + team_omega.get(team_nr).unwrap() * (player.sigma_sq / team_sigma_sq.get(team_nr).unwrap());
            let mut sigma_adj = 1.0 - (player.sigma_sq / team_sigma_sq.get(team_nr).unwrap() * team_delta.get(team_nr).unwrap());
            if sigma_adj < 0.0001 {
                sigma_adj = 0.0001;
            };
            let new_sigma_sq = player.sigma_sq * sigma_adj;

            result.push(Rating { mu: new_mu, sigma: new_sigma_sq.sqrt(), sigma_sq: new_sigma_sq });
        }

        Ok(result)

    }
}

#[derive(PartialEq)]
pub struct Rating {
    mu: f64,
    sigma: f64,
    sigma_sq: f64
}

impl Default for Rating {
    fn default() -> Rating {
        Rating{ mu: 25.0, sigma: 25.0/3.0, sigma_sq: f64::powf(25.0/3.0, 2.0) }
    }
}

impl PartialOrd for Rating {
    fn partial_cmp(&self, other: &Rating) -> Option<std::cmp::Ordering> {
        (self.mu - 3.0 * self.sigma).partial_cmp(&(other.mu - 3.0 * other.sigma))
    }
}

impl Rating {
    fn new(mu: f64, sigma: f64) -> Rating {
        Rating { mu: mu, sigma: sigma, sigma_sq: sigma.powf(2.0) }
    }
}

// TODO: Serialization/Deserialization for Rating

#[cfg(test)]
mod test {

    #[test]
    fn can_instantiate_ratings() {
        let default_rating = ::Rating::default();
        let new_rating = ::Rating::new(25.0, 25.0/3.0);
        assert!(default_rating == new_rating)
    }

    #[test]
    fn two_player_duel() {
        let p1 = ::Rating::default();
        let p2 = ::Rating::default();

        let rater = ::Rater::default();
        let mut ranks = ::std::collections::HashMap::new();
        ranks.insert(0, 0);
        ranks.insert(1, 1);
        let new_rs = rater.update_ratings(vec![p1, p2], vec![0, 1], ranks).unwrap();

        assert!(new_rs[0].mu > new_rs[1].mu);
    }
}
