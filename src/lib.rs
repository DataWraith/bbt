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
                      teams: Vec<Vec<Rating>>,
                      ranks: Vec<usize>) -> Result<Vec<Vec<Rating>>, &'static str> {

        if teams.len() != ranks.len() {
            return Err("`teams` and `ranks` vectors must be of the same length")
        }

        let mut team_mu = Vec::with_capacity(teams.len());
        let mut team_sigma_sq = Vec::with_capacity(teams.len());
        let mut team_omega = Vec::with_capacity(teams.len());
        let mut team_delta = Vec::with_capacity(teams.len());

        for _ in 0..teams.len() {
            team_mu.push(0.0);
            team_sigma_sq.push(0.0);
            team_omega.push(0.0);
            team_delta.push(0.0);
        }

        ////////////////////////////////////////////////////////////////////////
        // Step 1 - Collect Team skill and variance ////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        for (team_idx, team) in teams.iter().enumerate() {
            if team.is_empty() {
                return Err("At least one of the teams contains no players")
            }

            for player in team.iter() {
                team_mu[team_idx] += player.mu;
                team_sigma_sq[team_idx] += player.sigma_sq;
            }
        }

        ////////////////////////////////////////////////////////////////////////
        // Step 2 - Compute Team Omega and Delta ///////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        for (team_idx, _) in teams.iter().enumerate() {
            for (team2_idx, _) in teams.iter().enumerate() {
                if team_idx == team2_idx {
                    continue
                }

                let c = (team_sigma_sq[team_idx] + team_sigma_sq[team2_idx] + 2.0 * self.beta_sq).sqrt();
                let e1 = (team_mu[team_idx] / c).exp();
                let e2 = (team_mu[team2_idx] / c).exp();
                let piq = e1 / (e1 + e2);
                let pqi = e2 / (e1 + e2);
                let ri = ranks[team_idx];
                let rq = ranks[team2_idx];
                let s = if rq > ri {
                    1.0
                } else if rq == ri {
                    0.5
                } else {
                    0.0
                };
                let delta = (team_sigma_sq[team_idx] / c) * (s - piq);
                let gamma = team_sigma_sq[team_idx].sqrt() / c;
                let eta = gamma * (team_sigma_sq[team_idx] / (c * c)) * piq * pqi;

                team_omega[team_idx] += delta;
                team_delta[team_idx] += eta;
            }
        }

        ////////////////////////////////////////////////////////////////////////
        // Step 3 - Individual skill update ////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        let mut result = Vec::with_capacity(teams.len());

        for (team_idx, team) in teams.iter().enumerate() {
            let mut team_result = Vec::with_capacity(team.len());

            for player in team.iter() {
                let new_mu = player.mu + (player.sigma_sq / team_sigma_sq[team_idx]) * team_omega[team_idx];
                let mut sigma_adj = 1.0 - (player.sigma_sq / team_sigma_sq[team_idx]) * team_delta[team_idx];
                if sigma_adj < 0.0001 {
                    sigma_adj = 0.0001;
                }
                let new_sigma_sq = player.sigma_sq * sigma_adj;

                team_result.push(Rating { mu: new_mu, sigma: new_sigma_sq.sqrt(), sigma_sq: new_sigma_sq });
            }

            result.push(team_result);
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
        let new_rs = rater.update_ratings(vec![vec![p1], vec![p2]], vec![0, 1]).unwrap();

        assert!(new_rs[0][0].mu > new_rs[1][0].mu);
    }
}
