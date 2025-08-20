//! BBT is an implementation of a skill-rating system similar to Elo, Glicko or
//! TrueSkill. It follows `Algorithm 1` from the paper [A Bayesian Approximation
//! Method for Online Ranking][ABAMOR].
//!
//! [ABAMOR]: http://jmlr.csail.mit.edu/papers/volume12/weng11a/weng11a.pdf
//!
//! ## Usage
//!
//! As a first step, you need to instantiate a Rater:
//!
//! ```rust
//! let rater = bbt::Rater::new(25.0/6.0);
//! ```
//!
//! The new() function takes one parameter, β. This parameter describes how much
//! randomness (variance in outcomes) your game has. For example, a game like
//! Hearthstone is much more luck-based than chess and should have a higher
//! variance; you may need to experiment to see which value has the highest
//! predictive power.
//!
//! ### Two-player games (e.g. Chess)
//!
//! BBT has a convenience function for two-player games that returns the new
//! ratings for the two players after a game. In the example, p1 wins against
//! p2:
//!
//! ```rust
//! let rater = bbt::Rater::default();
//!
//! let p1 = bbt::Rating::default();
//! let p2 = bbt::Rating::default();
//!
//! let (new_p1, new_p2) = rater.duel(p1, p2, bbt::Outcome::Win);
//! ```
//!
//! The `bbt::Outcome` enum can take on the values `Win`, `Loss` and `Draw`.
//!
//! ### Multiplayer games
//!
//! Games with more than two players will have to use the general
//! `update_ratings` method. It takes a vector of teams and a vector of ranks,
//! with each team being a vector of player ratings. If no error occurs, the
//! method returns a vector of the same form as the input with updated ratings.
//!
//! #### Example 1: Racing Game
//!
//! In a racing game without teams, each player is represented as a "team" of
//! one, and since there are usually no ties in a racing game, the list of ranks
//! contains no duplicates:
//!
//! ```rust
//! let rater = bbt::Rater::default();
//!
//! let p1 = bbt::Rating::default();
//! let p2 = bbt::Rating::default();
//! let p3 = bbt::Rating::default();
//! let p4 = bbt::Rating::default();
//! let p5 = bbt::Rating::default();
//! let p6 = bbt::Rating::default();
//!
//! let team1 = [p1]; let team2 = [p2]; let team3 = [p3];
//! let team4 = [p4]; let team5 = [p5]; let team6 = [p6];
//! let teams = [&team1[..], &team2[..], &team3[..], &team4[..], &team5[..], &team6[..]];
//! let new_ratings = rater.update_ratings(&teams, [1, 2, 3, 4, 5, 6]).unwrap();
//! ```
//!
//! In the example, the first player places first, the second player second, and
//! so on.
//!
//! #### Example 2: Tied Teams
//!
//! Let's say you have a hypothetical game with four teams and two players per
//! team.
//!
//! | Team 1 | Team 2  | Team 3 | Team 4 |
//! | ------ | ------- | ------ | ------ |
//! | Alice  | Charlie | Eve    | Gabe   |
//! | Bob    | Dave    | Fred   | Henry  |
//!
//! If Team 1 wins, and Team 2 and 3 draw for second place and Team 4 loses, you
//! can call the `update_ratings` function as follows:
//!
//! ```rust
//! let rater = bbt::Rater::default();
//!
//! let alice   = bbt::Rating::default();
//! let bob     = bbt::Rating::default();
//! let charlie = bbt::Rating::default();
//! let dave    = bbt::Rating::default();
//! let eve     = bbt::Rating::default();
//! let fred    = bbt::Rating::default();
//! let gabe    = bbt::Rating::default();
//! let henry   = bbt::Rating::default();
//!
//! let team1 = [alice, bob]; let team2 = [charlie, dave];
//! let team3 = [eve, fred]; let team4 = [gabe, henry];
//! let teams = [&team1[..], &team2[..], &team3[..], &team4[..]];
//! let new_ratings = rater.update_ratings(&teams, [1, 2, 2, 4]).unwrap();
//! ```
//!
//! The second argument assigns a rank to the teams given in the first vector.
//! Team 1 placed first, teams 2 and 3 tie for second place and team 4 comes in
//! fourth.
//!
//! ## Rating scale
//!
//! The default rating scale follows TrueSkill's convention of ranks from 0 to 50.
//! You should be able to use a different scale by specifying the middle of that
//! scale in `Rating::new()`. For example, to use a more traditional scale of 0 to
//! 3000, you can initialize ratings with `Rating::new(1500.0, 1500.0/3.0)`. You'll
//! also need to adjust the β-value of the Rater instance accordingly:
//! `Rater::new(1500.0/6.0)`.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

/// Rater is used to calculate rating updates given the β-parameter.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct Rater {
    beta_sq: f64,
}

impl Rater {
    /// This method instantiates a new rater with the given β-parameter.
    pub fn new(beta: f64) -> Rater {
        Rater {
            beta_sq: beta * beta,
        }
    }
}

impl Default for Rater {
    /// This method instantiates a new rater with the default β-parameter of
    /// 25.0/6.0 used in the paper.
    fn default() -> Rater {
        Rater::new(25.0 / 6.0)
    }
}

impl std::fmt::Display for Rater {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rater(β={:.4})", self.beta_sq.sqrt())
    }
}

impl Rater {
    /// This method takes a slice of teams, with each team being a slice of
    /// player ratings, and a slice, Vec or array of ranks of the same size. The
    /// ranks specify the ranking of the corresponding team in the game. It
    /// returns either
    ///
    /// `Err(BBTError)` if the input is incorrect or
    /// `Ok(Vec<Vec<Rating>>)`.
    ///
    /// The returned vector contains updated ratings for each team.
    pub fn update_ratings<Ranks>(
        &self,
        teams: &[&[Rating]],
        ranks: Ranks,
    ) -> Result<Vec<Vec<Rating>>, BBTError>
    where
        Ranks: AsRef<[usize]>,
    {
        let ranks = ranks.as_ref();

        if teams.len() != ranks.len() {
            return Err(BBTError::MismatchedVectorLengths);
        }

        let mut team_mu = vec![0.0; teams.len()];
        let mut team_sigma_sq = vec![0.0; teams.len()];
        let mut team_omega = vec![0.0; teams.len()];
        let mut team_delta = vec![0.0; teams.len()];

        ////////////////////////////////////////////////////////////////////////
        // Step 1 - Collect Team skill and variance ////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        for (team_idx, team) in teams.iter().enumerate() {
            if team.is_empty() {
                return Err(BBTError::EmptyTeam);
            }

            for player in team.iter() {
                team_mu[team_idx] += player.mu;
                team_sigma_sq[team_idx] += player.sigma_sq();
            }
        }

        ////////////////////////////////////////////////////////////////////////
        // Step 2 - Compute Team Omega and Delta ///////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        for team_idx in 0..teams.len() {
            for team2_idx in 0..teams.len() {
                if team_idx == team2_idx {
                    continue;
                }

                let c = (team_sigma_sq[team_idx] + team_sigma_sq[team2_idx] + 2.0 * self.beta_sq)
                    .sqrt();
                let e1 = (team_mu[team_idx] / c).exp();
                let e2 = (team_mu[team2_idx] / c).exp();
                let piq = e1 / (e1 + e2);
                let pqi = e2 / (e1 + e2);
                let ri = ranks[team_idx];
                let rq = ranks[team2_idx];

                let s = match rq.cmp(&ri) {
                    Ordering::Greater => 1.0,
                    Ordering::Equal => 0.5,
                    Ordering::Less => 0.0,
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
                let new_mu = player.mu
                    + (player.sigma_sq() / team_sigma_sq[team_idx]) * team_omega[team_idx];

                let mut sigma_adj =
                    1.0 - (player.sigma_sq() / team_sigma_sq[team_idx]) * team_delta[team_idx];

                if sigma_adj < 0.0001 {
                    sigma_adj = 0.0001;
                }

                let new_sigma_sq = player.sigma_sq() * sigma_adj;

                team_result.push(Rating {
                    mu: new_mu,
                    sigma: new_sigma_sq.sqrt(),
                });
            }

            result.push(team_result);
        }

        Ok(result)
    }

    /// This method calculates the new ratings for two players after a
    /// head-to-head duel. The outcome is from the first player `p1`'s
    /// perspective, i.e. `Win` if the first player won, `Loss` if the second
    /// player won and `Draw` if neither player won.
    pub fn duel(&self, p1: Rating, p2: Rating, outcome: Outcome) -> (Rating, Rating) {
        let team1 = [p1];
        let team2 = [p2];
        let teams = [&team1[..], &team2[..]];
        let ranks = match outcome {
            Outcome::Win => [1, 2],
            Outcome::Loss => [2, 1],
            Outcome::Draw => [1, 1],
        };

        let result = self.update_ratings(&teams, ranks).unwrap();

        (result[0][0], result[1][0])
    }
}

/// Outcome represents the outcome of a head-to-head duel between two players.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Outcome {
    /// The first player won the game
    Win,

    /// The first player lost the game
    Loss,

    /// Neither player won
    Draw,
}

/// Rating represents the skill of a player.
#[derive(PartialEq, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Rating {
    mu: f64,
    sigma: f64,
}

impl Default for Rating {
    /// Instantiates a Rating with the default values of mu=25.0 and sigma=25.0/3.0
    fn default() -> Rating {
        Rating {
            mu: 25.0,
            sigma: 25.0 / 3.0,
        }
    }
}

impl PartialOrd for Rating {
    fn partial_cmp(&self, other: &Rating) -> Option<std::cmp::Ordering> {
        self.conservative_estimate()
            .partial_cmp(&other.conservative_estimate())
    }
}

impl fmt::Display for Rating {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.conservative_estimate())
    }
}

impl fmt::Debug for Rating {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}±{}", self.mu, 3.0 * self.sigma)
    }
}

impl Rating {
    pub fn new(mu: f64, sigma: f64) -> Rating {
        Rating { mu, sigma }
    }

    /// Returns the estimated skill of the player.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Returns the variance on the estimate of the player's skill.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Returns the conservative estimate of the player's skill.
    pub fn conservative_estimate(&self) -> f64 {
        (self.mu - 3.0 * self.sigma).max(0.0)
    }

    fn sigma_sq(&self) -> f64 {
        self.sigma.powf(2.0)
    }
}

/// Error type for BBT rating calculation
#[derive(Debug, Clone, PartialEq)]
pub enum BBTError {
    /// The teams and ranks have different lengths.
    MismatchedVectorLengths,
    /// One or more teams contain no players.
    EmptyTeam,
}

impl fmt::Display for BBTError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BBTError::MismatchedVectorLengths => {
                write!(f, "`teams` and `ranks` must be of the same length")
            }
            BBTError::EmptyTeam => {
                write!(f, "At least one of the teams contains no players")
            }
        }
    }
}

impl Error for BBTError {}

#[cfg(test)]
mod test {
    use crate::{BBTError, Outcome, Rater, Rating};

    // Tests for Rater struct
    #[test]
    fn rater_display() {
        let beta = 4.0;
        let rater = Rater::new(beta);
        let display_str = format!("{}", rater);
        assert!(display_str.contains("Rater(β=4.0000)"));
    }

    // Tests for Rating struct
    #[test]
    fn can_instantiate_ratings() {
        let default_rating = Rating::default();
        let new_rating = Rating::new(25.0, 25.0 / 3.0);

        assert_eq!(default_rating, new_rating)
    }

    #[test]
    fn rating_getters() {
        let rating = Rating::default();
        assert_eq!(rating.mu(), 25.0);
        assert_eq!(rating.sigma(), 25.0 / 3.0);
    }

    #[test]
    fn rating_display() {
        let rating = Rating::new(25.0, 8.0);
        let display_str = format!("{}", rating);
        // Conservative estimate should be max(0.0, mu - 3*sigma) = max(0.0, 25 - 24) = 1
        assert_eq!(display_str, "1");
    }

    #[test]
    fn rating_debug() {
        let rating = Rating::new(25.0, 8.0);
        let debug_str = format!("{:?}", rating);
        assert_eq!(debug_str, "25±24");
    }

    #[test]
    fn rating_display_negative_conservative_estimate() {
        let rating = Rating::new(5.0, 8.0);
        let display_str = format!("{}", rating);
        // Conservative estimate would be negative, so should show 0
        assert_eq!(display_str, "0");
    }

    // Rating calculation tests
    #[test]
    fn duel_win() {
        let rater = Rater::default();
        let p1 = Rating::default();
        let p2 = Rating::default();

        let (new_p1, new_p2) = rater.duel(p1, p2, Outcome::Win);

        assert!(new_p1.mu > p1.mu);
        assert!(new_p2.mu < p2.mu);
        assert!(new_p1.sigma < p1.sigma);
        assert!(new_p2.sigma < p2.sigma);
    }

    #[test]
    fn duel_loss() {
        let rater = Rater::default();
        let p1 = Rating::default();
        let p2 = Rating::default();

        let (new_p1, new_p2) = rater.duel(p1, p2, Outcome::Loss);

        assert!(new_p1.mu < p1.mu);
        assert!(new_p2.mu > p2.mu);
        assert!(new_p1.sigma < p1.sigma);
        assert!(new_p2.sigma < p2.sigma);
    }

    #[test]
    fn duel_tie() {
        let p1 = Rating::default();
        let p2 = Rating::default();

        let rater = Rater::default();
        let (new_p1, new_p2) = rater.duel(p1, p2, Outcome::Draw);

        assert_eq!(new_p1.mu, 25.0);
        assert_eq!(new_p2.mu, 25.0);
        assert!((new_p1.sigma - 8.0655063).abs() < 1.0 / 1000000.0);
        assert!((new_p2.sigma - 8.0655063).abs() < 1.0 / 1000000.0);
    }

    #[test]
    fn two_player_duel_win_loss() {
        let p1 = Rating::default();
        let p2 = Rating::default();

        let rater = Rater::default();
        let team1 = [p1];
        let team2 = [p2];
        let teams = [&team1[..], &team2[..]];
        let new_rs = rater.update_ratings(&teams, [0, 1]).unwrap();

        assert!((new_rs[0][0].mu - 27.63523138).abs() < 1.0 / 100000000.0);
        assert!((new_rs[0][0].sigma - 8.0655063).abs() < 1.0 / 1000000.0);
        assert!((new_rs[1][0].mu - 22.36476861).abs() < 1.0 / 100000000.0);
        assert!((new_rs[1][0].sigma - 8.0655063).abs() < 1.0 / 1000000.0);
    }

    #[test]
    fn four_player_race() {
        let p1 = Rating::default();
        let p2 = Rating::default();
        let p3 = Rating::default();
        let p4 = Rating::default();

        let rater = Rater::default();
        let team1 = [p1];
        let team2 = [p2];
        let team3 = [p3];
        let team4 = [p4];
        let teams = [&team1[..], &team2[..], &team3[..], &team4[..]];
        let ranks = vec![1, 2, 3, 4];

        let new_ratings = rater.update_ratings(&teams, ranks).unwrap();

        assert!((new_ratings[0][0].mu - 32.9056941).abs() < 1.0 / 10000000.0);
        assert!((new_ratings[1][0].mu - 27.6352313).abs() < 1.0 / 10000000.0);
        assert!((new_ratings[2][0].mu - 22.3647686).abs() < 1.0 / 10000000.0);
        assert!((new_ratings[3][0].mu - 17.0943058).abs() < 1.0 / 10000000.0);

        assert!((new_ratings[0][0].sigma - 7.50121906).abs() < 1.0 / 1000000.0);
        assert!((new_ratings[1][0].sigma - 7.50121906).abs() < 1.0 / 1000000.0);
        assert!((new_ratings[2][0].sigma - 7.50121906).abs() < 1.0 / 1000000.0);
        assert!((new_ratings[3][0].sigma - 7.50121906).abs() < 1.0 / 1000000.0);
    }

    #[test]
    fn uneven_teams() {
        let p1 = Rating::default();
        let p2 = Rating::default();
        let p3 = Rating::default();

        let rater = Rater::default();
        let team1 = [p1];
        let team2 = [p2, p3];
        let teams = [&team1[..], &team2[..]];
        let ranks = [1, 2];

        let new_ratings = rater.update_ratings(&teams, ranks).unwrap();

        assert!(new_ratings[0][0].mu > new_ratings[1][0].mu);
        assert!(new_ratings[0][0].mu > new_ratings[1][1].mu);
        assert!(new_ratings[1][0].mu == new_ratings[1][1].mu);
    }

    // Error handling tests
    #[test]
    fn update_ratings_mismatched_lengths() {
        let rater = Rater::default();
        let team1 = [Rating::default()];
        let team2 = [Rating::default()];
        let teams = [&team1[..], &team2[..]];
        let ranks = [1, 2, 3]; // Wrong length

        let result = rater.update_ratings(&teams, ranks);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), BBTError::MismatchedVectorLengths);
    }

    #[test]
    fn update_ratings_empty_team() {
        let rater = Rater::default();
        let team1 = [Rating::default()];
        let team2: [Rating; 0] = []; // Empty team
        let teams = [&team1[..], &team2[..]];
        let ranks = [1, 2];

        let result = rater.update_ratings(&teams, ranks);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), BBTError::EmptyTeam);
    }

    #[test]
    fn update_ratings_empty_input() {
        let rater = Rater::default();
        let teams: &[&[Rating]] = &[];
        let ranks: &[usize] = &[];

        let result = rater.update_ratings(teams, ranks);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    // Edge cases and boundary conditions
    #[test]
    fn update_ratings_single_team() {
        let rater = Rater::default();
        let original_rating = Rating::default();
        let team1 = [original_rating];
        let teams = [&team1[..]];
        let ranks = vec![1];

        let result = rater.update_ratings(&teams, ranks).unwrap();
        // With only one team, rating should remain unchanged
        assert_eq!(result[0][0].mu, original_rating.mu);
        assert_eq!(result[0][0].sigma, original_rating.sigma);
    }

    // Conversion from Vec<Vec<Rating>> to slices
    #[test]
    fn vec_to_slice_conversion() {
        let rater = Rater::default();
        let p1 = Rating::default();
        let p2 = Rating::default();
        let p3 = Rating::default();
        let p4 = Rating::default();

        // Start with Vec<Vec<Rating>> (old API style)
        let teams_vec = vec![vec![p1], vec![p2, p3], vec![p4]];
        let ranks = [1, 2, 3];

        // Convert to slice of slices for the new API
        let teams_slices: Vec<&[Rating]> = teams_vec.iter().map(|team| team.as_slice()).collect();
        let result = rater.update_ratings(&teams_slices, ranks).unwrap();

        // Verify the results are correct
        assert!(result[0][0].mu > result[1][0].mu); // Team 1 beat team 2
        assert!(result[1][0].mu > result[2][0].mu); // Team 2 beat team 3
        assert_eq!(result[1][0].mu, result[1][1].mu); // Team members have equal ratings
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 1);
        assert_eq!(result[1].len(), 2);
        assert_eq!(result[2].len(), 1);
    }

    // Error enum tests
    #[test]
    fn error_messages_are_accessible() {
        let mismatch_error = BBTError::MismatchedVectorLengths;
        let empty_team_error = BBTError::EmptyTeam;

        assert_eq!(
            "`teams` and `ranks` must be of the same length",
            mismatch_error.to_string()
        );

        assert_eq!(
            "At least one of the teams contains no players",
            empty_team_error.to_string()
        );
    }
}
