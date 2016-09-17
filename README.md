# BBT

BBT is an implementation of a skill-rating system similar to Elo, Glicko or
TrueSkill. It follows `Algorithm 1` from the paper
[A Bayesian Approximation Method for Online Ranking][ABAMOR].

[ABAMOR]: http://jmlr.csail.mit.edu/papers/volume12/weng11a/weng11a.pdf

## Usage

As a first step, you need to instantiate a Rater:

```rust
let rater = bbt::Rater::new(25.0/6)
```

The new() function takes one parameter, ꞵ. This parameter describes how much
randomness (variance in outcomes) your game has. For example, a game like
Hearthstone is much more luck-based than chess and should have a higher
variance, you may need to experiment to see which value has the highest
predictive power.

### Two-player games (e.g. Chess)

TODO

### Multiplayer games without teams (e.g. Racing games)

#### Without draws

TODO

#### With draws

TODO

### Multiplayer games with teams

#### Without draws

TODO

#### With draws

Let's say you have a hypothetical game with four teams and two players per team.

| Team 1 | Team 2  | Team 3 | Team 4 |
| ------ | ------- | ------ | ------ |
| Alice  | Charlie | Eve    | Gabe   |
| Bob    | Dave    | Fred   | Henry  |

If Team 1 wins, and Team 2 and 3 draw for second place and Team 4 loses, you can
call the `update_ratings` function as follows:

```rust
rater = bbt::Rater::new(25.0/6)

alice = bbt::Rating::default()
bob   = bbt::Rating::default()
// ...
henry = bbt::Rating::default()

rater::update_ratings(vec![vec![alice, bob],
                           vec![charlie, dave],
                           vec![eve, fred],
                           vec![gabe, henry]],
                      vec![1, 2, 2, 4]);
)
```

The second vector assigns a rank to the teams given in the first vector. Team 1
placed first, teams 2 and 3 second and team 4 fourth.
