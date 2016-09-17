# BBT

BBT is an implementation of a skill-rating system similar to Elo, Glicko or
TrueSkill. It follows `Algorithm 1` from the paper
[A Bayesian Approximation Method for Online Ranking][ABAMOR].

[ABAMOR]: http://jmlr.csail.mit.edu/papers/volume12/weng11a/weng11a.pdf

## Usage

As a first step, you need to instantiate a Rater:

```rust
let rater = bbt::Rater::new(25.0/6.0);
```

The new() function takes one parameter, ꞵ. This parameter describes how much
randomness (variance in outcomes) your game has. For example, a game like
Hearthstone is much more luck-based than chess and should have a higher
variance; you may need to experiment to see which value has the highest
predictive power.

### Two-player games (e.g. Chess)

BBT has a convenience function for two-player games that returns the new ratings
for the two players after a game. In the example, p1 wins against p2:

```rust
let p1 = bbt::Rating::default();
let p2 = bbt::Rating::default();
let (new_p1, new_p2) = rater::duel(p1, p2, bbt::Outcome::Win);
```

The `bbt::Outcome` enum can take on the values `Win`, `Loss` and `Tie`.

### Multiplayer games

Games with more than two players will have to use the general `update_ratings`
method. It takes a vector of teams and a vector of ranks, with each team being a
vector of player ratings. If no error occurs, the method returns a vector of the
same form as the input with updated ratings.

#### Example 1: Racing Game

In a racing game without teams, each player is represented as a "team" of one,
and since there are usually no ties in a racing game, the list of ranks contains
no duplicates:

```rust
let p1 = bbt::Rating::default();
// ...
let p6 = bbt::Rating::default();

let new_ratings = rater::update_ratings(vec![vec![p1], vec![p2], vec![p3],
                                             vec![p4], vec![p5], vec![p6]],
                                        vec![1, 2, 3, 4, 5, 6]);
```

In the example, the first player places first, the second player second, and so
on.

#### Example 2: Tied Teams

Let's say you have a hypothetical game with four teams and two players per team.

| Team 1 | Team 2  | Team 3 | Team 4 |
| ------ | ------- | ------ | ------ |
| Alice  | Charlie | Eve    | Gabe   |
| Bob    | Dave    | Fred   | Henry  |

If Team 1 wins, and Team 2 and 3 draw for second place and Team 4 loses, you can
call the `update_ratings` function as follows:

```rust
rater = bbt::Rater::default();

alice = bbt::Rating::default();
bob   = bbt::Rating::default();
// ...
henry = bbt::Rating::default();

rater::update_ratings(vec![vec![alice, bob],
                           vec![charlie, dave],
                           vec![eve, fred],
                           vec![gabe, henry]],
                      vec![1, 2, 2, 4]);
```

The second vector assigns a rank to the teams given in the first vector. Team 1
placed first, teams 2 and 3 tie for second place and team 4 comes in fourth.

## Rating scale

The default rating scale follows TrueSkill's convention of ranks from 0 to 50.
You should be able to use a different scale by specifying the middle of that
scale in `Rating::new()`. For example, to use a more traditional scale of 0 to
3000, you can initialize ratings with `Rating::new(1500.0, 1500.0/3.0)`. You'll
also need to adjust the ϐ-value of the Rater instance accordingly:
`Rater::new(1500.0/6.0)`.
