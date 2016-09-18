[![Build Status](https://travis-ci.org/DataWraith/bbt.svg?branch=master)](https://travis-ci.org/DataWraith/bbt)

# BBT

BBT is an implementation of a skill-rating system similar to Elo, Glicko or
TrueSkill. It follows `Algorithm 1` from the paper
[A Bayesian Approximation Method for Online Ranking][ABAMOR].

[ABAMOR]: http://jmlr.csail.mit.edu/papers/volume12/weng11a/weng11a.pdf

## Usage

Add BBT to your Cargo.toml:

```toml
[dependencies]
bbt = "0.1.0"
```

See the [Documentation](https://datawraith.github.io/bbt/doc/bbt/index.html) for
information on how to use the crate.
