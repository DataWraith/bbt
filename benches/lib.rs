#![feature(test)]

extern crate bbt;
extern crate test;

#[cfg(test)]
mod beench {
    #[bench]
    fn hundred_duels(b: &mut test::Bencher) {
        let mut player_one = bbt::Rating::default();
        let mut player_two = bbt::Rating::default();
        let mut rater = bbt::Rater::default();

        b.iter(|| {
            for _ in 0..100 {
                rater.duel(player_one.clone(), player_two.clone(), bbt::Outcome::Win);
            }
        });
    }
}