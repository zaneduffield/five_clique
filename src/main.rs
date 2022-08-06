use std::io::BufRead;

use itertools::Itertools;

const LEN: usize = 5;
const SLEN: usize = 5;
type Word = [u8; LEN];

#[derive(Clone)]
struct Sentence {
    len: u8,
    words: [u8; LEN * SLEN],
    chars: u32,
}

impl Sentence {
    fn new() -> Sentence {
        Sentence {
            len: 0,
            words: [0; LEN * SLEN],
            chars: 0,
        }
    }

    fn add(&mut self, w: Word) -> &Self {
        let i = (self.len as usize) * LEN;
        self.words[i..i + LEN].copy_from_slice(&w);
        self.len += 1;

        w.iter().for_each(|b| self.chars |= 1 << (b - b'a'));
        self
    }

    fn shares_chars_with(&self, w: Word) -> bool {
        w.iter().any(|b| self.chars & (1 << (b - b'a')) != 0)
    }

    fn current(&self) -> &[u8] {
        &self.words[..(self.len as usize) * LEN]
    }
}

fn show(bs: &[u8]) -> String {
    String::from_utf8_lossy(bs).into_owned()
}

fn main() {
    let words = words();
    let mut sentences = vec![];
    for word in &words {
        let mut s = Sentence::new();
        s.add(*word);
        for word in &words {
            if !s.shares_chars_with(*word) {
                let mut s = s.clone();
                s.add(*word);
                sentences.push(s);
            }
        }
    }

    println!("{} 2 word sentences found!", sentences.len());
}

fn words() -> Vec<Word> {
    include_bytes!("words_five.txt")
        .lines()
        .map(|line| {
            line.expect("failed to read line")
                .as_bytes()
                .try_into()
                .expect("line is not the right length in bytes")
        })
        .filter(|w| distinct_letters(*w))
        .unique_by(|w| word_chars_sorted(*w))
        .collect()
}

fn distinct_letters(w: Word) -> bool {
    let mut h = 0u64;
    w.iter().all(|b| {
        let x = 1 << (b - b'a');
        if x & h != 0 {
            false
        } else {
            h |= x;
            true
        }
    })
}

fn word_chars_sorted(w: Word) -> u64 {
    // concatenate the sorted bytes into a u64
    w.iter()
        .sorted()
        .enumerate()
        .fold(0u64, |out, (i, b)| out | ((*b as u64) << (i * 8)))
}
