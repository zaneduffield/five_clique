use std::io::BufRead;

use itertools::Itertools;
use rand::{seq::IteratorRandom, thread_rng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

const WLEN: usize = 5;
const SLEN: usize = 5;

type Word = [u8; WLEN];

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord)]
struct LowerAsciiCharset(u32);
impl From<Word> for LowerAsciiCharset {
    fn from(w: Word) -> Self {
        let mut chars = 0;
        w.iter().for_each(|b| chars |= 1 << (b - b'a'));
        Self(chars)
    }
}

impl LowerAsciiCharset {
    fn default() -> Self {
        LowerAsciiCharset(0)
    }

    fn intersects(&self, w: Word) -> bool {
        w.iter().any(|b| self.0 & (1 << (b - b'a')) != 0)
    }

    fn add(&mut self, b: u8) {
        self.0 |= 1 << (b - b'a');
    }
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord)]
struct Sentence<const N: usize> {
    words: [u8; N],
    len: u8,
    charset: LowerAsciiCharset,
}

impl<const N: usize> Sentence<N> {
    fn new() -> Sentence<N> {
        Sentence {
            len: 0,
            words: [0; N],
            charset: LowerAsciiCharset::default(),
        }
    }

    fn add(&mut self, w: Word) -> &Self {
        let i = (self.len as usize) * WLEN;
        self.words[i..i + WLEN].copy_from_slice(&w);
        self.len += 1;

        w.iter().for_each(|b| self.charset.add(*b));
        self
    }

    fn shares_chars_with(&self, w: Word) -> bool {
        self.charset.intersects(w)
    }

    fn last(&self) -> Word {
        let i = ((self.len - 1) as usize) * WLEN;
        self.words[i..i + WLEN].try_into().unwrap()
    }

    fn as_string(&self) -> String {
        self.words.chunks_exact(WLEN).map(show).join(" ")
    }
}

fn show(bs: &[u8]) -> String {
    String::from_utf8_lossy(bs).into_owned()
}

fn show_words(ws: &[Word]) -> String {
    ws.iter().map(|w| show(w)).join(" ")
}

fn main() {
    let words = words();
    let graph = build_graph(words);
    eprintln!("Done generating map!");

    let mut sols: Vec<Sentence<{ SLEN * WLEN }>> = graph
        .par_iter()
        .flat_map(|(w, nbs)| {
            let mut sols = vec![];
            let mut init = Sentence::<{ SLEN * WLEN }>::new();
            init.add(*w);
            find(&mut sols, &graph, init, *w, nbs);
            sols
        })
        .collect();

    sols.sort();
    for sol in sols {
        println!("{}", sol.as_string());
    }
}

fn find<const N: usize>(
    sols: &mut Vec<Sentence<N>>,
    graph: &WordGraph,
    cur_sol: Sentence<N>,
    last: Word,
    nbs: &[Word],
) {
    for nb in nbs.iter().skip_while(|nb| nb <= &&last) {
        if cur_sol.shares_chars_with(*nb) {
            continue;
        }
        let mut sol = cur_sol.clone();
        sol.add(*nb);
        if sol.len as usize >= N / WLEN {
            sols.push(sol);
        } else {
            find(
                sols,
                graph,
                sol,
                *nb,
                graph.get(nb).expect("word not found in graph"),
            );
        }
    }
}

type WordGraph = FxHashMap<Word, Vec<Word>>;
fn build_graph(words: Vec<Word>) -> WordGraph {
    words
        .par_iter()
        .map(|w| {
            let charset = LowerAsciiCharset::from(*w);
            (
                *w,
                words
                    .iter()
                    .copied()
                    .filter(|w2| !charset.intersects(*w2))
                    .sorted()
                    .collect(),
            )
        })
        .collect()
}

fn n_random(n: usize, words: &[Word]) -> Vec<Word> {
    let mut rng = thread_rng();
    words.iter().copied().choose_multiple(&mut rng, n)
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
