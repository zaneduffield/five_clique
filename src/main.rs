use std::io::BufRead;

use itertools::Itertools;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

const WLEN: usize = 5;
const SLEN: usize = 5;

type Word = [u8; WLEN];

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
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

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
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

    fn add(&mut self, w: Word) {
        let i = (self.len as usize) * WLEN;
        self.words[i..i + WLEN].copy_from_slice(&w);
        self.len += 1;

        w.iter().for_each(|b| self.charset.add(*b));
    }

    fn shares_chars_with(&self, w: Word) -> bool {
        self.charset.intersects(w)
    }

    fn words(&self) -> Vec<Word> {
        self.words
            .chunks_exact(WLEN)
            .map(|w| w.try_into().unwrap())
            .collect()
    }

    fn as_string(&self) -> String {
        self.words().iter().map(|w| show(w)).join(" ")
    }
}

fn show(bs: &[u8]) -> String {
    String::from_utf8_lossy(bs).into_owned()
}

fn main() {
    let anagrams = anagram_groups();
    let anagram_map = anagram_map(&anagrams);
    let anagram_reps = anagram_map.keys().copied().collect_vec();
    let graph = build_graph(anagram_reps);
    eprintln!("Done generating map!");

    let mut sols: Vec<Sentence<{ SLEN * WLEN }>> = graph
        .par_iter()
        .flat_map(|(w, nbs)| {
            let mut sols = vec![];
            let mut init = Sentence::<{ SLEN * WLEN }>::new();
            init.add(*w);
            find_sols(&mut sols, &graph, init, *w, nbs);
            sols
        })
        .collect();

    let mut ana_sols = vec![];
    for sol in &sols {
        expand_anagrams(&mut ana_sols, &anagram_map, *sol);
    }
    sols.extend(ana_sols);

    for sol in sols.iter().sorted() {
        println!("{}", sol.as_string());
    }
}

fn expand_anagrams<const N: usize>(
    sols: &mut Vec<Sentence<N>>,
    anagram_map: &FxHashMap<Word, Vec<Word>>,
    sol: Sentence<N>,
) {
    let mut a_idxs = vec![0; sol.len.into()];
    let agrams = sol
        .words()
        .iter()
        .map(|w| anagram_map.get(w).unwrap())
        .collect_vec();

    loop {
        let mut i = 0;
        a_idxs[0] += 1;
        while a_idxs[i] >= agrams[i].len() {
            a_idxs[i] = 0;
            i += 1;
            if i >= a_idxs.len() {
                return;
            }
            a_idxs[i] += 1;
        }

        let mut sentence = Sentence::<N>::new();
        a_idxs
            .iter()
            .enumerate()
            .map(|(i, idx)| agrams[i][*idx])
            .sorted()
            .for_each(|w| sentence.add(w));

        sols.push(sentence);
    }
}

fn find_sols<const N: usize>(
    sols: &mut Vec<Sentence<N>>,
    graph: &WordGraph,
    cur_sol: Sentence<N>,
    last: Word,
    nbs: &[Word],
) {
    let pos = match nbs.binary_search(&last) {
        Ok(_) => nbs.len(),
        Err(i) => i,
    };

    for nb in &nbs[pos..] {
        if cur_sol.shares_chars_with(*nb) {
            continue;
        }
        let mut sol = cur_sol;
        sol.add(*nb);
        if sol.len as usize >= N / WLEN {
            sols.push(sol);
        } else {
            find_sols(
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

fn anagram_groups() -> Vec<Vec<Word>> {
    include_bytes!("words_five.txt")
        .lines()
        .map(|line| {
            line.expect("failed to read line")
                .as_bytes()
                .try_into()
                .expect("line is not the right length in bytes")
        })
        .filter(|w| distinct_letters(*w))
        .sorted_unstable_by_key(|w| word_chars_sorted(*w))
        .group_by(|w| word_chars_sorted(*w))
        .into_iter()
        .map(|(_, group)| group.collect())
        .collect()
}

fn anagram_map(anagram_groups: &[Vec<Word>]) -> FxHashMap<Word, Vec<Word>> {
    let mut out = FxHashMap::default();
    for group in anagram_groups {
        out.insert(group[0], group.to_vec());
    }
    out
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
