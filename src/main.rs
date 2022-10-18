mod filter_vec;

use filter_vec::filter_vec;
use itertools::Itertools;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::io::BufRead;

const WLEN: usize = 5;
const SLEN: usize = 5;

type Word = [u8; WLEN];

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct LowerAsciiCharset(u32);
impl From<Word> for LowerAsciiCharset {
    fn from(w: Word) -> Self {
        let mut chars = 0;
        w.iter().for_each(|b| chars |= 1 << b);
        Self(chars)
    }
}

impl LowerAsciiCharset {
    fn default() -> Self {
        LowerAsciiCharset(0)
    }

    fn intersects(&self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    fn union(&mut self, other: Self) {
        self.0 |= other.0
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
struct Sentence<const N: usize> {
    words: [Option<Word>; N],
    len: u8,
}

impl<const N: usize> Sentence<N> {
    fn new() -> Sentence<N> {
        Sentence {
            len: 0,
            words: [None; N],
        }
    }

    fn add(&mut self, w: Word) {
        self.words[self.len as usize] = Some(w);
        self.len += 1;
    }

    fn as_string(&self) -> String {
        self.words.into_iter().flatten().map(show).join(" ")
    }
}

impl<I, const N: usize> From<I> for Sentence<N>
where
    I: IntoIterator<Item = Word>,
{
    fn from(words: I) -> Self {
        let mut out = Self::new();
        words.into_iter().for_each(|w| out.add(w));
        out
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
struct CharsetSentence<const N: usize> {
    words: [LowerAsciiCharset; N],
    len: u8,
    charset: LowerAsciiCharset,
}

impl<const N: usize> CharsetSentence<N> {
    fn new() -> CharsetSentence<N> {
        CharsetSentence {
            len: 0,
            words: [LowerAsciiCharset::default(); N],
            charset: LowerAsciiCharset::default(),
        }
    }

    fn add(&mut self, c: LowerAsciiCharset) {
        self.words[self.len as usize] = c;
        self.len += 1;

        self.charset.union(c);
    }
}

fn show(mut w: Word) -> String {
    w.iter_mut().for_each(|b| *b += b'a');
    String::from_utf8_lossy(&w).into_owned()
}

fn expand_anagrams<const N: usize>(
    sols: &mut Vec<Sentence<N>>,
    anagram_map: &FxHashMap<LowerAsciiCharset, Vec<Word>>,
    sol: CharsetSentence<N>,
) {
    let mut a_idxs = vec![0; sol.len.into()];
    let agrams = sol
        .words
        .iter()
        .map(|w| anagram_map.get(w).unwrap())
        .collect_vec();

    loop {
        /*
        The idea is to maintain a vec of indexes into the anagram vector, so that a_idx[i] is the index into agrams[i].
        We want to find all permutations of anagrams, each of which corresponds to a value for a_idx, so we just count
        through all the possible values for a_idx by 'ticking' the first index, and let it overflow (or carry-over)
        into the next indexes when necessary. We stop when the last index overflows.

        We could also do it with five nested for-loops, but where's the fun in that? Also, it wouldn't generalise.
        */
        let sentence: Sentence<N> = a_idxs
            .iter()
            .enumerate()
            .map(|(i, idx)| agrams[i][*idx])
            .sorted()
            .into();

        sols.push(sentence);

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
    }
}

fn find_sols<const N: usize>(
    sols: &mut Vec<CharsetSentence<N>>,
    cur_sol: CharsetSentence<N>,
    last_added: LowerAsciiCharset,
    nbs: &[LowerAsciiCharset],
) {
    let nbs = filter_vec(nbs, cur_sol.charset, last_added);
    if nbs.is_empty() {
        return;
    }

    let sols_to_explore = nbs.iter().map(|&c| {
        let mut sol = cur_sol;
        sol.add(c);
        (c, sol)
    });

    if cur_sol.len + 1 >= N as u8 {
        sols.extend(sols_to_explore.map(|(_, s)| s));
    } else {
        sols_to_explore.for_each(|(c, sol)| {
            find_sols(sols, sol, c, &nbs);
        });
    }
}

type WordGraph = FxHashMap<LowerAsciiCharset, Vec<LowerAsciiCharset>>;
fn build_graph(words: Vec<Word>) -> WordGraph {
    words
        .par_iter()
        .map(|&w| {
            let charset = LowerAsciiCharset::from(w);
            let words: Vec<LowerAsciiCharset> = words
                .iter()
                .map(|w2| LowerAsciiCharset::from(*w2))
                .filter(|c| !charset.intersects(*c))
                .collect();
            (w.into(), words)
        })
        .collect()
}

fn anagram_groups() -> Vec<Vec<Word>> {
    include_bytes!("words_alpha.txt")
        .lines()
        .flat_map(|line| line.expect("failed to read line").as_bytes().try_into())
        .map(|mut w: Word| {
            w.make_ascii_lowercase();

            // It's more efficient to shift all the characters to be based on 'a' now and then undo it right at the end.
            // Otherwise we would be doing this shift in the hottest path of the program (`Sentence<N>::shares_chars_with`)
            w.iter_mut().for_each(|b| *b -= b'a');

            w
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
        let x = 1 << b;
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

fn main() {
    eprint!("Generating anagram maps...");
    let anagrams = anagram_groups();
    let anagram_map = anagram_map(&anagrams);
    let anagram_reps = anagram_map.keys().copied().collect_vec();
    eprintln!(" done!");

    eprint!("Generating adjacency matrix...");
    let graph = build_graph(anagram_reps);
    eprintln!(" done!");

    eprint!("Finding solutions modulo anagram...");
    let sols: Vec<CharsetSentence<SLEN>> = graph
        .par_iter()
        .map(|(w, nbs)| {
            let mut sols = vec![];
            let mut init = CharsetSentence::<SLEN>::new();
            init.add(*w);
            find_sols(&mut sols, init, *w, nbs);
            sols
        })
        .flatten()
        .collect();
    eprintln!(" done!");

    eprint!("Expanding anagram solutions...");
    let mut sols_with_agrams = vec![];
    let anagram_map_by_charset: FxHashMap<LowerAsciiCharset, Vec<Word>> = anagram_map
        .into_iter()
        .map(|(k, v)| (k.into(), v))
        .collect();
    for sol in sols {
        expand_anagrams::<SLEN>(&mut sols_with_agrams, &anagram_map_by_charset, sol);
    }
    eprintln!(" done!");

    for sol in sols_with_agrams.iter().sorted() {
        println!("{}", sol.as_string());
    }
}
