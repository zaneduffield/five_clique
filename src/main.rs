use std::io::BufRead;

use itertools::Itertools;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

const WLEN: usize = 5;
const SLEN: usize = 5;

type Word = [u8; WLEN];
struct WordWithCharset {
    word: Word,
    charset: LowerAsciiCharset,
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
struct LowerAsciiCharset(u32);
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

    fn add(&mut self, b: u8) {
        self.0 |= 1 << b;
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

    fn shares_chars_with(&self, c: LowerAsciiCharset) -> bool {
        self.charset.intersects(c)
    }

    fn words(&self) -> Vec<Word> {
        self.words
            .chunks_exact(WLEN)
            .map(|w| w.try_into().unwrap())
            .collect()
    }

    fn as_string(&self) -> String {
        self.words().into_iter().map(show).join(" ")
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

fn show(mut w: Word) -> String {
    w.iter_mut().for_each(|b| *b += b'a');
    String::from_utf8_lossy(&w).into_owned()
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
    sols: &mut Vec<Sentence<N>>,
    graph: &WordGraph,
    cur_sol: Sentence<N>,
    last: Word,
    nbs: &[WordWithCharset],
) {
    let pos = nbs.partition_point(|nb| nb.word[0] < last[0]);
    for nb in &nbs[pos..] {
        if cur_sol.shares_chars_with(nb.charset) {
            continue;
        }
        let mut sol = cur_sol;
        sol.add(nb.word);
        if sol.len >= (N / WLEN) as u8 {
            sols.push(sol);
        } else {
            find_sols(
                sols,
                graph,
                sol,
                nb.word,
                graph.get(&nb.word).expect("word not found in graph"),
            );
        }
    }
}

type WordGraph = FxHashMap<Word, Vec<WordWithCharset>>;
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
                    .map(|w| WordWithCharset {
                        word: w,
                        charset: LowerAsciiCharset::from(w),
                    })
                    .filter(|w2| !charset.intersects(w2.charset))
                    .sorted_by_key(|w| w.word)
                    .collect(),
            )
        })
        .collect()
}

fn anagram_groups() -> Vec<Vec<Word>> {
    include_bytes!("words_five.txt")
        .lines()
        .map(|line| {
            let mut w: Word = line
                .expect("failed to read line")
                .as_bytes()
                .try_into()
                .expect("line is not the right length in bytes");
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
    let sols: Vec<Sentence<{ SLEN * WLEN }>> = graph
        .par_iter()
        .flat_map(|(w, nbs)| {
            let mut sols = vec![];
            let mut init = Sentence::<{ SLEN * WLEN }>::new();
            init.add(*w);
            find_sols(&mut sols, &graph, init, *w, nbs);
            sols
        })
        .collect();
    eprintln!(" done!");

    eprint!("Expanding anagram solutions...");
    let mut sols_with_agrams = vec![];
    for sol in sols {
        expand_anagrams(&mut sols_with_agrams, &anagram_map, sol);
    }
    eprintln!(" done!");

    for sol in sols_with_agrams.iter().sorted() {
        println!("{}", sol.as_string());
    }
}
