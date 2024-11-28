use burn::prelude::*;
use core::fmt;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    convert::Into,
    io::BufRead,
};

pub fn parse_input(reader: impl BufRead) -> Instance<B> {
    let lines = reader.lines().skip(1);
    let strings = lines
        .map(|l| l.unwrap().bytes().map(|b| b - b'a').collect())
        // .take(10)
        .collect();
    Instance {
        strings,
        ..Default::default()
    }
}

#[derive(Default, Clone)]
pub struct Instance<B: Backend> {
    pub strings: Vec<Guess>,
    tensor: Option<Tensor<B, 2, Int>>,
    device: B::Device,
}

impl<B: Backend> fmt::Debug for Instance<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for line in &self.strings {
            for &byte in line {
                write!(f, "{}", (byte + b'a') as char)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

trait Distance<'a>: Iterator<Item = &'a u8> + Sized {
    fn distance(self, other: impl Iterator<Item = &'a u8>) -> usize {
        self.zip(other).filter(|(a, b)| a != b).count()
    }
    fn diff_vec(self, other: impl Iterator<Item = &'a u8>) -> Vec<bool> {
        self.zip(other).map(|(a, b)| a != b).collect()
    }
}

impl<'a, T: Iterator<Item = &'a u8> + Sized> Distance<'a> for T {}

type Guess = Vec<u8>;

impl<B: Backend> Instance<B> {
    pub fn dedup(&mut self) {
        self.strings.sort();
        self.strings.dedup();
    }

    // filter out characters which are the same for all strings
    pub fn shorten(&mut self) -> Vec<bool> {
        let reference = self.strings[0].clone();
        let keep = self.compute_total_differences(&reference);

        for line in &mut self.strings {
            let char_iter = line.iter().zip(&keep);
            *line = char_iter.filter_map(|(&c, e)| e.then_some(c)).collect();
        }

        keep
    }

    fn compute_total_differences(&self, reference: &[u8]) -> Vec<bool> {
        let mut keep = vec![false; reference.len()];

        for line in &self.strings {
            for i in 0..line.len() {
                keep[i] |= line[i] != reference[i];
            }
        }
        keep
    }

    pub fn populate_tensor(&mut self) {
        let concat: Vec<i32> = self.strings.concat().into_iter().map(Into::into).collect();
        let one_d_tensor: Tensor<B, 1, _> = Tensor::from_ints(concat.as_slice(), &self.device);
        let words_tensor = one_d_tensor.reshape([self.strings.len(), self.strings[0].len()]);
        // println!("{:.2}", words_tensor);
        self.tensor = Some(words_tensor);
    }

    pub fn distance_to(&mut self, guess: &[i32]) -> u32 {
        let word: Tensor<B, 1, _> = Tensor::from_ints(guess, &self.device);
        if self.tensor.is_none() {
            self.populate_tensor();
        }
        let tensor = self.tensor.clone().unwrap();
        let words = word.expand(tensor.shape());
        let equals = tensor.not_equal(words);
        let sums = equals.int().sum_dim(1);
        let max = sums.max();
        let max = max.into_scalar();
        let max: i32 = max.to_i32();
        max as u32
    }
    pub fn distances_to(&self, guesses: &[&[i32]]) -> Vec<i32> {
        let guess = guesses.concat();
        // dbg!(&guesses[0].len());
        // dbg!(&self.strings[0].len());
        let word: Tensor<B, 1, _> = Tensor::from_ints(guess.as_slice(), &self.device);
        let word = word.reshape([guesses.len() as i32, 1, -1]);
        // let word = word.expand::<3>();
        // let word = word.reshape([guesses[0].len() as i32, 1, -1]);

        if self.tensor.is_none() {
            panic!("tensors not populated");
            // self.populate_tensor();
        }
        let tensor = self.tensor.clone().unwrap();
        let tensor = tensor.unsqueeze::<3>();
        let tensor = tensor.expand([guesses.len(), self.strings.len(), guesses[0].len()]);
        // let word = word.transpose();
        let words = word.expand(tensor.shape());
        let equals = tensor.not_equal(words);
        let sums = equals.int().sum_dim(2);
        let sums = sums.squeeze::<2>(2);
        // eprintln!("sums: {}", &sums);
        let max = sums.max_dim(1);
        let max_values = max.to_data();
        // let max = max.into_scalar();
        // let max: i32 = max.to_i32();
        let values: Vec<i32> = max_values.to_vec().unwrap();
        let max_values = values.into_iter().map(|x| x as i32).collect();
        max_values
        // max_values.to_vec().unwrap()
    }

    pub fn restrict_instance_to_d_core(&self, d: usize) -> (Instance<B>, Vec<bool>) {
        let pairs = self.find_pairs_with_d(d);
        let indices: HashSet<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        let mut new_instance = Instance {
            strings: self
                .strings
                .iter()
                .enumerate()
                .filter(|(i, _)| indices.contains(i))
                .map(|(_, x)| x.clone())
                .collect(),
            device: self.device.clone(),
            ..Default::default()
        };
        let keep = new_instance.shorten();
        new_instance.dedup();
        new_instance.populate_tensor();
        (new_instance, keep)
    }
    pub fn solve(&self) -> Guess {
        let guess = self.initial_guess();
        let (distance, _) = self.max_distance_to(&guess);
        let mut stack = BinaryHeap::new();
        stack.push(Reverse((distance, guess.clone(), vec![false; guess.len()])));
        let mut min = distance;
        let mut best_guess = guess.clone();
        let min_possible_distance = (self.find_max_pairwise_d() + 1) / 2;
        let mut iterations = 0u128;
        // let mut delay = Vec::new();

        loop {
            let mut mutations: Vec<(Vec<i32>, Vec<bool>, usize, usize)> = Vec::new();

            let mut enqueue = |guess: Vec<u8>,
                               fixed: Vec<bool>,
                               stack: &BinaryHeap<_>,
                               mutations: &mut Vec<_>| {
                let free_positions = fixed.iter().filter(|&&x| !x).count();

                iterations += 1;
                if iterations % 10000 == 0 {
                    dbg!(stack.len(), free_positions);
                }
                // dbg!(d);
                let (d, worst) = self.max_distance_to(&guess);
                if d + 1 >= min + free_positions {
                    return;
                }
                // eprintln!("guess: {:?}", guess);
                // eprintln!("worst: {:?}", worst);
                // eprintln!("{:?}", worst);
                let mut diff_vec = guess.iter().diff_vec(worst.iter());
                diff_vec.iter_mut().zip(&fixed).for_each(|(a, b)| *a &= !b);
                if d < min {
                    dbg!(d);
                    println!("guess: {:?}", guess);
                    min = d;
                    best_guess = guess.clone();
                }

                // let mut indices = Vec::new();
                while let Some(pos) = diff_vec.iter().position(|&x| x) {
                    diff_vec[pos] = false;
                    let mut new_guess = guess.clone();
                    new_guess[pos] = worst[pos];
                    let mut new_fixed = fixed.clone();
                    new_fixed[pos] = true;
                    // indices.push(new_fixed);
                    mutations.push((
                        new_guess.iter().map(|&x| x as i32).collect(),
                        new_fixed,
                        free_positions,
                        d,
                    ));
                }
            };

            if stack.len() > 4000 {
                break;
            }

            while mutations.len() < 32 && !stack.is_empty() {
                let Some(Reverse((_, guess, fixed))) = stack.pop() else {
                    continue;
                };
                enqueue(guess, fixed, &stack, &mut mutations);
            }

            if min <= min_possible_distance {
                break;
            }
            let slices: Vec<_> = mutations.iter().map(|(x, _, _, _)| x.as_slice()).collect();
            if mutations.is_empty() {
                continue;
            }
            let distances = self.distances_to(&slices);
            // eprintln!("distances: {:?}", distances);
            let results = mutations.iter().zip(distances);
            // dbg!(&results);
            for ((guess, fixed, free_positions, d), new_d) in results {
                if new_d > *d as i32 || new_d as usize + 1 >= min + free_positions {
                    continue;
                }
                stack.push(Reverse((
                    (new_d as usize) * 100 - ((*free_positions as f32).sqrt() * 20.) as usize,
                    guess.iter().map(|&x| x as u8).collect(),
                    fixed.clone(),
                )))
            }
        }
        best_guess
    }
    pub fn initial_guess(&self) -> Guess {
        let guess = self.strings[0].clone();
        let d = self.find_max_pairwise_d();
        let (start, end) = self.find_pairs_with_d(d)[0];

        [
            &self.strings[start][0..guess.len() / 2],
            &self.strings[end][(guess.len() / 2)..],
        ]
        .concat()
    }

    fn max_distance_to(&self, guess: &[u8]) -> (usize, Guess) {
        let mut max = 0;
        let mut max_s = vec![];
        for s in &self.strings {
            let distance = guess.iter().distance(s.iter());
            if distance > max {
                max = distance;
                max_s = s.clone();
            }
        }

        (max, max_s)
    }

    pub fn find_max_pairwise_d(&self) -> usize {
        let mut d = 0;

        for i in 0..self.strings.len() {
            for b in &self.strings[i..] {
                let a = &self.strings[i];
                let distance = a.iter().distance(b.iter());
                d = d.max(distance);
            }
        }
        d
    }
    pub fn find_pairs_with_d(&self, d: usize) -> Vec<(usize, usize)> {
        let mut pairs = vec![];

        for i in 0..self.strings.len() {
            for j in i..self.strings.len() {
                let a = &self.strings[i];
                let b = &self.strings[j];
                let distance = a.iter().distance(b.iter());
                if distance == d {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }

    // compute pair wise distances and choose max pairwise distance as a start
}

use burn::tensor::{cast::ToElement, Tensor};

type B = burn::backend::Wgpu;

#[cfg(test)]
mod test {
    use super::*;
    use std::io::BufReader;

    #[test]
    fn test_instance0() {
        test_instance("string0.in");
    }
    #[test]
    fn test_instance1() {
        test_instance("string1.in");
    }
    #[test]
    fn test_instance2() {
        test_instance("string2.in");
    }
    #[test]
    fn test_instance3() {
        test_instance("string3.in");
    }
    #[test]
    fn test_instance4() {
        test_instance("string4.in");
    }

    fn test_instance(file: &str) -> u32 {
        let instance = parse_input(BufReader::new(std::fs::File::open(file).unwrap()));
        let cover = instance.compute_s();
        dbg!(cover.vertecies.len() as u32);
        assert!(instance.validate_cover(&cover));
        cover.vertecies.len() as u32
    }
}
