use std::fmt::Debug;

use libafl::{
    corpus::{CorpusId},
    fuzzer::ExecuteInputResult,
    bolts::rands::{Rand, StdRand},
};

// id, iter_id, iter_count, mut_sz, parent_dist, eval_result, mut_dist, target
// Фильтер получает эти признаки до запуска и принимает решение о допуске
#[derive(Clone, Debug)]
pub struct MutantFeatures {
    pub id: CorpusId,
    pub iter_id: u64,
    pub iter_count: u64,
    pub mut_sz: usize,
    pub parent_dist: f64
}

// Фильтр получает эти признаки после запуска и дообучает модель
#[derive(Debug)]
pub struct MutantOutcome {
    pub eval_result: ExecuteInputResult,
    pub mut_dist: f64,
    pub target: u64   
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decision {
    Execute,
    Discard,
    ExecuteDespite
}

pub trait FilterTrait: Debug {
    // Принимаем решение о запуске мутанта на основе его признаков
    fn make_decision(&mut self, features: &MutantFeatures) -> Decision;
    
    // Дообучаем модель на основе выхода мутанта
    fn make_observation(&mut self, _features: &MutantFeatures, _outcome: &MutantOutcome) {}
}

#[derive(Debug)]
pub struct PassAll;

// Пропускает все (Тестовый фильтер)
impl FilterTrait for PassAll {
    fn make_decision(&mut self, _features: &MutantFeatures) -> Decision {
        Decision::Execute
    }
}

#[derive(Debug)]
pub struct RandomDrop {
    p: f64,
    rng: StdRand,
}

impl RandomDrop {
    pub fn new(p: f64, seed: u64) -> Self {
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
        
        Self {p, rng: StdRand::with_seed(seed)}
    }
}

// Отбрасывает вход с вероятностью p (тестовый фильтр).
impl FilterTrait for RandomDrop {
    fn make_decision(&mut self, _features: &MutantFeatures) -> Decision {
        let val = self.rng.below(1_000_000);
        let threshold = (self.p * 1_000_000.0) as u64;

        if val < threshold {
            Decision::Discard
        } else {
            Decision::Execute
        }
    }
}
