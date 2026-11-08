use core::{fmt::Debug, marker::PhantomData};
use std::{fs::File, io::{BufWriter, Write}};

use crate::DistancePowerTestcaseScore;
use crate::DistanceObserver;
use crate::DistanceTestcaseMetadata;

use crate::input_filter::{FilterTrait, MutantFeatures, MutantOutcome, Decision};

use libafl::{
    corpus::{Corpus, CorpusId},
    observers::MapObserver,
    executors::{Executor, HasObservers},
    fuzzer::Evaluator,
    mutators::{MutationResult, Mutator},
    schedulers::TestcaseScore,
    stages::{mutational::MutatedTransform, mutational::MutatedTransformPost, MutationalStage, Stage},
    state::{HasClientPerfMonitor, HasCorpus, HasMetadata, HasRand, UsesState},
    inputs::HasTargetBytes,
    bolts::AsSlice,
    bolts::tuples::MatchName,
    Error,
};

/// The mutational stage using power schedules
#[derive(Debug)]
pub struct FilterPowerMutationalStage<E, F, EM, I, M, Z, O1, O2> {
    mutator: M,
    #[allow(clippy::type_complexity)]
    phantom: PhantomData<(E, F, EM, I, Z, O1, O2)>,
    buff: BufWriter<File>,
    dist_name: String,
    targets_name: String,
    filter: Box<dyn FilterTrait>
}

impl<E, F, EM, I, M, Z, O1, O2> UsesState for FilterPowerMutationalStage<E, F, EM, I, M, Z, O1, O2>
where
    E: UsesState,
{
    type State = E::State;
}

impl<E, F, EM, I, M, Z, O1, O2> MutationalStage<E, EM, I, M, Z> for FilterPowerMutationalStage<E, F, EM, I, M, Z, O1, O2>
where
    E: Executor<EM, Z> + HasObservers,
    EM: UsesState<State = E::State>,
    F: TestcaseScore<E::State>,
    M: Mutator<I, E::State>,
    E::State: HasClientPerfMonitor + HasCorpus + HasMetadata + HasRand,
    Z: Evaluator<E, EM, State = E::State>,
    I: MutatedTransform<E::Input, E::State> + Clone,
    E::Input: HasTargetBytes,
    O1: DistanceObserver<E::State>,
    O2: MapObserver,
{
    /// The mutator, added to this stage
    #[inline]
    fn mutator(&self) -> &M {
        &self.mutator
    }

    /// The list of mutators, added to this stage (as mutable ref)
    #[inline]
    fn mutator_mut(&mut self) -> &mut M {
        &mut self.mutator
    }

    /// Gets the number of iterations as a random number
    #[allow(clippy::cast_sign_loss)]
    fn iterations(&self, state: &mut E::State, corpus_idx: CorpusId) -> Result<u64, Error> {
        // Update handicap
        let mut testcase = state.corpus().get(corpus_idx)?.borrow_mut();
        let score = F::compute(state, &mut *testcase)? as u64;

        Ok(score)
    }

    // TODO Create fn perform_mutational that doing filtration
    /// Runs this (mutational) stage for the given testcase
    #[allow(clippy::cast_possible_wrap)] // more than i32 stages on 32 bit system - highly unlikely...
    fn perform_mutational(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        state: &mut Z::State,
        manager: &mut EM,
        corpus_idx: CorpusId,
    ) -> Result<(), Error> {
        let num = self.iterations(state, corpus_idx)?;
        
        let mut testcase = state.corpus().get(corpus_idx)?.borrow_mut();
        
        let test_case_metadata = testcase.metadata::<DistanceTestcaseMetadata>()?;
        let parent_dist = test_case_metadata.distance();
        
        let Ok(input) = I::try_transform_from(&mut testcase, state, corpus_idx) else { return Ok(()); };
        drop(testcase);
    
        for i in 0..num {
            let mut input = input.clone();

            let mutated = self.mutator_mut().mutate(state, &mut input, i as i32)?;

            if mutated == MutationResult::Skipped {
                continue;
            }

            // Time is measured directly the `evaluate_input` function
            let (untransformed, post) = input.try_transform_into(state)?;
            
            // Считаем размер мутанта
            let mut_sz = untransformed.target_bytes().as_slice().len();

            // MutantFeatures для принятия решение о запуске
            let mut_features = MutantFeatures{
                id: corpus_idx,
                iter_id: i,
                iter_count: num,
                mut_sz: mut_sz,
                parent_dist: parent_dist
            };

            let filter_res = self.filter.make_decision(&mut_features);

            if filter_res == Decision::Discard {
                writeln!(self.buff, "{},{},{},{},{},,,,{:?}", 
                    corpus_idx, i, num, mut_sz, parent_dist, filter_res)?;

                self.mutator_mut().post_exec(state, i as i32, None)?;
                post.post_exec(state, i as i32, None)?;

                continue;
            }

            let (eval_result, new_idx) = fuzzer.evaluate_input(state, executor, manager, untransformed)?;

            // TODO: Делаем сбор данных    
            
            // Находим обсерверы и достаем их них информацию о расстоянии и достижении цели
            let observers = executor.observers();

            let dist_observer = observers.match_name::<O1>(&self.dist_name).ok_or_else(|| {
                libafl::Error::key_not_found("DistanceObserver not found".to_string())
            })?;
            
            let targets_observer = observers.match_name::<O2>(&self.targets_name).ok_or_else(|| {
                libafl::Error::key_not_found("MapObserver not found".to_string())
            })?;
            
            
            let mut_dist = dist_observer.distance();
            let target = targets_observer.count_bytes();

            // MutantOutcome для дообучения модели
            let mut_outcome = MutantOutcome{
                eval_result: eval_result,
                mut_dist: mut_dist,
                target: target  
            };

            self.filter.make_observation(&mut_features, &mut_outcome);

            // id, iter_id, iter_count, mut_sz, parent_dist, eval_result, mut_dist, target, decision
            writeln!(self.buff, "{},{},{},{},{},{:?},{},{},{:?}", 
                corpus_idx, i, num, mut_sz, parent_dist, mut_outcome.eval_result, mut_dist, target, filter_res)?;

            self.mutator_mut().post_exec(state, i as i32, new_idx)?;
            post.post_exec(state, i as i32, new_idx)?;
        }

        self.buff.flush()?;
        Ok(())
    }
}

impl<E, F, EM, I, M, Z, O1, O2> Stage<E, EM, Z> for FilterPowerMutationalStage<E, F, EM, I, M, Z, O1, O2>
where
    E: Executor<EM, Z> + HasObservers,
    EM: UsesState<State = E::State>,
    F: TestcaseScore<E::State>,
    M: Mutator<I, E::State>,
    E::State: HasClientPerfMonitor + HasCorpus + HasMetadata + HasRand,
    Z: Evaluator<E, EM, State = E::State>,
    I: MutatedTransform<E::Input, E::State> + Clone,
    E::Input: HasTargetBytes,
    O1: DistanceObserver<E::State>,
    O2: MapObserver,
{
    #[inline]
    #[allow(clippy::let_and_return)]
    fn perform(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        state: &mut E::State,
        manager: &mut EM,
        corpus_idx: CorpusId,
    ) -> Result<(), Error> {
        let ret = self.perform_mutational(fuzzer, executor, state, manager, corpus_idx);
        ret
    }
}

impl<E, F, EM, M, Z, O1, O2> FilterPowerMutationalStage<E, F, EM, E::Input, M, Z, O1, O2>
where
    E: Executor<EM, Z> + HasObservers,
    EM: UsesState<State = E::State>,
    F: TestcaseScore<E::State>,
    M: Mutator<E::Input, E::State>,
    E::State: HasClientPerfMonitor + HasCorpus + HasMetadata + HasRand,
    Z: Evaluator<E, EM, State = E::State>,
    O1: DistanceObserver<E::State>,
    O2: MapObserver,
{
    /// Creates a new [`FilterPowerMutationalStage`]
    pub fn new(mutator: M, dist_observer: &O1, targets_observer: &O2, filter: Box<dyn FilterTrait>) -> Result<Self, Error> {
        Self::transforming(mutator, dist_observer, targets_observer, filter)
    }
}

impl<E, F, EM, I, M, Z, O1, O2> FilterPowerMutationalStage<E, F, EM, I, M, Z, O1, O2>
where
    E: Executor<EM, Z> + HasObservers,
    EM: UsesState<State = E::State>,
    F: TestcaseScore<E::State>,
    M: Mutator<I, E::State>,
    E::State: HasClientPerfMonitor + HasCorpus + HasMetadata + HasRand,
    Z: Evaluator<E, EM, State = E::State>,
    O1: DistanceObserver<E::State>,
    O2: MapObserver,
{
    /// Creates a new transforming [`FilterPwerMutationalStage`]
    pub fn transforming(mutator: M, dist_observer: &O1, targets_observer: &O2, filter: Box<dyn FilterTrait>) -> Result<Self, Error> {
        let mut out = BufWriter::new(File::create("data.csv")?);
        writeln!(out, "id,iter_id,iter_count,mut_sz,parent_dist,eval_result,mut_dist,target,decision")?;
        
        Ok(Self {
            mutator,
            phantom: PhantomData,
            buff: out,
            dist_name: dist_observer.name().to_string(),
            targets_name: targets_observer.name().to_string(),
            filter
        })
    }
}

pub type FilterDistancePowerMutationalStage<E, EM, I, M, Z, O1, O2> =
    FilterPowerMutationalStage<E, DistancePowerTestcaseScore, EM, I, M, Z, O1, O2>;


