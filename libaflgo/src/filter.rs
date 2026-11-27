use core::{fmt::Debug, marker::PhantomData};
use std::{fs::File, io::{BufWriter, Write}};

use crate::DistancePowerTestcaseScore;
use crate::DistanceObserver;
use crate::DistanceTestcaseMetadata;
use crate::png_features::{extract_png_features};

use crate::symcc::{StagnationTracker, SymccConfig, run_symcc, read_unique_inputs};
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
pub struct FilterPowerMutationalStage<E, F, EM, I, M, Z, O1, O2, O3> {
    mutator: M,
    #[allow(clippy::type_complexity)]
    phantom: PhantomData<(E, F, EM, I, Z, O1, O2, O3)>,
    buff: BufWriter<File>,
    dist_name: String,
    targets_name: String,
    intermediate_name: String,
    filter: Box<dyn FilterTrait>,
    stagnation_tracker: StagnationTracker,
    symcc_config: Option<SymccConfig>,
}

impl<E, F, EM, I, M, Z, O1, O2, O3> UsesState for FilterPowerMutationalStage<E, F, EM, I, M, Z, O1, O2, O3>
where
    E: UsesState,
{
    type State = E::State;
}

impl<E, F, EM, I, M, Z, O1, O2, O3> MutationalStage<E, EM, I, M, Z> for FilterPowerMutationalStage<E, F, EM, I, M, Z, O1, O2, O3>
where
    E: Executor<EM, Z> + HasObservers,
    EM: UsesState<State = E::State>,
    F: TestcaseScore<E::State>,
    M: Mutator<I, E::State>,
    E::State: HasClientPerfMonitor + HasCorpus + HasMetadata + HasRand,
    Z: Evaluator<E, EM, State = E::State>,
    I: MutatedTransform<E::Input, E::State> + Clone,
    E::Input: HasTargetBytes + From<Vec<u8>>,
    O1: DistanceObserver<E::State>,
    O2: MapObserver,
    O3: MapObserver,
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

        self.stagnation_tracker.observe_candidate(corpus_idx, parent_dist);

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

            // Извлекаем признаки png
            let png = extract_png_features(
                untransformed.target_bytes().as_slice(),
            );

            // MutantFeatures для принятия решение о запуске
            let mut_features = MutantFeatures{
                id: corpus_idx,
                iter_id: i,
                iter_count: num,
                mut_sz: mut_sz,
                parent_dist: parent_dist,
                png: png
            };

            let png_ref = &mut_features.png;

            let filter_res = self.filter.make_decision(&mut_features);

            if filter_res == Decision::Discard {
                writeln!(self.buff, "{},{},{},{},{},,,,{:?},{},{},{},{},{}", 
                    corpus_idx, i, num, mut_sz, parent_dist, filter_res,
                    png_ref.valid_signature, 
                    png_ref.width.map(|v| v.to_string()).unwrap_or_default(), 
                    png_ref.height.map(|v| v.to_string()).unwrap_or_default(), 
                    png_ref.bit_depth.map(|v| v.to_string()).unwrap_or_default(), 
                    png_ref.color_type.map(|v| v.to_string()).unwrap_or_default()
                )?;

                self.mutator_mut().post_exec(state, i as i32, None)?;
                post.post_exec(state, i as i32, None)?;

                continue;
            }

            let (eval_result, new_idx) = fuzzer.evaluate_input(state, executor, manager, untransformed)?;

            let mut mut_dist = 0.0_f64;
            let mut target = 0_u64;
            let mut reached_targets: Vec<usize> = vec![];
            let mut reached_intermediates: Vec<usize> = vec![];
            
            self.do_observers(executor, &mut reached_intermediates, &mut reached_targets,
                &mut mut_dist, &mut target)?;
            
            if let Some(id) = new_idx {
                self.stagnation_tracker.observe_candidate(id, mut_dist);
            }

            // Проверяем застопорился ли фаззер (Нужно ли вызывать symCC?)
            self.stagnation_tracker.observe(new_idx.is_some(), mut_dist, &reached_intermediates, &reached_targets);

            // MutantOutcome для дообучения модели
            let mut_outcome = MutantOutcome{
                eval_result: eval_result,
                mut_dist: mut_dist,
                target: target,
                reached_targets,
                reached_intermediates
            };

            self.filter.make_observation(&mut_features, &mut_outcome);

            // id, iter_id, iter_count, mut_sz, parent_dist, eval_result, mut_dist, target, decision
            writeln!(self.buff, "{},{},{},{},{},{:?},{},{},{:?},{},{},{},{},{}", 
                corpus_idx, i, num, mut_sz, parent_dist, 
                mut_outcome.eval_result, mut_dist, target, filter_res,
                    png_ref.valid_signature, 
                    png_ref.width.map(|v| v.to_string()).unwrap_or_default(), 
                    png_ref.height.map(|v| v.to_string()).unwrap_or_default(), 
                    png_ref.bit_depth.map(|v| v.to_string()).unwrap_or_default(), 
                    png_ref.color_type.map(|v| v.to_string()).unwrap_or_default()
                )?;

            self.mutator_mut().post_exec(state, i as i32, new_idx)?;
            post.post_exec(state, i as i32, new_idx)?;

            if self.symcc_config.is_some() && self.stagnation_tracker.should_trigger() {
                println!("[SYMCC] Stagnation detected; symbolic execution requested");

                let attempt_result = (|| -> Result<(), Error> {
                    let bytes_symcc = loop {
                        let candidate_idx = match self.stagnation_tracker.best_candidate() {
                            Some(id) => id,
                            None => {
                                println!("[SYMCC] No unused candidates");
                                
                                let p_bytes = {
                                    let mut testcase = state.corpus().get(corpus_idx)?.borrow_mut();
                                    let input = testcase.load_input(state.corpus())?;
                                    input.target_bytes().as_slice().to_vec()
                                };

                                if self.stagnation_tracker.register_input(&p_bytes) {
                                    break p_bytes;
                                }

                                return Ok(());
                            }
                        };

                        let bytes = {
                            let mut testcase = state.corpus().get(candidate_idx)?.borrow_mut();
                            let input = testcase.load_input(state.corpus())?;
                            input.target_bytes().as_slice().to_vec()
                        };

                        if self.stagnation_tracker.register_input(&bytes) {
                            break bytes;
                        }
                    };

                    match run_symcc(&bytes_symcc, self.symcc_config.as_ref().ok_or_else(|| Error::illegal_argument("Missing SymCC configuration"))?) {
                        Ok(res_dir) => {
                            println!("[SYMCC] Path to result dir: {}", res_dir.display());
                            let inputs_symcc = match read_unique_inputs(&res_dir) {
                                Ok(res_read) => Some(res_read),
                                Err(err) => {
                                    println!("[SYMCC] Failed to read inputs: {}", err);
                                    None
                                }
                            };

                            let mut eval_counter = 0;
                            let mut added_counter = 0;
                            if let Some(inputs) = inputs_symcc {
                                for bytes in inputs {
                                    let input = E::Input::from(bytes);
                                    let (_result, new_idx) = fuzzer.evaluate_input(state, executor, manager, input)?;
                                    eval_counter += 1;

                                    if new_idx.is_some() {
                                        added_counter += 1;
                                    }

                                    let mut distance = 0.0_f64;
                                    let mut _target = 0_u64;
                                    let mut reached_targets: Vec<usize> = vec![];
                                    let mut reached_intermediates: Vec<usize> = vec![];

                                    self.do_observers(executor, &mut reached_intermediates, &mut reached_targets,
                                        &mut distance, &mut _target)?;
                                    
                                    if let Some(id) = new_idx {
                                        self.stagnation_tracker.observe_candidate(id, distance);
                                    }

                                    self.stagnation_tracker.observe(
                                        new_idx.is_some(),
                                        distance,
                                        &reached_intermediates,
                                        &reached_targets
                                    );
                                }
                                println!("[SYMCC] eval_counter: {} | added_counter {}", eval_counter, added_counter);
                            }
                        },
                        Err(err) => {
                            println!("[SYMCC] Failed to run symcc: {}", err);
                        }
                    }
                    Ok(())
                })();
                self.stagnation_tracker.finish_attempt();
                attempt_result?;
            }
        }

        self.buff.flush()?;
        Ok(())
    }
}

impl<E, F, EM, I, M, Z, O1, O2, O3> Stage<E, EM, Z> for FilterPowerMutationalStage<E, F, EM, I, M, Z, O1, O2, O3>
where
    E: Executor<EM, Z> + HasObservers,
    EM: UsesState<State = E::State>,
    F: TestcaseScore<E::State>,
    M: Mutator<I, E::State>,
    E::State: HasClientPerfMonitor + HasCorpus + HasMetadata + HasRand,
    Z: Evaluator<E, EM, State = E::State>,
    I: MutatedTransform<E::Input, E::State> + Clone,
    E::Input: HasTargetBytes + From<Vec<u8>>,
    O1: DistanceObserver<E::State>,
    O2: MapObserver,
    O3: MapObserver,
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

impl<E, F, EM, M, Z, O1, O2, O3> FilterPowerMutationalStage<E, F, EM, E::Input, M, Z, O1, O2, O3>
where
    E: Executor<EM, Z> + HasObservers,
    EM: UsesState<State = E::State>,
    F: TestcaseScore<E::State>,
    M: Mutator<E::Input, E::State>,
    E::State: HasClientPerfMonitor + HasCorpus + HasMetadata + HasRand,
    Z: Evaluator<E, EM, State = E::State>,
    O1: DistanceObserver<E::State>,
    O2: MapObserver,
    O3: MapObserver,
{
    /// Creates a new [`FilterPowerMutationalStage`]
    pub fn new(mutator: M, dist_observer: &O1, targets_observer: &O2, intermediate_observer: &O3, filter: Box<dyn FilterTrait>, stagnation_tracker: StagnationTracker, symcc_config: Option<SymccConfig>) -> Result<Self, Error> {
        Self::transforming(mutator, dist_observer, targets_observer, intermediate_observer, filter, stagnation_tracker, symcc_config)
    }
}


impl<E, F, EM, I, M, Z, O1, O2, O3> FilterPowerMutationalStage<E, F, EM, I, M, Z, O1, O2, O3>
where
    E: Executor<EM, Z> + HasObservers,
    EM: UsesState<State = E::State>,
    F: TestcaseScore<E::State>,
    M: Mutator<I, E::State>,
    E::State: HasClientPerfMonitor + HasCorpus + HasMetadata + HasRand,
    Z: Evaluator<E, EM, State = E::State>,
    O1: DistanceObserver<E::State>,
    O2: MapObserver,
    O3: MapObserver,
{
    /// Creates a new transforming [`FilterPwerMutationalStage`]
    pub fn transforming(mutator: M, dist_observer: &O1, targets_observer: &O2, intermediate_observer: &O3, filter: Box<dyn FilterTrait>, stagnation_tracker: StagnationTracker, symcc_config: Option<SymccConfig>) -> Result<Self, Error> {
        let mut out = BufWriter::new(File::create("data.csv")?);
        writeln!(out, "id,iter_id,iter_count,mut_sz,parent_dist,eval_result,mut_dist,target,decision,valid_signature,width,height,bit_depth,color_type")?;
        
        Ok(Self {
            mutator,
            phantom: PhantomData,
            buff: out,
            dist_name: dist_observer.name().to_string(),
            targets_name: targets_observer.name().to_string(),
            intermediate_name: intermediate_observer.name().to_string(),
            filter,
            stagnation_tracker,
            symcc_config,
        })
    }

    pub fn do_observers(&self, executor: &E, reached_intermediates: &mut Vec<usize>, 
        reached_targets: &mut Vec<usize>, distance: &mut f64, target: &mut u64) -> Result<(), Error> {
            let observers = executor.observers();

            let dist_observer = observers.match_name::<O1>(&self.dist_name).ok_or_else(|| {
                libafl::Error::key_not_found("DistanceObserver not found".to_string())
            })?;
            
            let targets_observer = observers.match_name::<O2>(&self.targets_name).ok_or_else(|| {
                libafl::Error::key_not_found("MapObserver not found".to_string())
            })?;

            let intermediate_observer = observers.match_name::<O3>(&self.intermediate_name).ok_or_else(|| {
                libafl::Error::key_not_found("IntermediateObserver not found".to_string())
            })?;

            reached_targets.clear();
            reached_intermediates.clear();

            let initial = targets_observer.initial();

            for index in 0..targets_observer.usable_count() {
                let value = *targets_observer.get(index);

                if value != initial {
                    reached_targets.push(index);
                }
            }

            for index in 0..intermediate_observer.usable_count() {
                let value = *intermediate_observer.get(index);

                if value != intermediate_observer.initial() {
                    reached_intermediates.push(index);
                }
            }

            *distance = dist_observer.distance();
            *target = targets_observer.count_bytes();

            Ok(())
    }
}

pub type FilterDistancePowerMutationalStage<E, EM, I, M, Z, O1, O2, O3> =
    FilterPowerMutationalStage<E, DistancePowerTestcaseScore, EM, I, M, Z, O1, O2, O3>;


