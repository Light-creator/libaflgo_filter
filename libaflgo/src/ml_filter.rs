
use crate::input_filter::{MutantFeatures, MutantOutcome, FilterTrait, Decision};
use libafl::bolts::rands::{Rand, StdRand};
use std::fs::File;
use std::collections::HashSet;
use std::io::{self, BufRead, BufReader};

#[derive(Debug)]
pub struct TrainingSample {
    pub features: MutantFeatures,
    pub reached_intermediates: Vec<usize>,
    pub reached_final: bool
}

#[derive(Debug, PartialEq)]
pub enum FilterPhase {
    Filtering { decisions: usize },
    Collecting
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LearningTarget {
    Intermediate(usize),
    Final,
}

#[derive(Debug)]
pub struct MlFilter {
    samples: Vec<TrainingSample>,
    max_samples: usize,
    library: libloading::Library,
    pred_succes: usize,
    pred_errors: usize,
    phase: FilterPhase,
    threshold: f32,
    exploration_prob: f64,
    rng: StdRand,
    executed: u64,
    discarded: u64,
    explored: u64,
    intermediate_hits: [usize; 256],
    learning_target: LearningTarget,
    intermediate_chain: Vec<usize>
}

pub fn prep_mutant_features(features: &MutantFeatures) -> [f32; 10] {
    
    let res: [f32; 10] = [
        (features.mut_sz as f32).ln_1p(),
        (features.png.valid_signature as i32) as f32,
        ((features.png.width.is_some()) as i32) as f32,
        (features.png.width.unwrap_or(0) as f32).ln_1p(),
        ((features.png.height.is_some()) as i32) as f32,
        (features.png.height.unwrap_or(0) as f32).ln_1p(),
        ((features.png.bit_depth.is_some()) as i32) as f32,
        ((features.png.bit_depth.unwrap_or(0)) as f32),
        ((features.png.color_type.is_some()) as i32) as f32,
        ((features.png.color_type.unwrap_or(0)) as f32),
    ]; 
    
    res
} 

impl MlFilter {
    pub fn new(max_samples: usize, path: &str, seed: u64, node_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let library = unsafe { libloading::Library::new(path) }?;
    
        let intermediate_chain = Self::read_target_csv(node_path)?;
        println!("intermediate_chain: {:?}", intermediate_chain);

        Ok(Self {
            samples: Vec::new(),
            max_samples,
            library,
            pred_succes: 0,
            pred_errors: 0,
            phase: FilterPhase::Collecting,
            threshold: 0.5,
            exploration_prob: 0.1,
            rng: StdRand::with_seed(seed),
            executed: 0,
            discarded: 0,
            explored: 0,
            intermediate_hits: [0; 256],
            learning_target: LearningTarget::Final,
            intermediate_chain
        })
    }

    pub fn predict(&self, features: &MutantFeatures) -> Result<f32, libloading::Error> {
        
        let mut_features = prep_mutant_features(features);
        
        type Predict = unsafe extern "C" fn(*const f32, usize) -> f32;
        let model_func: libloading::Symbol<Predict> = unsafe { self.library.get(b"model_func\0") }?;
        
        let prob = unsafe{ model_func(mut_features.as_ptr(), mut_features.len()) };
        
        Ok(prob)
    }

    pub fn train(&self) -> Result<i32, libloading::Error> {
        if self.samples.len() == 0 {
            return Ok(-1);
        }

        let mut features: Vec<f32> = vec![];
        let mut labels: Vec<f32> = vec![];
        for sample in self.samples.iter() {
            features.extend_from_slice(&prep_mutant_features(&sample.features));
            if self.reached_learning_target(sample) { 
                labels.push(1.0); 
            } else {
                labels.push(0.0);
            }
        }

        type Train = unsafe extern "C" fn(*const f32, *const f32, usize) -> i32;
        let model_train: libloading::Symbol<Train> = unsafe { self.library.get(b"model_train\0") }?;

        let train_res = unsafe{ model_train(features.as_ptr(), labels.as_ptr(), self.samples.len()) };

        Ok(train_res)
    }

    fn reached_learning_target(&self, sample: &TrainingSample) -> bool {
        match self.learning_target {
            LearningTarget::Intermediate(id) => sample.reached_intermediates.contains(&id),
            LearningTarget::Final => sample.reached_final,
        }
    }

    fn read_target_csv(path: &str) -> io::Result<Vec<usize>> {
        let invalid = |message: String| io::Error::new(io::ErrorKind::InvalidData, message);
        let mut lines = BufReader::new(File::open(path)?).lines();
        let header = lines.next().transpose()?.ok_or_else(|| {
            invalid("Missing intermediate-node CSV header".to_string())
        })?;
        if header != "target_id,node_id,depth,function,block" {
            return Err(invalid("Unexpected intermediate-node CSV header".to_string()));
        }

        let mut target = None;
        let mut nodes = Vec::new();
        let mut seen_ids = HashSet::new();
        let mut seen_depths = HashSet::new();
        for (index, line) in lines.enumerate() {
            let line = line?;
            let line_number = index + 2;
            let mut parts = line.splitn(4, ',');
            let mut read_number = |field: &str| -> io::Result<usize> {
                let value = parts.next().ok_or_else(|| {
                    invalid(format!("Line {line_number}: missing {field}"))
                })?;
                value.parse().map_err(|_| {
                    invalid(format!("Line {line_number}: invalid {field}"))
                })
            };
            let target_id = read_number("target_id")?;
            let node_id = read_number("node_id")?;
            let depth = read_number("depth")?;
            if parts.next().is_none() {
                return Err(invalid(format!("Line {line_number}: missing node description")));
            }
            if let Some(expected) = target {
                if target_id != expected {
                    return Err(invalid(format!("Line {line_number}: multiple final targets are unsupported")));
                }
            } else {
                target = Some(target_id);
            }
            if node_id >= 256 {
                return Err(invalid(format!("Line {line_number}: node_id exceeds map capacity")));
            }
            if !seen_ids.insert(node_id) || !seen_depths.insert(depth) {
                return Err(invalid(format!("Line {line_number}: duplicate node or depth")));
            }
            nodes.push((depth, node_id));
        }
        nodes.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        Ok(nodes.into_iter().map(|(_, node_id)| node_id).collect())
    }

    fn select_learning_target(&self) -> Option<LearningTarget> {
        let reached_final_count = self.samples.iter().filter(|sample| sample.reached_final).count();
        if reached_final_count >= 50 && self.samples.len() - reached_final_count >= 50 {
            return Some(LearningTarget::Final);
        }

        for intermediate_index in self.intermediate_chain.iter().rev() {
            let reached_intermediates_count = self.samples.iter()
                .filter(|sample| sample.reached_intermediates.contains(intermediate_index)).count();
            if reached_intermediates_count >= 50 && self.samples.len() - reached_intermediates_count >= 50 {
                return Some(LearningTarget::Intermediate(*intermediate_index));
            }
        }

        None
    }

    fn model_reset(&self) -> Result<i32, libloading::Error> {
        type ModelReset = unsafe extern "C" fn() -> i32;
        let model_reset: libloading::Symbol<ModelReset> = unsafe { self.library.get(b"model_reset\0") }?;

        let reset_res = unsafe { model_reset() };

        Ok(reset_res)
    }

}


impl FilterTrait for MlFilter {
    
    fn make_decision(&mut self, features: &MutantFeatures) -> Decision {
        if let FilterPhase::Filtering { decisions } = &mut self.phase {
            if *decisions >= 10_000 {
                self.phase = FilterPhase::Collecting;
                self.samples.clear();
                println!("Start Collectig...");
            } else {
                *decisions += 1;
            }
        } 


        
        let decision = if self.phase == FilterPhase::Collecting {
            Decision::Execute
        } else {
            // The stage observes this same mutant immediately after execution.
            let prob = match self.predict(features) {
                Ok(prob) => {
                    if prob.is_finite() && prob >= 0.0 && prob <= 1.0 { 
                        self.pred_succes += 1;
                        Some(prob)
                    } else {
                        self.pred_errors += 1;
                        None 
                    }
                },
                Err(_) => { 
                    self.pred_errors += 1;
                    None
                }
            };

            if self.pred_succes + self.pred_errors == 1000 {
                println!("ERRORS: {} | SUCCESSES: {}", self.pred_errors, self.pred_succes);
            }

            match prob {
                Some(p) if p < self.threshold => {
                    let val = self.rng.below(1_000_000);
                    let th = (self.exploration_prob * 1_000_000.0) as u64;
                    if val < th {
                        Decision::ExecuteDespite
                    } else {
                        Decision::Discard
                    }
                }
                _ => Decision::Execute,
            }
        };

        match decision {
            Decision::Execute => self.executed += 1,
            Decision::Discard => self.discarded += 1,
            Decision::ExecuteDespite => self.explored += 1,
        }
        let total = self.executed + self.discarded + self.explored;
        if total % 10_000 == 0 {
            println!(
                "Filter decisions: execute={} | discard={} | explore={}",
                self.executed, self.discarded, self.explored
            );
        }
        decision
    }
    
    fn make_observation(&mut self, _features: &MutantFeatures, _outcome: &MutantOutcome) {
        if self.phase != FilterPhase::Collecting {
            return;
        }

        for intermidiate in _outcome.reached_intermediates.iter() {
            if *intermidiate < self.intermediate_hits.len() {
                self.intermediate_hits[*intermidiate] += 1;
            }
        }

        if self.samples.len() < self.max_samples {
            let training_sample = TrainingSample{
                features: _features.clone(),
                reached_intermediates: _outcome.reached_intermediates.clone(),
                reached_final: _outcome.target > 0,
            };    
        
            self.samples.push(training_sample);
        
            if self.samples.len() == self.max_samples {
                let learning_target = match self.select_learning_target() {
                    Some(target) => target,
                    None => {
                        println!("Failed to get new learning_target");
                        self.samples.clear();
                        self.intermediate_hits.fill(0);
                        return;
                    }
                };

                if learning_target != self.learning_target {
                    match self.model_reset() {
                        Ok(0) => {
                            println!("model_reset because of changing learning_target: {:?} -> {:?}",
                                self.learning_target, learning_target);
                            self.learning_target = learning_target;
                        },
                        Ok(code) => {
                            println!("model_reset Error: code({})...", code);
                            self.samples.clear();
                            self.phase = FilterPhase::Collecting;
                            self.intermediate_hits.fill(0);
                            return;
                        },
                        Err(err) => {
                            println!("model_reset Error: code({})...", err);
                            self.samples.clear();
                            self.phase = FilterPhase::Collecting;
                            self.intermediate_hits.fill(0);
                            return;
                        }
                    }
                }

                let positive = self.samples.iter().filter(|sample| self.reached_learning_target(sample)).count();
                let negative = self.samples.len() - positive;
                println!(
                    "Training samples: {} | positive: {} | negative: {}",
                    self.samples.len(), positive, negative
                );
                
                for i in 0..self.intermediate_hits.len() {
                    if self.intermediate_hits[i] != 0 {
                        println!("Intermediate {}: {} / {}", i, self.intermediate_hits[i], self.samples.len());
                    }
                }

                self.intermediate_hits.fill(0);

                match self.train() {
                    Ok(0) => {
                        self.phase = FilterPhase::Filtering { decisions: 0 };
                        self.samples.clear();
                        println!("Training completed; filtering started...");
                    },
                    Ok(code) => {
                        self.phase = FilterPhase::Collecting;
                        self.samples.clear();
                        println!("Failed to train model (Code: {})...", code);
                    },
                    Err(err) => {
                        self.phase = FilterPhase::Collecting;
                        self.samples.clear();
                        println!("Error occured: {}", err)
                    }
                }
            }

            return;
        }
    }

}
