use std::path::Path;

use crate::input_filter::{MutantFeatures, MutantOutcome, FilterTrait, Decision};
use libafl::bolts::rands::{Rand, StdRand};
use std::fs::File;
use std::collections::HashSet;
use std::io::{self, BufRead, BufReader};

const CALIBRATION_SAMPLES: usize = 1000;
const POSITIVE_THRESHOLD: usize = 30;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingSample {
    pub features: MutantFeatures,
    pub reached_intermediates: Vec<usize>,
    pub reached_final: bool
}

#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FilterPhase {
    Filtering { decisions: usize },
    Collecting,
    Calibrating { calibrations: usize }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
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
    learning_target: Option<LearningTarget>,
    intermediate_chain: Vec<usize>,
    final_target_id: usize,
    model_ready: bool,

    last_decision: Option<Decision>,
    target_execute: u64,
    target_explore: u64,

    calibrating_probs: Vec<Option<f32>>,
    calibrating_labels: Vec<bool>,
}

// pub fn prep_mutant_features(features: &MutantFeatures) -> [f32; 10] {
//    
//     let res: [f32; 10] = [
//         (features.mut_sz as f32).ln_1p(),
//         (features.png.valid_signature as i32) as f32,
//         ((features.png.width.is_some()) as i32) as f32,
//         (features.png.width.unwrap_or(0) as f32).ln_1p(),
//         ((features.png.height.is_some()) as i32) as f32,
//         (features.png.height.unwrap_or(0) as f32).ln_1p(),
//         ((features.png.bit_depth.is_some()) as i32) as f32,
//         ((features.png.bit_depth.unwrap_or(0)) as f32),
//         ((features.png.color_type.is_some()) as i32) as f32,
//         ((features.png.color_type.unwrap_or(0)) as f32),
//     ]; 
//    
//     res
// } 

impl MlFilter {
    pub fn new(max_samples: usize, path: &Path, seed: u64, node_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let library = unsafe { libloading::Library::new(path) }?;
    
        let intermediate_chain = Self::read_target_csv(node_path, 0)?;
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
            learning_target: None,
            intermediate_chain,
            final_target_id: 0,
            model_ready: false,
            last_decision: None,
            target_execute: 0,
            target_explore: 0,
            calibrating_probs: Vec::new(),
            calibrating_labels: Vec::new()
        })
    }

    pub fn predict(&self, features: &MutantFeatures) -> Result<f32, libloading::Error> {
        
        // let mut_features = prep_mutant_features(features);
        let mut_features = features.features.as_slice();
        
        type Predict = unsafe extern "C" fn(*const f32, usize) -> f32;
        let model_func: libloading::Symbol<Predict> = unsafe { self.library.get(b"model_func\0") }?;
        
        let prob = unsafe{ model_func(mut_features.as_ptr(), mut_features.len()) };
        
        Ok(prob)
    }

    pub fn train(&self) -> Result<i32, libloading::Error> {
        if self.learning_target.is_none() {
            return Ok(-1);
        }

        if self.samples.len() == 0 {
            return Ok(-1);
        }

        let mut features: Vec<f32> = vec![];
        let mut labels: Vec<f32> = vec![];
        for sample in self.samples.iter() {
            // features.extend_from_slice(&prep_mutant_features(&sample.features));
            features.extend_from_slice(&sample.features.features.as_slice());
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
            Some(LearningTarget::Intermediate(id)) => sample.reached_intermediates.contains(&id),
            Some(LearningTarget::Final) => sample.reached_final,
            None => false
        }
    }

    fn read_target_csv(path: &Path, final_target_id: usize) -> io::Result<Vec<usize>> {
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
            
            if target_id != final_target_id {
               continue;
            }

            let node_id = read_number("node_id")?;
            let depth = read_number("depth")?;
            if parts.next().is_none() {
                return Err(invalid(format!("Line {line_number}: missing node description")));
            }
            
            target = Some(target_id);
            
            // if let Some(expected) = target {
            //     if target_id != expected {
            //         return Err(invalid(format!("Line {line_number}: multiple final targets are unsupported")));
            //     }
            // } else {
            //     target = Some(target_id);
            // }

            if node_id >= 256 {
                return Err(invalid(format!("Line {line_number}: node_id exceeds map capacity")));
            }

            if !seen_ids.insert(node_id) || !seen_depths.insert(depth) {
                return Err(invalid(format!("Line {line_number}: duplicate node or depth")));
            }

            nodes.push((depth, node_id));
        }

        if target.is_none() {
            return Err(invalid(format!(
                        "Target {final_target_id} not found in intermediate-node CSV"
            )));
        }

        nodes.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        Ok(nodes.into_iter().map(|(_, node_id)| node_id).collect())
    }

    fn select_learning_target(&self) -> Option<LearningTarget> {
        match self.learning_target {
            None => {
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
            },
            Some(LearningTarget::Intermediate(id)) => {
                let reached_final_count = self.samples.iter().filter(|sample| sample.reached_final).count();
                if reached_final_count >= 50 && self.samples.len() - reached_final_count >= 50 {
                    return Some(LearningTarget::Final);
                }
                
                let Some(pos_id) = self.intermediate_chain.iter().position(|&x| x == id) else { 
                    println!("Failed to get position of pos_id...");
                    return None;
                };

                for intermediate_index in self.intermediate_chain[pos_id..].iter().rev() {
                    let reached_intermediates_count = self.samples.iter()
                        .filter(|sample| sample.reached_intermediates.contains(intermediate_index)).count();
                    if reached_intermediates_count >= 50 && self.samples.len() - reached_intermediates_count >= 50 {
                        return Some(LearningTarget::Intermediate(*intermediate_index));
                    }
                }
                return Some(LearningTarget::Intermediate(id));
            },
            Some(LearningTarget::Final) => return Some(LearningTarget::Final)
        }

        None
    }

    fn model_file(&self, path: &Path, load: bool) -> io::Result<()> {
        let path = std::ffi::CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let operation: libloading::Symbol<unsafe extern "C" fn(*const std::os::raw::c_char, bool) -> i32> =
            unsafe { self.library.get(b"model_checkpoint\0") }
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        if unsafe { operation(path.as_ptr(), load) } != 0 {
            return Err(io::Error::new(io::ErrorKind::Other, "Model checkpoint failed"));
        }
        Ok(())
    }
    fn model_reset(&self) -> Result<i32, libloading::Error> {
        type ModelReset = unsafe extern "C" fn() -> i32;
        let model_reset: libloading::Symbol<ModelReset> = unsafe { self.library.get(b"model_reset\0") }?;

        let reset_res = unsafe { model_reset() };

        Ok(reset_res)
    }

}


impl FilterTrait for MlFilter {
    
    fn checkpoint(&self, dir: &Path) -> io::Result<()> {
        self.model_file(&dir.join("model.bpk"), false)?;
        let data = bincode::serialize(&((&self.samples, &self.pred_succes, &self.pred_errors, &self.phase, &self.threshold, &self.rng, &self.executed, &self.discarded, &self.explored), (&self.intermediate_hits.to_vec(), &self.learning_target, &self.model_ready, &self.last_decision, &self.target_execute, &self.target_explore, &self.calibrating_probs, &self.calibrating_labels)))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        std::fs::write(dir.join("filter.bin"), data)
    }
    fn restore(&mut self, dir: &Path) -> io::Result<()> {
        let snapshot: ((Vec<TrainingSample>, usize, usize, FilterPhase, f32, StdRand, u64, u64, u64), (Vec<usize>, Option<LearningTarget>, bool, Option<Decision>, u64, u64, Vec<Option<f32>>, Vec<bool>)) = bincode::deserialize(&std::fs::read(dir.join("filter.bin"))?)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        self.model_file(&dir.join("model.bpk"), true)?;
        self.samples = snapshot.0.0;
        self.pred_succes = snapshot.0.1;
        self.pred_errors = snapshot.0.2;
        self.phase = snapshot.0.3;
        self.threshold = snapshot.0.4;
        self.rng = snapshot.0.5;
        self.executed = snapshot.0.6;
        self.discarded = snapshot.0.7;
        self.explored = snapshot.0.8;
        self.intermediate_hits = snapshot.1.0.try_into().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid hits map"))?;
        self.learning_target = snapshot.1.1;
        self.model_ready = snapshot.1.2;
        self.last_decision = snapshot.1.3;
        self.target_execute = snapshot.1.4;
        self.target_explore = snapshot.1.5;
        self.calibrating_probs = snapshot.1.6;
        self.calibrating_labels = snapshot.1.7;
        Ok(())
    }
    fn make_decision(&mut self, features: &MutantFeatures) -> Decision {
        if let FilterPhase::Calibrating { calibrations } = &mut self.phase {
            if *calibrations == 0 {
                self.calibrating_probs.clear();
                self.calibrating_labels.clear();
            }
        }


        if let FilterPhase::Filtering { decisions } = &mut self.phase {
            if *decisions >= 10_000 {
                println!("Filtering target hits: execute={} | explore={}", self.target_execute, self.target_explore);
                self.target_execute = 0;
                self.target_explore = 0;

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

            if matches!(self.phase, FilterPhase::Calibrating { .. }) {
                self.calibrating_probs.push(prob);
                Decision::Execute
            } else {
                match prob {
                    Some(p) if p < self.threshold => {
                        let val = self.rng.below(1_000_000);
                        let th = (self.exploration_prob * 1_000_000.0) as u64;
                        if val < th { Decision::ExecuteDespite } else { Decision::Discard }
                    }
                    _ => Decision::Execute,
                }
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

        self.last_decision = Some(decision);
        decision
    }
    
    fn make_observation(&mut self, _features: &MutantFeatures, _outcome: &MutantOutcome) {
        let last_decision = self.last_decision.take();
        
        let reached = match self.learning_target {
            Some(LearningTarget::Final) => _outcome.reached_targets.contains(&self.final_target_id),
            Some(LearningTarget::Intermediate(id)) => _outcome.reached_intermediates.contains(&id),
            None => false,
        };

        if let FilterPhase::Calibrating { calibrations } = &mut self.phase {
            self.calibrating_labels.push(reached);
            *calibrations += 1;

            if *calibrations == CALIBRATION_SAMPLES {
                let mut positive_probs: Vec<f32> = self.calibrating_probs.iter()
                    .zip(&self.calibrating_labels)
                    .filter_map(|(prob, label)| if *label { *prob } else { None })
                    .collect();
                let errors = self.calibrating_probs.iter().filter(|p| p.is_none()).count();
                // With insufficient evidence, threshold zero disables rejection.
                if positive_probs.len() < POSITIVE_THRESHOLD || errors != 0 {
                    self.threshold = 0.0;
                    println!("Calibration insufficient: positives={} errors={}; rejection disabled this cycle",
                             positive_probs.len(), errors);
                } else {
                    positive_probs.sort_by(f32::total_cmp);
                    self.threshold = positive_probs[positive_probs.len() / 100];
                    println!("Calibration threshold: {} | positives: {}",
                             self.threshold, positive_probs.len());
                }
                self.calibrating_probs.clear();
                self.calibrating_labels.clear();
                self.phase = FilterPhase::Filtering { decisions: 0 };
            }
            return;
        }

        if matches!(self.phase, FilterPhase::Filtering { .. }) {
            if reached {
                match last_decision {
                    Some(Decision::Execute) => self.target_execute += 1,
                    Some(Decision::ExecuteDespite) => self.target_explore += 1,
                    _ => {},
                }
            }
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
                reached_final: _outcome.reached_targets.contains(&self.final_target_id),
            };    
        
            self.samples.push(training_sample);
        
            if self.samples.len() == self.max_samples {
                println!("Collected: {}", self.samples.len());
                for idx in self.intermediate_chain.iter() {
                    let reached_intermediates_count = self.samples.iter()
                        .filter(|sample| sample.reached_intermediates.contains(idx)).count();
                    println!("intermidiate {}: {}/{}", *idx, reached_intermediates_count, self.samples.len());
                }

                let learning_target = match self.select_learning_target() {
                    Some(target) => Some(target),
                    None => {
                        println!("Failed to get new learning_target");
                        if self.model_ready {
                            self.phase = FilterPhase::Filtering{ decisions: 0 };
                        }
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

                            self.model_ready = false;
                        },
                        Ok(code) => {
                            println!("model_reset Error: code({})...", code);
                            self.samples.clear();
                            self.phase = FilterPhase::Collecting;
                            self.intermediate_hits.fill(0);
                            
                            self.model_ready = false;
                            return;
                        },
                        Err(err) => {
                            println!("model_reset Error: code({})...", err);
                            self.samples.clear();
                            self.phase = FilterPhase::Collecting;
                            self.intermediate_hits.fill(0);
                            
                            self.model_ready = false;
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
                
                if positive < 50 || negative < 50 {
                    println!("Skipping training: insufficient examples of both classes; keeping current model");
                    if self.model_ready {
                        self.phase = FilterPhase::Filtering { decisions: 0 };
                    }

                    self.samples.clear();
                    self.intermediate_hits.fill(0);
                    return;
                }

                for i in 0..self.intermediate_hits.len() {
                    if self.intermediate_hits[i] != 0 {
                        println!("Intermediate {}: {} / {}", i, self.intermediate_hits[i], self.samples.len());
                    }
                }

                self.intermediate_hits.fill(0);

                match self.train() {
                    Ok(0) => {
                        self.phase = FilterPhase::Calibrating { calibrations: 0 };
                        self.samples.clear();
                        self.model_ready = true;
                        println!("Training completed; filtering started...");
                    },
                    Ok(code) => {
                        self.phase = FilterPhase::Collecting;
                        self.samples.clear();
                        self.model_ready = false;
                        println!("Failed to train model (Code: {})...", code);
                    },
                    Err(err) => {
                        self.phase = FilterPhase::Collecting;
                        self.samples.clear();
                        self.model_ready = false;
                        println!("Error occured: {}", err)
                    }
                }
            }

            return;
        }
    }

}
