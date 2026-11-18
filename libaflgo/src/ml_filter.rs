
use crate::input_filter::{MutantFeatures, MutantOutcome, FilterTrait, Decision};
use libafl::bolts::rands::{Rand, StdRand};

#[derive(Debug)]
pub struct TrainingSample {
    pub features: MutantFeatures,
    pub reached: bool
}

#[derive(Debug, PartialEq)]
pub enum FilterPhase {
    Filtering { decisions: usize },
    Collecting
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
    pub fn new(max_samples: usize, path: &str, seed: u64) -> Result<Self, libloading::Error> {
        let library = unsafe { libloading::Library::new(path) }?;

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
            if sample.reached { 
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

        if self.samples.len() < self.max_samples {
            let training_sample = TrainingSample{
                features: _features.clone(),
                reached: (_outcome.target > 0)
            };    
        
            self.samples.push(training_sample);
        
            if self.samples.len() == self.max_samples {
                let positive = self.samples.iter().filter(|sample| sample.reached).count();
                let negative = self.samples.len() - positive;
                println!(
                    "Training samples: {} | positive: {} | negative: {}",
                    self.samples.len(), positive, negative
                );
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
