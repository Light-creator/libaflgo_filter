
use crate::input_filter::{MutantFeatures, MutantOutcome, FilterTrait, Decision};

#[derive(Debug)]
pub struct TrainingSample {
    pub features: MutantFeatures,
    pub reached: bool
}

#[derive(Debug)]
pub struct MlFilter {
    samples: Vec<TrainingSample>,
    max_samples: usize
}

impl MlFilter {
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: Vec::new(),
            max_samples
        }
    }
}


impl FilterTrait for MlFilter {
    
    fn make_decision(&mut self, _features: &MutantFeatures) -> Decision {
        Decision::Execute
    }
    
    fn make_observation(&mut self, _features: &MutantFeatures, _outcome: &MutantOutcome) {
        if self.samples.len() < self.max_samples {
            let training_sample = TrainingSample{
                features: _features.clone(),
                reached: (_outcome.target > 0)
            };    
        
            self.samples.push(training_sample);
        }
    }

}
