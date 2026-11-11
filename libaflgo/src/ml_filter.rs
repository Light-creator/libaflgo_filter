
use crate::input_filter::{MutantFeatures, MutantOutcome, FilterTrait, Decision};

#[derive(Debug)]
pub struct TrainingSample {
    pub features: MutantFeatures,
    pub reached: bool
}

#[derive(Debug)]
pub struct MlFilter {
    samples: Vec<TrainingSample>,
    max_samples: usize,
    library: libloading::Library,
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
    pub fn new(max_samples: usize, path: &str) -> Result<Self, libloading::Error> {
        let library = unsafe { libloading::Library::new(path) }?;

        Ok(Self {
            samples: Vec::new(),
            max_samples,
            library
        })
    }

    pub fn predict(&self, features: &MutantFeatures) -> Result<f32, libloading::Error> {
        
        let mut_features = prep_mutant_features(features);
        
        type Predict = unsafe extern "C" fn(*const f32, usize) -> f32;
        let model_func: libloading::Symbol<Predict> = unsafe { self.library.get(b"model_func\0") }?;
        
        let prob = unsafe{ model_func(mut_features.as_ptr(), mut_features.len()) };
        
        Ok(prob)
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
            if self.samples.len() == self.max_samples {
                println!("Limit: {}", self.samples.len());
            }
        }  
    }

}
