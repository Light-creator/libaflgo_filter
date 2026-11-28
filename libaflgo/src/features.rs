
pub trait FeatureExtractor: std::fmt::Debug {
    
    fn get_features_count(&self) -> usize;
    
    fn extract_features(&self, data: &[u8]) -> Vec<f32>;

    fn get_features_name(&self) -> &str;
    
    fn get_features_version(&self) -> u32;
}
