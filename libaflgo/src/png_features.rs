
use crate::features::FeatureExtractor;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PngFeatures {
    pub valid_signature: bool,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub bit_depth: Option<u8>,
    pub color_type: Option<u8>,
}

#[derive(Debug)]
pub struct PngFeatureExtractor;

impl FeatureExtractor for PngFeatureExtractor {
    fn extract_features(&self, data: &[u8]) -> Vec<f32> {
        let mut features: Vec<f32> = vec![0.0_f32; 10];
        
        features[0] = (data.len() as f32).ln_1p();

        if data.starts_with(b"\x89PNG\r\n\x1a\n") {
            features[1] = 1.0;
        } else {
            features[1] = 0.0;
        }

        if data.len() < 33 {
            return features;
        }

        let ihdr_length = u32::from_be_bytes([
            data[8], data[9], data[10], data[11]
        ]);
        
        if ihdr_length != 13 {
            return features;
        }

        if let Some(sub_data) = data.get(12..) {
            if !sub_data.starts_with(b"IHDR") {
                return features;
            }
        }

        let width = Some(u32::from_be_bytes([
                data[16], data[17], data[18], data[19]
        ]));

        features[2] = ((width.is_some()) as i32) as f32;
        features[3] = (width.unwrap_or(0) as f32).ln_1p();


        let height = Some(u32::from_be_bytes([
                data[20], data[21], data[22], data[23]
        ]));

        features[4] = ((height.is_some()) as i32) as f32;
        features[5] = (height.unwrap_or(0) as f32).ln_1p();
        
        let bit_depth = Some(data[24]);
        features[6] = ((bit_depth.is_some()) as i32) as f32;
        features[7] = bit_depth.unwrap_or(0) as f32;

        
        let color_type = Some(data[25]);
        features[8] = ((color_type.is_some()) as i32) as f32;
        features[9] = color_type.unwrap_or(0) as f32;

        features
    }

    fn get_features_count(&self) -> usize {
        10
    }

    fn get_features_name(&self) -> &str {
        "PNG"
    }

    fn get_features_version(&self) -> u32 {
        1
    }
}

pub fn extract_png_features(data: &[u8]) -> PngFeatures {
    let mut features = PngFeatures{
        valid_signature: false,
        width: None,
        height: None,
        bit_depth: None,
        color_type: None,
    };

    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        features.valid_signature = true;
    }

    if data.len() < 33 {
        return features;
    }

    let ihdr_length = u32::from_be_bytes([
        data[8], data[9], data[10], data[11]
    ]);

    if ihdr_length != 13 {
        return features;
    }

    if let Some(sub_data) = data.get(12..) {
        if !sub_data.starts_with(b"IHDR") {
            return features;
        }
    }

    features.width = Some(u32::from_be_bytes([
        data[16], data[17], data[18], data[19]
    ]));

    features.height = Some(u32::from_be_bytes([
        data[20], data[21], data[22], data[23]
    ]));   

    features.bit_depth = Some(data[24]);
    
    features.color_type = Some(data[25]);

    features
}
