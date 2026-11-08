
#[derive(Debug, Clone)]
pub struct PngFeatures {
    pub valid_signature: bool,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub bit_depth: Option<u8>,
    pub color_type: Option<u8>,
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
