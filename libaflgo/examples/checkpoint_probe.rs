use libaflgo::{input_filter::{FilterTrait, MutantFeatures}, ml_filter::MlFilter, png_features::PngFeatures};
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 { return Err("Usage: checkpoint_probe library nodes.csv new-directory".into()); }
    let dir = Path::new(&args[3]);
    std::fs::create_dir(dir)?;
    let original = MlFilter::new(1000, Path::new(&args[1]), 1337, Path::new(&args[2]))?;
    let features = MutantFeatures { id: 0_usize.into(), iter_id: 0, iter_count: 1, mut_sz: 0, parent_dist: f64::INFINITY,
        png: PngFeatures { valid_signature: false, width: None, height: None, bit_depth: None, color_type: None } };
    let before = original.predict(&features)?;
    original.checkpoint(dir)?;
    let mut restored = MlFilter::new(1000, Path::new(&args[1]), 999, Path::new(&args[2]))?;
    restored.restore(dir)?;
    assert_eq!(before, restored.predict(&features)?);
    println!("PASS: filter checkpoint round trip");
    Ok(())
}
