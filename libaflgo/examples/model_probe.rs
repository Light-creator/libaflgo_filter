use std::{env, error::Error, io};

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args_os().nth(1).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "Usage: model_probe <library.so>")
    })?;

    // Load only our trusted model library: loading runs its initializers.
    let library = unsafe { libloading::Library::new(path) }?;
    type Predict = unsafe extern "C" fn(*const f32, usize) -> f32;
    // This signature must match the model library's exported C interface.
    let predict: libloading::Symbol<Predict> = unsafe { library.get(b"model_func\0") }?;

    let features = [0.0_f32; 10];
    // The array is aligned, readable, and remains alive throughout the call.
    // `library` also remains loaded while its function is being used.
    let probability = unsafe { predict(features.as_ptr(), features.len()) };
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Model returned an invalid probability: {probability}"),
        )
        .into());
    }
    println!("Probability: {probability}");
    Ok(())
}
