use std::{collections::HashSet, fs, io, path::Path, time::{Duration, Instant}};
use std::{path::PathBuf, process::Command, sync::atomic::{AtomicU64, Ordering}};

static RUN_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Run the separately instrumented target once. Partial results remain usable
/// after a timeout or target crash; spawn and filesystem errors are propagated.
pub fn run_symcc(input: &[u8]) -> io::Result<PathBuf> {
    let root = Path::new("/home/light/Projects/fuzz_hybrid/test_libpng/symcc/runs");
    fs::create_dir_all(root)?;
    // create_dir atomically reserves a fresh directory, including after restarts.
    let run_dir = loop {
        let number = RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!("run-{}-{number}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => break path,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    let input_path = run_dir.join("input.png");
    let results = run_dir.join("results");
    fs::create_dir(&results)?;
    fs::write(&input_path, input)?;

    let status = Command::new("timeout")
        .args(["-k", "5s", "30s"])
        .arg("/home/light/Projects/fuzz_hybrid/test_libpng/symcc/png_symcc")
        .arg(&input_path)
        .env("SYMCC_INPUT_FILE", &input_path)
        .env("SYMCC_OUTPUT_DIR", &results)
        .status()?;
    println!("[SYMCC] Completed: {} | results: {}", status, results.display());
    // timeout's own launch/invocation failures must not look like a completed run.
    if matches!(status.code(), Some(125 | 126 | 127)) {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to execute SymCC target: {status}"),
        ));
    }
    Ok(results)
}

#[derive(Debug, PartialEq)]
pub enum StagnationState {
    Idle, Running
}

#[derive(Debug)]
pub struct StagnationTracker {
    last_progress: Instant,
    timeout: Duration,
    best_distance: Option<f64>,
    seen_intermediates: [bool; 256],
    seen_targets: [bool; 256],
    last_trigger: Option<Instant>,
    diagnostic_at: Instant,
    trigger_checks: u64,
    observations: u64,
    corpus_progress: u64,
    distance_progress: u64,
    node_progress: u64,
    progress_reason: &'static str,
    state: StagnationState
}

impl StagnationTracker {
    
    pub fn new(timeout: Duration) -> Self {
        Self {
            last_progress: Instant::now(),
            timeout,
            best_distance: None,
            seen_intermediates: [false; 256],
            seen_targets: [false; 256],
            last_trigger: None,
            diagnostic_at: Instant::now(),
            trigger_checks: 0,
            observations: 0,
            corpus_progress: 0,
            distance_progress: 0,
            node_progress: 0,
            progress_reason: "initialization",
            state: StagnationState::Idle
        }
    }

    pub fn observe(&mut self, new_corpus_entry: bool, distance: f64, 
        reached_intermediates: &[usize], reached_targets: &[usize]) {
        self.observations += 1;
        if new_corpus_entry {
            self.corpus_progress += 1;
            self.progress_reason = "corpus entry";
            self.last_progress = Instant::now();
        }

        if distance.is_finite() {
            if let Some(best_distance) = self.best_distance {
                if distance < best_distance {
                    self.best_distance = Some(distance);
                    self.distance_progress += 1;
                    self.progress_reason = "distance improved";
                    self.last_progress = Instant::now();
                }
            } else {
                self.best_distance = Some(distance);
                self.distance_progress += 1;
                self.progress_reason = "first finite distance";
                self.last_progress = Instant::now();
            }
        }

        for &i in reached_intermediates {
            if let Some(intermediate) = self.seen_intermediates.get_mut(i) {
                if !*intermediate {
                    *intermediate = true;
                    self.node_progress += 1;
                    self.progress_reason = "new intermediate";
                    self.last_progress = Instant::now();
                }
            }
        }

        for &i in reached_targets {
            if let Some(target) = self.seen_targets.get_mut(i) {
                if !*target {
                    *target = true;
                    self.node_progress += 1;
                    self.progress_reason = "new final target";
                    self.last_progress = Instant::now();
                }
            }
        }
    }

    pub fn finish_attempt(&mut self) {
        self.state = StagnationState::Idle;
        self.last_trigger = Some(Instant::now());
    }

    pub fn is_stagnating(&self) -> bool {
        if self.timeout <= self.last_progress.elapsed() {
            return true;
        }

        false
    }
    
    pub fn should_trigger(&mut self) -> bool {
        self.trigger_checks += 1;
        if self.diagnostic_at.elapsed() >= Duration::from_secs(5) {
            eprintln!(
                "[SYMCC] diagnostic: state={:?} idle={:.1}s timeout={:.1}s last_reason={} best_distance={:?} checks={} observations={} progress(corpus={}, distance={}, nodes={}) cooldown_age={:?}",
                self.state, self.last_progress.elapsed().as_secs_f64(),
                self.timeout.as_secs_f64(), self.progress_reason, self.best_distance,
                self.trigger_checks, self.observations, self.corpus_progress,
                self.distance_progress, self.node_progress,
                self.last_trigger.map(|time| time.elapsed().as_secs_f64())
            );
            self.diagnostic_at = Instant::now();
        }
        if self.state == StagnationState::Running || !self.is_stagnating() {
            return false;
        }

        match self.last_trigger {
            Some(last_trigger) => {
                if last_trigger.elapsed() >= self.timeout {
                    self.state = StagnationState::Running;
                    return true;
                } else {
                    return false;
                }
            },
            None => {
                self.state = StagnationState::Running;
                return true;               
            }
        } 
    }

}

/// Read one completed SymCC run, in filename order, deduplicating by contents.
/// This function does not execute inputs or change the fuzzer corpus.
pub fn read_unique_inputs(directory: &Path) -> io::Result<Vec<Vec<u8>>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            paths.push(entry.path());
        }
    }
    paths.sort();

    let file_count = paths.len();
    let mut seen = HashSet::new();
    let mut inputs = Vec::new();
    for path in paths {
        let bytes = fs::read(path)?;
        if seen.insert(bytes.clone()) {
            inputs.push(bytes);
        }
    }
    println!("[SYMCC] Files: {} | unique inputs: {}", file_count, inputs.len());
    Ok(inputs)
}
