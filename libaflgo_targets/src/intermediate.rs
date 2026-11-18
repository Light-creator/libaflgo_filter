use libafl::prelude::StdMapObserver;

const INTERMEDIATE_MAP_SIZE: usize = 256;
static mut INTERMEDIATE_MAP: [u8; INTERMEDIATE_MAP_SIZE] = [0; INTERMEDIATE_MAP_SIZE];

// Called by the target instrumentation
#[no_mangle]
pub unsafe extern "C" fn __aflgo_trace_intermediate(node_id: u32) {
    if node_id < INTERMEDIATE_MAP_SIZE.try_into().unwrap() {
        INTERMEDIATE_MAP[node_id as usize] = 1;
    }
}

pub unsafe fn get_intermediate_map_observer<'a, S>(name: S) -> StdMapObserver<'a, u8, false>
where
    S: Into<String>,
{
    StdMapObserver::new(name, &mut INTERMEDIATE_MAP)
}
