//! `bab`: thin entry point. Everything lives in the library crate (`bab::cli::main`);
//! this is the ONLY place that calls `std::process::exit`.

fn main() {
    let argv: Vec<std::ffi::OsString> = std::env::args_os().collect();
    std::process::exit(bab::cli::main(argv));
}
