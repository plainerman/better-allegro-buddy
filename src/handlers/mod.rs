//! Command handlers: ports of `handlers/*.py`. Owned by the P3 agents
//! (D: [`resources`]; E: [`why`]; F: [`dev`], [`job`], [`config`] + cli wiring).
//!
//! Each `cmd_*` mirrors the Python function of the same name, takes the already-parsed
//! (typed) arguments from [`crate::cli`], prints through [`crate::render::console`] and
//! returns `Result<()>`; failures are `Err(die(..))` (see [`crate::error`]).
//! Handlers never call `std::process::exit`.

pub mod config;
pub mod dev;
pub mod job;
pub mod resources;
pub mod why;
