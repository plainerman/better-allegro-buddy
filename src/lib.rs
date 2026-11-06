//! better-allegro-buddy (`bab`): a 1:1 Rust rewrite of the Python CLI allegro-buddy 2.7 (`ab`).
//!
//! # Ground rules for every module (binding for all agents)
//!
//! * **Byte-identical UI.** Every byte on stdout/stderr, every exit code and every mutating
//!   Slurm call (in order) must match the Python reference (allegro-buddy==2.7 with rich==15.0.0, click==8.3.2,
//!   rich-click==1.9.9), except the program name: `ab` -> `bab` ([`core::CLI_NAME`]). The
//!   Python source is commit `91689b9` (upstream v2.7): `git show 91689b9:src/allegro_buddy/<file>`.
//!   Each Rust item names the Python function it mirrors (`Mirrors core.foo`).
//! * **Drop-in compatible**: same config paths, env vars (`AB_CLUSTER`, `AB_SITE_CONFIG`, ...),
//!   cache files and dev job name (`ab-dev`), so `ab` and `bab` can be used side by side.
//! * **Errors are values.** Python's `die()` raises `SystemExit`; here [`error::die`] prints the
//!   message and *returns* an [`error::AbError`] that callers propagate with `?` /
//!   `return Err(die(..))`. Only `main.rs` calls `std::process::exit` (`--watch` must be able
//!   to recover from a `die()` and retry).
//! * **Output goes through [`render::console`]** (`stdout()` / `stderr()` consoles), never
//!   `println!`, so `--watch` can capture it. Exception: `click.echo` equivalents in
//!   [`cli`] (completion scripts) which are plain writes, like click.
//! * **Test hooks** in [`hooks`] replace wall clock, groups, hostname and isatty(stdin) when
//!   the `BAB_TEST_*` variables are set (only with the `test-hooks` cargo feature); real
//!   behaviour otherwise. Never read the clock,
//!   groups, user, hostname or stdin-tty state any other way.
//! * **Slurm calls** only through [`core::slurm`]. Read-only queries may start early and
//!   side by side ([`core::slurm::prefetch`], [`core::slurm::parallel`]), even when an
//!   earlier failure leaves one unused, or be replaced by cheaper ones carrying the same
//!   information ([`core::fast`]); results are still evaluated in Python's order, so output
//!   and errors are Python's. Never add, reorder or parallelize mutations.
//!
//! # Module map (Python -> Rust)
//!
//! | Python                        | Rust                                              |
//! |-------------------------------|---------------------------------------------------|
//! | rich (Text/Style/Table/Console) | [`render`] (style, text, markup, highlight, table, console, segment) |
//! | cli.py, click, rich-click     | [`cli`] (commands, parser, help, completion, prompt, watch) |
//! | core.py                       | [`core`] (slurm, parse, timefmt, gpu, users, nodes, partitions, jobs, reservations, ui) |
//! | settings.py                   | [`settings`]                                      |
//! | fit.py                        | [`fit`]                                           |
//! | handlers/*.py                 | [`handlers`] (resources, why, dev, job, config)   |
//! | (test monkeypatching)         | [`hooks`]                                         |
//! | KeyboardInterrupt             | [`signals`]                                       |
//! | repr()/shlex/format(:g)       | [`pyfmt`]                                         |
//!
//! # Ownership of the next phase
//!
//! * P2-A render: [`render`].
//! * P2-B cli framework: [`cli`] (parser, help, completion protocol/scripts, prompt, watch,
//!   dispatch), [`signals`].
//! * P2-C core: [`core`] (except `core::ui` rendering helpers, shared with P2-A),
//!   [`settings`], [`fit`], [`pyfmt`], [`hooks`]; ports of the Python unit tests.
//! * P3 handlers: [`handlers`].

// Skeleton phase: most items are stubs that later agents fill in.
#![allow(dead_code, unused)]

pub mod cli;
pub mod core;
pub mod error;
pub mod fit;
pub mod handlers;
pub mod hooks;
pub mod pyfmt;
pub mod render;
pub mod settings;
pub mod signals;

pub use error::{AbError, Result, die};

/// What `bab --version` prints after `"bab, version "` (click's version_option with the
/// reference renamed: `ab, version 2.7` -> `bab, version 2.7`).
pub const VERSION_DISPLAY: &str = "2.7";
