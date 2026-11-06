//! A byte-exact subset of Rich 15.0.0 (owned by P2-A).
//!
//! Python allegro-buddy prints everything through Rich; bab must emit the SAME bytes,
//! including ANSI SGR sequences, wrapping, padding and trailing whitespace. Reference source:
//! `rich==15.0.0`. When this code and Rich disagree, Rich wins. Each piece mirrors a Rich
//! module; generated tables ([`data`], [`cell_tables`], [`emoji_data`]) come from
//! `tools/gen_rich_data.py`.
//!
//! # Pipeline (as in Rich)
//!
//! A [`Renderable`] renders to a flat stream of [`Segment`]s including `"\n"` segments
//! (Rich `Console.render`, i.e. `__rich_console__` fully expanded). The [`Console`] crops
//! the stream to its width, renders each styled segment as `ESC[<sgr>m<text>ESC[0m`
//! ([`Style::render`]; nothing when there is no colour system) and writes or captures it.
//!
//! # Pieces
//!
//! * [`style`], [`color`], [`theme`]: `Style`/`Color` parsing, combining, SGR codes and
//!   downgrading; Rich's default theme (`repr.*`, `table.header`, ...).
//! * [`segment`]: `Segment` and the line helpers.
//! * [`cells`]: cell widths with Rich's own Unicode table.
//! * [`text`]: `Text` with spans, wrapping (`divide_line`), justification.
//! * [`markup`], [`emoji`], [`highlight`]: what `Console.render_str` does to plain `str`s.
//! * [`table`], [`boxes`], [`padding`]: `Table` (incl. `Table.grid`) with Rich's column
//!   width algorithm, box styles and `Padding`.
//! * [`console`], [`status`]: consoles (detection, print, capture, clear) and the spinner.
//!
//! # Testing against Rich
//!
//! The renderer was checked byte-for-byte against Rich on generated tables and texts
//! (plus a seeded random set for one-off checks); that harness lives at the git tag
//! `equivalence-harness-3` (`tests/render_diff.rs`, `tools/render/`).
//!
//! # Known divergences
//!
//! Links/meta styles are ignored (Rich's link ids are random); ASCII-only encodings (Rich
//! substitutes ASCII boxes) and legacy Windows are not supported; `UNICODE_VERSION` is
//! ignored (always Unicode 17.0.0 widths, see [`cells`]); see [`console`] for the
//! console-level ones.

pub mod boxes;
pub mod cell_tables;
pub mod cells;
pub mod color;
pub mod console;
pub mod data;
pub mod emoji;
pub mod emoji_data;
pub mod highlight;
pub mod markup;
pub mod padding;
pub mod segment;
pub mod status;
pub mod style;
pub mod table;
pub mod text;
pub mod theme;

pub use boxes::BoxStyle;
pub use console::{Console, ConsoleConfig, PrintOptions, RichStr, stderr, stdout};
pub use padding::Padding;
pub use segment::{Segment, VerticalAlign};
pub use style::Style;
pub use table::{Cell, Column, Table};
pub use text::{Justify, Overflow, Text, TextPart};
pub use theme::Theme;

/// Rich `ConsoleOptions`.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderOptions {
    /// Console size (width, height).
    pub size: (usize, usize),
    pub min_width: usize,
    /// Maximum width in cells for this render (`options.max_width`).
    pub max_width: usize,
    pub is_terminal: bool,
    pub max_height: usize,
    pub justify: Option<Justify>,
    pub overflow: Option<Overflow>,
    pub no_wrap: Option<bool>,
    /// `None` = console default (Rich `options.highlight`).
    pub highlight: Option<bool>,
    /// `None` = console default (Rich `options.markup`).
    pub markup: Option<bool>,
    pub height: Option<usize>,
}

impl RenderOptions {
    /// Default options for a console of the given size (Rich `Console.options`).
    pub fn for_size(width: usize, height: usize, is_terminal: bool) -> RenderOptions {
        RenderOptions {
            size: (width, height),
            min_width: 1,
            max_width: width,
            is_terminal,
            max_height: height,
            justify: None,
            overflow: None,
            no_wrap: Some(false),
            highlight: None,
            markup: None,
            height: None,
        }
    }

    /// Rich `update_width(width)` / `update(width=width)` (negative widths clamp to 0).
    pub fn update_width(&self, width: isize) -> RenderOptions {
        let width = width.max(0) as usize;
        RenderOptions {
            min_width: width,
            max_width: width,
            ..self.clone()
        }
    }

    /// Rich `update_height(height)`.
    pub fn update_height(&self, height: usize) -> RenderOptions {
        RenderOptions {
            max_height: height,
            height: Some(height),
            ..self.clone()
        }
    }

    /// Rich `update(height=height)` (`None` resets).
    pub fn with_height(&self, height: Option<usize>) -> RenderOptions {
        let mut options = self.clone();
        if let Some(h) = height {
            options.max_height = h;
        }
        options.height = height;
        options
    }

    /// Rich `reset_height()`.
    pub fn reset_height(&self) -> RenderOptions {
        RenderOptions {
            height: None,
            ..self.clone()
        }
    }
}

/// Rich `Measurement`: minimum and maximum width (cells) a renderable needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Measurement {
    pub minimum: usize,
    pub maximum: usize,
}

impl Measurement {
    pub fn new(minimum: usize, maximum: usize) -> Measurement {
        Measurement { minimum, maximum }
    }

    /// Rich `normalize()`.
    pub fn normalize(self) -> Measurement {
        let minimum = self.minimum.min(self.maximum);
        Measurement::new(minimum, minimum.max(self.maximum))
    }

    /// Rich `with_maximum(width)`.
    pub fn with_maximum(self, width: usize) -> Measurement {
        Measurement::new(self.minimum.min(width), self.maximum.min(width))
    }

    /// Rich `with_minimum(width)`.
    pub fn with_minimum(self, width: usize) -> Measurement {
        Measurement::new(self.minimum.max(width), self.maximum.max(width))
    }

    /// Rich `clamp(min_width, max_width)`.
    pub fn clamp(self, min_width: Option<usize>, max_width: Option<usize>) -> Measurement {
        let mut m = self;
        if let Some(w) = min_width {
            m = m.with_minimum(w);
        }
        if let Some(w) = max_width {
            m = m.with_maximum(w);
        }
        m
    }

    /// Rich `Measurement.get(console, options, renderable)`.
    pub fn get(
        console: &Console,
        options: &RenderOptions,
        renderable: &dyn Renderable,
    ) -> Measurement {
        let max_width = options.max_width;
        if max_width < 1 {
            return Measurement::new(0, 0);
        }
        let measured = renderable
            .measure(console, options)
            .normalize()
            .with_maximum(max_width);
        if measured.maximum < 1 {
            return Measurement::new(0, 0);
        }
        measured.normalize()
    }
}

/// Something the console can print (Rich `__rich_console__` + `__rich_measure__`).
pub trait Renderable {
    /// Render to a flat segment stream including `"\n"` segments (Rich `Console.render`).
    fn render(&self, console: &Console, options: &RenderOptions) -> Vec<Segment>;

    /// Rich `__rich_measure__`; the default is Rich's for renderables without one.
    fn measure(&self, _console: &Console, options: &RenderOptions) -> Measurement {
        Measurement::new(0, options.max_width)
    }

    /// `Some` for `Text` (Console.print joins text objects with `sep`/`end`).
    fn as_text(&self) -> Option<&Text> {
        None
    }
}
