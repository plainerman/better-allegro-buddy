//! Rich `Console` subset: the two process-wide consoles and everything they do.
//!
//! Python has `CONSOLE = Console()` and `ERROR_CONSOLE = Console(stderr=True)` in core.py;
//! here they are [`stdout()`] and [`stderr()`]. [`Console::new`] builds others (e.g. the
//! in-memory console rich-click renders help on).
//!
//! # Detection (Rich 15.0.0 `Console.__init__`, `is_terminal`, `size`, `_detect_color_system`)
//!
//! * `is_terminal` (re-evaluated on every use): `force_terminal`; else `TTY_COMPATIBLE`
//!   `0`/`1`; else `FORCE_COLOR` set (non-empty -> True, empty -> False); else `isatty(fd)`.
//! * `is_dumb_terminal`: terminal and `TERM` in (`dumb`, `unknown`).
//! * colour system (once, at creation): `None` when not a terminal or dumb; else `COLORTERM`
//!   in (`truecolor`, `24bit`) -> TrueColor; else by the part of `TERM` after the last `-`:
//!   `256color`/`kitty` -> EightBit, anything else Standard.
//! * `NO_COLOR` non-empty strips colours but keeps attributes.
//! * width: `COLUMNS` (digits, read at creation) wins, even when piped; else the terminal size
//!   of the first of stdin/stdout/stderr that has one, else 80. Dumb terminal -> 80x25.
//!   Height likewise with `LINES` and 25.
//! * `is_interactive` (spinner): `TTY_INTERACTIVE` `0`/`1`, else terminal and not dumb.
//!
//! # Printing
//!
//! `print(renderable)` renders with `options.max_width = min(width, console.width)`, crops
//! every line to the console width, renders each styled segment with [`Style::render`]
//! (no colours under `NO_COLOR`, plain text without a colour system) and writes + flushes,
//! or appends to the capture buffer. Like Rich's `_collect_renderables`, a `Text` (or a
//! `str`, after markup/emoji/highlighting) is re-joined into a new text whose `end` is the
//! print's `end`: the text's own `justify`/`overflow`/`no_wrap`/`end` are not used.
//!
//! # Divergences (documented)
//!
//! * Capture is per console (Rich's buffer is thread-local); bab prints from one thread.
//! * A broken pipe makes Rich raise `SystemExit(1)` from inside `print`; here the console
//!   goes quiet, stdout is pointed at /dev/null (as Rich does) and [`broken_pipe`] returns
//!   true so `main` can exit with status 1.
//! * Invalid markup (Rich raises `MarkupError`, i.e. a traceback) renders the string literally.
//! * `print(justify=...)` (Rich wraps in `Align`) and Jupyter/Windows paths are not supported.

use super::color::ColorSystem;
use super::emoji::emoji_replace;
use super::highlight::{RegexHighlighter, repr_highlighter};
use super::segment::{self, Segment};
use super::style::Style;
use super::text::{Justify, Overflow, Text};
use super::theme::Theme;
use super::{Measurement, RenderOptions, Renderable};
use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Which stream a console writes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
}

impl Stream {
    fn fd(self) -> i32 {
        match self {
            Stream::Stdout => 1,
            Stream::Stderr => 2,
        }
    }
}

/// Rich `Console(color_system=...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorChoice {
    /// `"auto"`: detect from the environment and the terminal.
    #[default]
    Auto,
    /// `None`, `"standard"`, `"256"`, `"truecolor"`.
    Fixed(Option<ColorSystem>),
}

/// `Console(...)` arguments.
#[derive(Debug, Clone)]
pub struct ConsoleConfig {
    /// Where output goes; `None` = an in-memory console ([`Console::take_output`]).
    pub stream: Option<Stream>,
    /// Stream whose tty-ness `ColorChoice::Auto` and `is_interactive` use at creation when
    /// `stream` is `None` (rich-click creates its console on stdout, then swaps the file).
    pub detect_stream: Option<Stream>,
    pub force_terminal: Option<bool>,
    pub force_interactive: Option<bool>,
    pub color_system: ColorChoice,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub no_color: Option<bool>,
    pub theme: Theme,
    pub markup: bool,
    pub emoji: bool,
    pub highlight: bool,
    pub soft_wrap: bool,
    pub tab_size: usize,
    /// Environment override (Rich `_environ`); `None` = the process environment.
    pub env: Option<HashMap<String, String>>,
}

impl Default for ConsoleConfig {
    fn default() -> ConsoleConfig {
        ConsoleConfig {
            stream: Some(Stream::Stdout),
            detect_stream: None,
            force_terminal: None,
            force_interactive: None,
            color_system: ColorChoice::Auto,
            width: None,
            height: None,
            no_color: None,
            theme: Theme::default(),
            markup: true,
            emoji: true,
            highlight: true,
            soft_wrap: false,
            tab_size: 8,
            env: None,
        }
    }
}

/// `Console.print(..)` keyword arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct PrintOptions {
    /// `width=`: capped at the console width.
    pub width: Option<usize>,
    /// `style=`: applied over everything printed.
    pub style: Option<Style>,
    /// `end=` (default `"\n"`), used for `Text`/`str` objects only, like Rich.
    pub end: String,
    pub justify: Option<Justify>,
    pub overflow: Option<Overflow>,
    pub no_wrap: Option<bool>,
    pub markup: Option<bool>,
    pub highlight: Option<bool>,
    pub emoji: Option<bool>,
    pub soft_wrap: Option<bool>,
    pub crop: bool,
}

impl Default for PrintOptions {
    fn default() -> PrintOptions {
        PrintOptions {
            width: None,
            style: None,
            end: "\n".to_owned(),
            justify: None,
            overflow: None,
            no_wrap: None,
            markup: None,
            highlight: None,
            emoji: None,
            soft_wrap: None,
            crop: true,
        }
    }
}

/// `Console.render_str(text, ...)` keyword arguments.
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderStr<'a> {
    pub style: Style,
    pub justify: Option<Justify>,
    pub overflow: Option<Overflow>,
    pub emoji: Option<bool>,
    pub markup: Option<bool>,
    pub highlight: Option<bool>,
    pub highlighter: Option<&'a RegexHighlighter>,
}

/// A plain `str` as Rich renders it (`Console.render(str)`): markup + emoji, highlighted
/// per `options.highlight` (`None` = the console default).
#[derive(Debug, Clone, Copy)]
pub struct RichStr<'a>(pub &'a str);

impl Renderable for RichStr<'_> {
    fn render(&self, console: &Console, options: &RenderOptions) -> Vec<Segment> {
        let text = console.render_str(
            self.0,
            RenderStr {
                highlight: options.highlight,
                markup: options.markup,
                ..RenderStr::default()
            },
        );
        text.render(console, options)
    }

    fn measure(&self, console: &Console, options: &RenderOptions) -> Measurement {
        let text = console.render_str(
            self.0,
            RenderStr {
                markup: options.markup,
                highlight: Some(false),
                ..RenderStr::default()
            },
        );
        text.measure_text()
    }
}

#[derive(Default)]
struct ConsoleState {
    capture: Option<String>,
    /// Output of an in-memory console.
    output: String,
    quiet: bool,
}

/// A Rich console bound to stdout, stderr or memory.
pub struct Console {
    config: ConsoleConfig,
    width_override: Option<usize>,
    height_override: Option<usize>,
    color_system: Option<ColorSystem>,
    no_color: bool,
    interactive: bool,
    state: Mutex<ConsoleState>,
    /// Serializes whole writes (prints vs. the spinner thread).
    write_lock: Mutex<()>,
    pub(crate) live: Mutex<Option<Arc<super::status::LiveShared>>>,
}

impl std::fmt::Debug for Console {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Console")
            .field("stream", &self.config.stream)
            .field("color_system", &self.color_system)
            .finish()
    }
}

static BROKEN_PIPE: AtomicBool = AtomicBool::new(false);

/// True once a write hit a broken pipe (Rich would have exited with status 1).
pub fn broken_pipe() -> bool {
    BROKEN_PIPE.load(Ordering::Relaxed)
}

fn isatty(fd: i32) -> bool {
    // SAFETY: isatty only inspects the descriptor.
    unsafe { libc::isatty(fd) == 1 }
}

fn terminal_size(fd: i32) -> Option<(usize, usize)> {
    // SAFETY: TIOCGWINSZ writes a winsize into the struct we own.
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        (libc::ioctl(fd, libc::TIOCGWINSZ, &mut ws) == 0)
            .then_some((ws.ws_col as usize, ws.ws_row as usize))
    }
}

/// Python `str.isdigit()` + `int()` for `COLUMNS`/`LINES`.
fn digits(value: Option<String>) -> Option<usize> {
    let value = value?;
    if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()) {
        Some(value.parse().unwrap_or(usize::MAX))
    } else {
        None
    }
}

/// Rich `_detect_color_system` given `is_terminal` and `is_dumb_terminal`.
pub fn detect_color_system(
    env: &dyn Fn(&str) -> Option<String>,
    is_terminal: bool,
    is_dumb: bool,
) -> Option<ColorSystem> {
    if !is_terminal || is_dumb {
        return None;
    }
    let color_term = env("COLORTERM").unwrap_or_default().trim().to_lowercase();
    if color_term == "truecolor" || color_term == "24bit" {
        return Some(ColorSystem::TrueColor);
    }
    let term = env("TERM").unwrap_or_default().trim().to_lowercase();
    let colors = term.rsplit_once('-').map_or(term.as_str(), |(_, c)| c);
    Some(match colors {
        "kitty" | "256color" => ColorSystem::EightBit,
        _ => ColorSystem::Standard,
    })
}

impl Console {
    /// `Console(...)`.
    pub fn new(config: ConsoleConfig) -> Console {
        let mut console = Console {
            width_override: None,
            height_override: None,
            color_system: None,
            no_color: false,
            interactive: false,
            state: Mutex::new(ConsoleState::default()),
            write_lock: Mutex::new(()),
            live: Mutex::new(None),
            config,
        };
        console.width_override = console
            .config
            .width
            .or_else(|| digits(console.env("COLUMNS")));
        console.height_override = console
            .config
            .height
            .or_else(|| digits(console.env("LINES")));
        let fd = console
            .config
            .stream
            .or(console.config.detect_stream)
            .map(Stream::fd);
        let is_terminal = console.is_terminal_fd(fd);
        let is_dumb = is_terminal && console.term_is_dumb();
        console.color_system = match console.config.color_system {
            ColorChoice::Auto => detect_color_system(&|k| console.env(k), is_terminal, is_dumb),
            ColorChoice::Fixed(system) => system,
        };
        console.no_color = console
            .config
            .no_color
            .unwrap_or_else(|| console.env("NO_COLOR").is_some_and(|v| !v.is_empty()));
        let force_interactive = console.config.force_interactive.or_else(|| {
            match console.env("TTY_INTERACTIVE").as_deref() {
                Some("0") => Some(false),
                Some("1") => Some(true),
                _ => None,
            }
        });
        console.interactive = force_interactive.unwrap_or(is_terminal && !is_dumb);
        console
    }

    fn env(&self, key: &str) -> Option<String> {
        match &self.config.env {
            Some(map) => map.get(key).cloned(),
            None => std::env::var(key).ok(),
        }
    }

    fn term_is_dumb(&self) -> bool {
        let term = self.env("TERM").unwrap_or_default().to_lowercase();
        term == "dumb" || term == "unknown"
    }

    fn is_terminal_fd(&self, fd: Option<i32>) -> bool {
        if let Some(force) = self.config.force_terminal {
            return force;
        }
        match self.env("TTY_COMPATIBLE").as_deref() {
            Some("0") => return false,
            Some("1") => return true,
            _ => {}
        }
        if let Some(force_color) = self.env("FORCE_COLOR") {
            return !force_color.is_empty();
        }
        fd.is_some_and(isatty)
    }

    pub fn stream(&self) -> Option<Stream> {
        self.config.stream
    }

    pub fn theme(&self) -> &Theme {
        &self.config.theme
    }

    pub fn tab_size(&self) -> usize {
        self.config.tab_size
    }

    /// `Console.is_terminal`.
    pub fn is_terminal(&self) -> bool {
        self.is_terminal_fd(self.config.stream.map(Stream::fd))
    }

    /// `Console.is_dumb_terminal`.
    pub fn is_dumb_terminal(&self) -> bool {
        self.is_terminal() && self.term_is_dumb()
    }

    /// `Console.is_interactive` (spinner).
    pub fn is_interactive(&self) -> bool {
        self.interactive
    }

    /// `Console.color_system` (`None` = no colour).
    pub fn color_system(&self) -> Option<ColorSystem> {
        self.color_system
    }

    /// `NO_COLOR` in effect.
    pub fn no_color(&self) -> bool {
        self.no_color
    }

    /// `Console.size` (width, height).
    pub fn size(&self) -> (usize, usize) {
        if let (Some(w), Some(h)) = (self.width_override, self.height_override) {
            return (w, h);
        }
        if self.is_dumb_terminal() {
            return (80, 25);
        }
        let mut width = None;
        let mut height = None;
        for fd in [0, 1, 2] {
            if let Some((w, h)) = terminal_size(fd) {
                width = Some(w);
                height = Some(h);
                break;
            }
        }
        if let Some(columns) = digits(self.env("COLUMNS")) {
            width = Some(columns);
        }
        if let Some(lines) = digits(self.env("LINES")) {
            height = Some(lines);
        }
        let width = width.filter(|&w| w != 0).unwrap_or(80);
        let height = height.filter(|&h| h != 0).unwrap_or(25);
        (
            self.width_override.unwrap_or(width),
            self.height_override.unwrap_or(height),
        )
    }

    /// `Console.width`.
    pub fn width(&self) -> usize {
        self.size().0
    }

    /// `Console.height`.
    pub fn height(&self) -> usize {
        self.size().1
    }

    /// Default render options for this console (`Console.options`).
    pub fn options(&self) -> RenderOptions {
        let (width, height) = self.size();
        RenderOptions::for_size(width, height, self.is_terminal())
    }

    /// `Console.get_style(name)`.
    pub fn get_style(&self, name: &str) -> Result<Style, super::style::StyleSyntaxError> {
        self.config.theme.get_style(name)
    }

    /// `Console.render(renderable, options)`.
    pub fn render(&self, renderable: &dyn Renderable, options: &RenderOptions) -> Vec<Segment> {
        if options.max_width < 1 {
            return Vec::new();
        }
        renderable.render(self, options)
    }

    /// `Console.render_lines(renderable, options, style=, pad=, new_lines=)`.
    pub fn render_lines(
        &self,
        renderable: &dyn Renderable,
        options: &RenderOptions,
        style: Option<Style>,
        pad: bool,
        new_lines: bool,
    ) -> Vec<Vec<Segment>> {
        let mut rendered = self.render(renderable, options);
        if segment::style_truthy(style) {
            rendered = segment::apply_style(rendered, style, None);
        }
        let mut lines =
            segment::split_and_crop_lines(rendered, options.max_width, style, pad, new_lines);
        if let Some(height) = options.height {
            lines.truncate(height);
            while lines.len() < height {
                let mut line = vec![Segment::new(" ".repeat(options.max_width), style)];
                if new_lines {
                    line.push(Segment::line());
                }
                lines.push(line);
            }
        }
        lines
    }

    /// `Console.render_str(text, ...)`: markup, emoji and highlighting of a plain `str`.
    pub fn render_str(&self, text: &str, o: RenderStr) -> Text {
        let emoji = o.emoji.unwrap_or(self.config.emoji);
        let markup = o.markup.unwrap_or(self.config.markup);
        let highlight = o.highlight.unwrap_or(self.config.highlight);
        let plain_text = || {
            let plain = if emoji {
                emoji_replace(text).into_owned()
            } else {
                text.to_owned()
            };
            Text::with_style(plain, o.style)
        };
        let mut rich_text = if markup {
            super::markup::render_with(text, o.style, &self.config.theme, emoji)
                .unwrap_or_else(|_| plain_text())
        } else {
            plain_text()
        };
        rich_text.justify = o.justify;
        rich_text.overflow = o.overflow;
        if highlight {
            let highlighter = o.highlighter.unwrap_or_else(|| repr_highlighter());
            let mut highlighted = Text::new(rich_text.plain());
            highlighter.highlight(&mut highlighted, &self.config.theme);
            highlighted.copy_styles(&rich_text);
            return highlighted;
        }
        rich_text
    }

    /// Rich `_render_buffer`: segments -> output string.
    pub fn render_buffer(&self, segments: &[Segment]) -> String {
        let mut out = String::new();
        let system = self.color_system;
        for segment in segments {
            match segment.style {
                Some(style) if !style.is_null() => {
                    let style = if self.no_color && system.is_some() {
                        style.without_color()
                    } else {
                        style
                    };
                    style.render_into(&segment.text, system, &mut out);
                }
                _ => out.push_str(&segment.text),
            }
        }
        out
    }

    /// `print(renderable)`.
    pub fn print(&self, renderable: &dyn Renderable) {
        self.print_with(renderable, PrintOptions::default())
    }

    /// `print(renderable, width=..., style=..., end=..., ...)`.
    pub fn print_with(&self, renderable: &dyn Renderable, options: PrintOptions) {
        match renderable.as_text() {
            Some(text) => {
                let joined = self.collect_text(text.clone(), &options);
                self.print_collected(&joined, &options);
            }
            None => self.print_collected(renderable, &options),
        }
    }

    /// `print(some_str, style=...)`: markup + ReprHighlighter + emoji.
    pub fn print_str(&self, s: &str, style: Option<&str>) {
        self.print_str_with(
            s,
            PrintOptions {
                style: style.map(|d| self.config.theme.get_style_or_null(d)),
                ..PrintOptions::default()
            },
        )
    }

    /// `print(some_str, **options)`.
    pub fn print_str_with(&self, s: &str, options: PrintOptions) {
        let highlight = options.highlight.unwrap_or(self.config.highlight);
        let text = self.render_str(
            s,
            RenderStr {
                emoji: options.emoji,
                markup: options.markup,
                highlight: Some(highlight),
                ..RenderStr::default()
            },
        );
        let joined = self.collect_text(text, &options);
        self.print_collected(&joined, &options);
    }

    /// `print()`: an empty line.
    pub fn newline(&self) {
        self.print_collected(&NewLine, &PrintOptions::default());
    }

    /// Rich `_collect_renderables` for text: `Text(sep, justify=, end=).join([text])`.
    fn collect_text(&self, text: Text, options: &PrintOptions) -> Text {
        let mut sep = Text::new(" ");
        sep.justify = options.justify;
        sep.end = options.end.clone();
        sep.join([text])
    }

    fn print_collected(&self, renderable: &dyn Renderable, options: &PrintOptions) {
        let soft_wrap = options.soft_wrap.unwrap_or(self.config.soft_wrap);
        let (mut no_wrap, mut overflow, mut crop) =
            (options.no_wrap, options.overflow, options.crop);
        if soft_wrap {
            no_wrap = no_wrap.or(Some(true));
            overflow = overflow.or(Some(Overflow::Ignore));
            crop = false;
        }
        let mut render_options = self.options();
        render_options.justify = options.justify;
        render_options.overflow = overflow;
        if let Some(width) = options.width {
            let width = width.min(self.width());
            render_options.min_width = width;
            render_options.max_width = width;
        }
        if no_wrap.is_some() {
            render_options.no_wrap = no_wrap;
        }
        render_options.markup = options.markup;
        render_options.highlight = options.highlight;
        let _guard = self.write_lock.lock().unwrap();
        let mut segments = self.render(renderable, &render_options);
        let live = self.live.lock().unwrap().clone();
        let mut prefix = String::new();
        if let Some(live) = live.as_ref().filter(|_| self.interactive) {
            if self.is_terminal() {
                prefix = live.position_cursor();
            }
            segments.extend(live.render(self, &render_options));
        }
        self.emit_segments(prefix, segments, options.style, crop);
    }

    fn emit_segments(
        &self,
        prefix: String,
        mut segments: Vec<Segment>,
        style: Option<Style>,
        crop: bool,
    ) {
        if let Some(style) = style {
            let mut styled = Vec::with_capacity(segments.len());
            for (line, terminated) in segment::split_lines_terminator(segments) {
                styled.extend(segment::apply_style(line, Some(style), None));
                if terminated {
                    styled.push(Segment::line());
                }
            }
            segments = styled;
        }
        if crop {
            let width = self.width();
            segments = segment::split_and_crop_lines(segments, width, None, false, true)
                .into_iter()
                .flatten()
                .collect();
        }
        let mut out = prefix;
        out.push_str(&self.render_buffer(&segments));
        self.write_raw(&out);
    }

    /// Spinner refresh (Rich `Live.refresh` -> `console.print(Control())`).
    pub(crate) fn refresh_live(&self, live: &super::status::LiveShared) {
        let _guard = self.write_lock.lock().unwrap();
        if !self.interactive {
            return;
        }
        let prefix = if self.is_terminal() {
            live.position_cursor()
        } else {
            String::new()
        };
        let segments = live.render(self, &self.options());
        self.emit_segments(prefix, segments, None, true);
    }

    /// Run `f` while holding this console's write lock.
    pub(crate) fn locked<R>(&self, f: impl FnOnce() -> R) -> R {
        let _guard = self.write_lock.lock().unwrap();
        f()
    }

    pub(crate) fn emit(&self, prefix: String, segments: Vec<Segment>) {
        self.emit_segments(prefix, segments, None, true);
    }

    /// Write already-rendered text (or append to the capture buffer). Used by `run_watched`
    /// to replay captured frames (`sys.stdout.write(capture.get())`).
    pub fn write_raw(&self, s: &str) {
        if s.is_empty() {
            return;
        }
        {
            let mut state = self.state.lock().unwrap();
            if state.quiet {
                return;
            }
            if let Some(capture) = state.capture.as_mut() {
                capture.push_str(s);
                return;
            }
            if self.config.stream.is_none() {
                state.output.push_str(s);
                return;
            }
        }
        self.write_stream(s);
    }

    fn write_stream(&self, s: &str) {
        let result = match self.config.stream {
            Some(Stream::Stdout) => {
                let mut out = std::io::stdout().lock();
                out.write_all(s.as_bytes()).and_then(|_| out.flush())
            }
            Some(Stream::Stderr) => {
                let mut err = std::io::stderr().lock();
                err.write_all(s.as_bytes()).and_then(|_| err.flush())
            }
            None => Ok(()),
        };
        if let Err(error) = result
            && error.kind() == std::io::ErrorKind::BrokenPipe
        {
            self.on_broken_pipe();
        }
    }

    /// Rich `on_broken_pipe`: go quiet and point stdout at /dev/null.
    fn on_broken_pipe(&self) {
        self.state.lock().unwrap().quiet = true;
        BROKEN_PIPE.store(true, Ordering::Relaxed);
        // SAFETY: plain descriptor juggling on our own process.
        unsafe {
            let devnull = libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY);
            if devnull >= 0 {
                libc::dup2(devnull, 1);
                libc::close(devnull);
            }
        }
    }

    /// Write control codes (Rich `Console.control`): only on a non-dumb terminal.
    pub fn control(&self, codes: &str) {
        if self.is_terminal() && !self.is_dumb_terminal() {
            self.write_raw(codes);
        }
    }

    /// Start capturing (`Console.begin_capture`).
    pub fn begin_capture(&self) {
        self.state.lock().unwrap().capture = Some(String::new());
    }

    /// Stop capturing and return what was printed meanwhile (`Console.end_capture`).
    pub fn end_capture(&self) -> String {
        self.state
            .lock()
            .unwrap()
            .capture
            .take()
            .unwrap_or_default()
    }

    /// Everything an in-memory console (`stream: None`) printed so far.
    pub fn take_output(&self) -> String {
        std::mem::take(&mut self.state.lock().unwrap().output)
    }

    /// `Console.clear()`: `ESC[2J ESC[H` on a terminal, nothing otherwise.
    pub fn clear(&self) {
        self.control("\x1b[2J\x1b[H");
    }

    /// `Console.show_cursor(show)`.
    pub fn show_cursor(&self, show: bool) -> bool {
        if self.is_terminal() {
            self.control(if show { "\x1b[?25h" } else { "\x1b[?25l" });
            return true;
        }
        false
    }

    /// `Console.status(text)`: transient spinner while the returned guard lives.
    pub fn status(&'static self, text: Text) -> super::status::Status {
        super::status::Status::start(self, text)
    }
}

/// Rich `NewLine` (what `print()` prints).
struct NewLine;

impl Renderable for NewLine {
    fn render(&self, _console: &Console, _options: &RenderOptions) -> Vec<Segment> {
        vec![Segment::line()]
    }
}

static STDOUT: OnceLock<Console> = OnceLock::new();
static STDERR: OnceLock<Console> = OnceLock::new();

/// The stdout console (Python `core.CONSOLE`).
pub fn stdout() -> &'static Console {
    STDOUT.get_or_init(|| Console::new(ConsoleConfig::default()))
}

/// The stderr console (Python `core.ERROR_CONSOLE`).
pub fn stderr() -> &'static Console {
    STDERR.get_or_init(|| {
        Console::new(ConsoleConfig {
            stream: Some(Stream::Stderr),
            ..ConsoleConfig::default()
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn console(env: &[(&str, &str)], force_terminal: Option<bool>) -> Console {
        Console::new(ConsoleConfig {
            stream: None,
            force_terminal,
            env: Some(
                env.iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            ),
            ..ConsoleConfig::default()
        })
    }

    #[test]
    fn detection() {
        assert_eq!(console(&[], None).color_system(), None);
        assert_eq!(
            console(&[("FORCE_COLOR", "1")], None).color_system(),
            Some(ColorSystem::Standard)
        );
        assert_eq!(
            console(&[("FORCE_COLOR", "")], Some(true)).color_system(),
            Some(ColorSystem::Standard)
        );
        assert_eq!(console(&[("FORCE_COLOR", "")], None).color_system(), None);
        assert_eq!(
            console(&[("FORCE_COLOR", "1"), ("TERM", "xterm-256color")], None).color_system(),
            Some(ColorSystem::EightBit)
        );
        assert_eq!(
            console(&[("FORCE_COLOR", "1"), ("COLORTERM", "truecolor")], None).color_system(),
            Some(ColorSystem::TrueColor)
        );
        assert_eq!(
            console(&[("FORCE_COLOR", "1"), ("TERM", "dumb")], None).color_system(),
            None
        );
        assert_eq!(
            console(&[("TTY_COMPATIBLE", "0"), ("FORCE_COLOR", "1")], None).color_system(),
            None
        );
        assert!(console(&[("NO_COLOR", "1")], None).no_color());
        assert!(!console(&[("NO_COLOR", "")], None).no_color());
        assert!(console(&[("FORCE_COLOR", "1")], None).is_interactive());
        assert!(!console(&[("FORCE_COLOR", "1"), ("TTY_INTERACTIVE", "0")], None).is_interactive());
    }

    #[test]
    fn width_from_columns() {
        assert_eq!(console(&[("COLUMNS", "123")], None).width(), 123);
        assert_eq!(
            console(
                &[("COLUMNS", "50"), ("FORCE_COLOR", "1"), ("TERM", "dumb")],
                None
            )
            .width(),
            80
        );
        assert_eq!(
            console(&[("FORCE_COLOR", "1"), ("TERM", "dumb")], None).width(),
            80
        );
    }

    #[test]
    fn print_and_capture() {
        let c = console(&[("FORCE_COLOR", "1"), ("COLUMNS", "20")], None);
        c.print(&Text::styled("hi", "bold"));
        assert_eq!(c.take_output(), "\x1b[1mhi\x1b[0m\n");
        c.print_str("Canceled 12.", None);
        assert_eq!(c.take_output(), "Canceled \x1b[1;36m12\x1b[0m.\n");
        c.begin_capture();
        c.newline();
        assert_eq!(c.end_capture(), "\n");
        c.clear();
        assert_eq!(c.take_output(), "\x1b[2J\x1b[H");
    }
}
