//! Rich `Console.status` (rich/status.py, live.py, live_render.py, spinner.py): a transient
//! `dots` spinner refreshed 12.5 times a second.
//!
//! On a terminal the cursor is hidden while the status runs; on an interactive terminal
//! every refresh (and every print on the same console) first moves back over the previous
//! spinner lines (`\r ESC[2K`, `ESC[1A ESC[2K` per extra line), prints, then redraws the
//! spinner. Stopping redraws it once more, ends the line, shows the cursor and erases the
//! spinner line(s). Off a terminal nothing at all is written. Like Rich, other consoles
//! (stderr) are not coordinated with the spinner.

use super::data::{DOTS_FRAMES, DOTS_INTERVAL_MS};
use super::segment::{self, Segment};
use super::text::{Text, TextPart};
use super::theme;
use super::{Console, RenderOptions};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

struct LiveState {
    text: Text,
    start: Option<Instant>,
    shape: Option<(usize, usize)>,
    visible_overflow: bool,
}

/// The renderable a running status shows (Rich `Live` + `LiveRender` + `Spinner`).
pub struct LiveShared {
    state: Mutex<LiveState>,
}

impl LiveShared {
    /// Rich `LiveRender.position_cursor()`.
    pub(crate) fn position_cursor(&self) -> String {
        match self.state.lock().unwrap().shape {
            Some((_, height)) => {
                let mut out = String::from("\r\x1b[2K");
                for _ in 1..height {
                    out.push_str("\x1b[1A\x1b[2K");
                }
                out
            }
            None => String::new(),
        }
    }

    /// Rich `LiveRender.restore_cursor()`.
    fn restore_cursor(&self) -> String {
        match self.state.lock().unwrap().shape {
            Some((_, height)) => {
                let mut out = String::from("\r");
                for _ in 0..height {
                    out.push_str("\x1b[1A\x1b[2K");
                }
                out
            }
            None => String::new(),
        }
    }

    fn last_render_height(&self) -> usize {
        self.state.lock().unwrap().shape.map_or(0, |(_, h)| h)
    }

    /// Rich `Spinner.render(time)`: `frame + " " + text`.
    fn spinner_text(state: &mut LiveState) -> Text {
        let now = Instant::now();
        let start = *state.start.get_or_insert(now);
        let frame_no = now.duration_since(start).as_secs_f64() / (DOTS_INTERVAL_MS / 1000.0);
        let frame = DOTS_FRAMES[(frame_no as usize) % DOTS_FRAMES.len()];
        let frame = Text::with_style(frame, theme::style("status.spinner"));
        if state.text.is_empty() {
            return frame;
        }
        Text::assemble([
            TextPart::Text(frame),
            " ".into(),
            TextPart::Text(state.text.clone()),
        ])
    }

    /// Rich `LiveRender.__rich_console__`: the spinner lines joined by `\n` (no trailing).
    pub(crate) fn render(&self, console: &Console, options: &RenderOptions) -> Vec<Segment> {
        let mut state = self.state.lock().unwrap();
        let text = Self::spinner_text(&mut state);
        let mut lines =
            console.render_lines(&text, options, Some(super::Style::NULL), false, false);
        let mut shape = segment::get_shape(&lines);
        let height = options.size.1;
        if shape.1 > height && !state.visible_overflow {
            lines.truncate(height.saturating_sub(1));
            let mut ellipsis = Text::with_style("...", theme::style("live.ellipsis"));
            ellipsis.overflow = Some(super::Overflow::Crop);
            ellipsis.justify = Some(super::Justify::Center);
            ellipsis.end = String::new();
            lines.push(console.render(&ellipsis, options));
            shape = segment::get_shape(&lines);
        }
        state.shape = Some(shape);
        let count = lines.len();
        let mut out = Vec::new();
        for (i, line) in lines.into_iter().enumerate() {
            out.extend(line);
            if i + 1 < count {
                out.push(Segment::line());
            }
        }
        out
    }
}

/// A running `Console.status` spinner; stops (and erases itself) on drop.
pub struct Status {
    console: &'static Console,
    shared: Option<Arc<LiveShared>>,
    stop: Arc<(Mutex<bool>, Condvar)>,
    thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Status")
            .field("active", &self.shared.is_some())
            .finish()
    }
}

/// Rich `Live.refresh()`.
fn refresh(console: &Console, shared: &LiveShared) {
    if console.is_terminal() && !console.is_dumb_terminal() {
        console.refresh_live(shared);
    }
}

impl Status {
    /// `with console.status(text):` (Rich `Live.start()`).
    pub(crate) fn start(console: &'static Console, text: Text) -> Status {
        Self::start_with_refresh(console, text, Some(Duration::from_secs_f64(1.0 / 12.5)))
    }

    /// `refresh: None` disables the background refresh (deterministic tests).
    fn start_with_refresh(
        console: &'static Console,
        text: Text,
        refresh_every: Option<Duration>,
    ) -> Status {
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let shared = Arc::new(LiveShared {
            state: Mutex::new(LiveState {
                text,
                start: None,
                shape: None,
                visible_overflow: false,
            }),
        });
        {
            let mut live = console.live.lock().unwrap();
            if live.is_some() {
                return Status {
                    console,
                    shared: None,
                    stop,
                    thread: None,
                };
            }
            *live = Some(shared.clone());
        }
        // `Status.__enter__` starts `Live` without an initial refresh: the first frame shows
        // after one interval or with the next print.
        console.show_cursor(false);
        let thread = refresh_every.map(|interval| {
            let shared = shared.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                let (lock, cvar) = &*stop;
                let mut stopped = lock.lock().unwrap();
                loop {
                    let (guard, _) = cvar.wait_timeout(stopped, interval).unwrap();
                    stopped = guard;
                    if *stopped {
                        return;
                    }
                    drop(stopped);
                    refresh(console, &shared);
                    stopped = lock.lock().unwrap();
                }
            })
        });
        Status {
            console,
            shared: Some(shared),
            stop,
            thread,
        }
    }

    /// `status.update(text)` (an empty text keeps the old one, like Rich).
    pub fn update(&self, text: Text) {
        if let Some(shared) = &self.shared
            && !text.is_empty()
        {
            shared.state.lock().unwrap().text = text;
        }
    }

    /// Rich `Live.stop()`.
    fn stop(&mut self) {
        let Some(shared) = self.shared.take() else {
            return;
        };
        {
            let (lock, cvar) = &*self.stop;
            *lock.lock().unwrap() = true;
            cvar.notify_all();
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        let console = self.console;
        shared.state.lock().unwrap().visible_overflow = true;
        console.locked(|| {
            let mut out = String::new();
            let mut segments = Vec::new();
            if console.is_terminal() && !console.is_dumb_terminal() && console.is_interactive() {
                out = shared.position_cursor();
                segments = shared.render(console, &console.options());
            }
            *console.live.lock().unwrap() = None;
            if console.is_terminal() && shared.last_render_height() > 0 {
                segments.push(Segment::line());
            }
            let mut rendered = out;
            rendered.push_str(
                &console.render_buffer(
                    &segment::split_and_crop_lines(segments, console.width(), None, false, true)
                        .concat(),
                ),
            );
            if console.is_terminal() && !console.is_dumb_terminal() {
                rendered.push_str("\x1b[?25h");
                rendered.push_str(&shared.restore_cursor());
            }
            console.write_raw(&rendered);
        });
    }
}

impl Drop for Status {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::console::{ColorChoice, ConsoleConfig};
    use crate::render::style::ColorSystem;

    fn console(env: &[(&str, &str)]) -> &'static Console {
        Box::leak(Box::new(Console::new(ConsoleConfig {
            stream: None,
            force_terminal: Some(true),
            color_system: ColorChoice::Fixed(Some(ColorSystem::Standard)),
            width: Some(30),
            height: Some(25),
            env: Some(
                env.iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            ),
            ..ConsoleConfig::default()
        })))
    }

    fn run(console: &'static Console) -> String {
        let status = Status::start_with_refresh(
            console,
            Text::assemble([
                "Waiting for ".into(),
                TextPart::Styled("123".into(), theme::style("cyan")),
            ]),
            None,
        );
        console.print_str("inside 1", None);
        status.update(Text::new("changed"));
        drop(status);
        console.take_output()
    }

    // Expected bytes from rich 15 (a status that ends within one refresh interval).
    #[test]
    fn matches_rich() {
        let out = run(console(&[]));
        assert_eq!(
            out,
            "\x1b[?25linside \x1b[1;36m1\x1b[0m\n\x1b[32m⠋\x1b[0m Waiting for \x1b[36m123\x1b[0m\r\x1b[2K\x1b[32m⠋\x1b[0m changed\n\x1b[?25h\r\x1b[1A\x1b[2K"
        );
        assert_eq!(
            run(console(&[("TTY_INTERACTIVE", "0")])),
            "\x1b[?25linside \x1b[1;36m1\x1b[0m\n\x1b[?25h"
        );
        assert_eq!(
            run(console(&[("TERM", "dumb")])),
            "inside \x1b[1;36m1\x1b[0m\n"
        );
    }
}
