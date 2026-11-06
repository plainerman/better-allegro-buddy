//! Rich `Padding` (rich/padding.py); tables wrap every cell in one.

use super::style::Style;
use super::{Console, Measurement, RenderOptions, Renderable, Segment};

/// Rich `Padding(renderable, pad, style=, expand=)`.
pub struct Padding<'a> {
    pub renderable: &'a dyn Renderable,
    pub top: usize,
    pub right: usize,
    pub bottom: usize,
    pub left: usize,
    pub style: Style,
    pub expand: bool,
}

impl<'a> Padding<'a> {
    /// `Padding(renderable, (top, right, bottom, left))`.
    pub fn new(
        renderable: &'a dyn Renderable,
        (top, right, bottom, left): (usize, usize, usize, usize),
    ) -> Self {
        Padding {
            renderable,
            top,
            right,
            bottom,
            left,
            style: Style::NULL,
            expand: true,
        }
    }

    /// Rich `Padding.unpack`: 1, 2 or 4 values -> (top, right, bottom, left).
    pub fn unpack(pad: &[usize]) -> (usize, usize, usize, usize) {
        match *pad {
            [p] => (p, p, p, p),
            [v, h] => (v, h, v, h),
            [t, r, b, l] => (t, r, b, l),
            _ => panic!(
                "1, 2 or 4 integers required for padding; {} given",
                pad.len()
            ),
        }
    }
}

impl Renderable for Padding<'_> {
    fn render(&self, console: &Console, options: &RenderOptions) -> Vec<Segment> {
        let style = self.style;
        let width = if self.expand {
            options.max_width
        } else {
            (Measurement::get(console, options, self.renderable).maximum + self.left + self.right)
                .min(options.max_width)
        };
        let mut render_options =
            options.update_width(width as isize - self.left as isize - self.right as isize);
        if let Some(height) = render_options.height {
            render_options =
                render_options.update_height(height.saturating_sub(self.top + self.bottom));
        }
        let lines =
            console.render_lines(self.renderable, &render_options, Some(style), true, false);
        let mut out = Vec::new();
        let blank = || Segment::styled(format!("{}\n", " ".repeat(width)), style);
        for _ in 0..self.top {
            out.push(blank());
        }
        for line in lines {
            if self.left > 0 {
                out.push(Segment::styled(" ".repeat(self.left), style));
            }
            out.extend(line);
            if self.right > 0 {
                out.push(Segment::styled(" ".repeat(self.right), style));
            }
            out.push(Segment::line());
        }
        for _ in 0..self.bottom {
            out.push(blank());
        }
        out
    }

    fn measure(&self, console: &Console, options: &RenderOptions) -> Measurement {
        let max_width = options.max_width;
        let extra = self.left + self.right;
        if max_width as isize - (extra as isize) < 1 {
            return Measurement::new(max_width, max_width);
        }
        let m = Measurement::get(console, options, self.renderable);
        Measurement::new(m.minimum + extra, m.maximum + extra).with_maximum(max_width)
    }
}
