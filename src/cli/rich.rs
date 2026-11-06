//! Rich renderables used only by the rich-click help/error layout (Rich 15.0.0 `Group`,
//! `Constrain`, `Align`, `Panel`, `Columns`, `Padding` with an owned child). Owned by P2-B;
//! the primitives (Text, Table, Segment, Console) live in [`crate::render`].

use crate::render::boxes::BoxStyle;
use crate::render::segment::{self, Segment};
use crate::render::style::Style;
use crate::render::table::{Cell, Table};
use crate::render::text::{Text, TextPart};
use crate::render::{Console, Measurement, Padding, RenderOptions, Renderable};
use std::rc::Rc;

pub type Node = Rc<dyn Renderable>;

/// Rich `measure_renderables`.
fn measure_all(
    console: &Console,
    options: &RenderOptions,
    items: &[&dyn Renderable],
) -> Measurement {
    if items.is_empty() {
        return Measurement::new(0, 0);
    }
    let measured: Vec<Measurement> = items
        .iter()
        .map(|r| Measurement::get(console, options, *r))
        .collect();
    Measurement::new(
        measured.iter().map(|m| m.minimum).max().unwrap_or(0),
        measured.iter().map(|m| m.maximum).max().unwrap_or(0),
    )
}

/// Rich `Group(*renderables, fit=True)`.
pub struct Group(pub Vec<Node>);

impl Renderable for Group {
    fn render(&self, console: &Console, options: &RenderOptions) -> Vec<Segment> {
        self.0
            .iter()
            .flat_map(|r| console.render(r.as_ref(), options))
            .collect()
    }

    fn measure(&self, console: &Console, options: &RenderOptions) -> Measurement {
        let items: Vec<&dyn Renderable> = self.0.iter().map(|r| r.as_ref()).collect();
        measure_all(console, options, &items)
    }
}

/// Rich `Constrain(renderable, width)`.
struct Constrain<'a> {
    renderable: &'a dyn Renderable,
    width: usize,
}

impl Renderable for Constrain<'_> {
    fn render(&self, console: &Console, options: &RenderOptions) -> Vec<Segment> {
        let child = options.update_width(self.width.min(options.max_width) as isize);
        console.render(self.renderable, &child)
    }

    fn measure(&self, console: &Console, options: &RenderOptions) -> Measurement {
        let options = options.update_width(self.width as isize);
        Measurement::get(console, &options, self.renderable)
    }
}

/// Rich `Align(renderable, "left", pad=False)` (the only form rich-click uses).
pub struct AlignLeft(pub Node);

impl Renderable for AlignLeft {
    fn render(&self, console: &Console, options: &RenderOptions) -> Vec<Segment> {
        let width = Measurement::get(console, options, self.0.as_ref()).maximum;
        let constrained = Constrain {
            renderable: self.0.as_ref(),
            width,
        };
        let rendered = console.render(&constrained, &options.with_height(None));
        let lines = segment::split_lines(rendered);
        let (width, height) = segment::get_shape(&lines);
        let lines = segment::set_shape(lines, width, Some(height), None);
        let mut out = Vec::new();
        for line in lines {
            out.extend(line);
            out.push(Segment::line());
        }
        out
    }

    fn measure(&self, console: &Console, options: &RenderOptions) -> Measurement {
        Measurement::get(console, options, self.0.as_ref())
    }
}

/// Rich `Padding` owning its child (so it can live in table cells and groups).
pub struct Padded {
    pub child: Node,
    pub pad: (usize, usize, usize, usize),
    pub style: Style,
}

impl Padded {
    pub fn new(child: Node, pad: (usize, usize, usize, usize), style: Style) -> Padded {
        Padded { child, pad, style }
    }

    fn padding(&self) -> Padding<'_> {
        let mut padding = Padding::new(self.child.as_ref(), self.pad);
        padding.style = self.style;
        padding
    }
}

impl Renderable for Padded {
    fn render(&self, console: &Console, options: &RenderOptions) -> Vec<Segment> {
        self.padding().render(console, options)
    }

    fn measure(&self, console: &Console, options: &RenderOptions) -> Measurement {
        self.padding().measure(console, options)
    }
}

/// Rich `Panel` (+ rich-click's `RichClickRichPanel` title handling).
pub struct Panel {
    pub renderable: Node,
    pub box_style: BoxStyle,
    /// Already prepared title (`Panel._title`), `None` for no title.
    pub title: Option<Text>,
    /// `left`, `center` or `right`.
    pub title_align: String,
    pub expand: bool,
    pub style: Style,
    pub border_style: Style,
    pub padding: (usize, usize, usize, usize),
    /// rich-click's `BLANK` box: `top`, `top_left`, `top_divider` are `""` and
    /// `top_right` is 800 spaces (cropped by the console).
    pub blank_top: bool,
}

impl Panel {
    /// Rich `Panel._title` for a plain Panel (`title_text.pad(1)`).
    pub fn rich_title(title: Text) -> Text {
        let mut title = Self::base_title(title);
        title.pad(1, ' ');
        title
    }

    /// rich-click `RichClickRichPanel._title` with `title_padding`.
    pub fn rich_click_title(title: Text, title_padding: usize) -> Text {
        let title = Self::base_title(title);
        if title_padding == 0 {
            return title;
        }
        let pad = Text::new(" ".repeat(title_padding));
        Text::new("").join([pad.clone(), title, pad])
    }

    fn base_title(mut title: Text) -> Text {
        title.end = String::new();
        let plain = title.plain().replace('\n', " ");
        title.set_plain(&plain);
        title.no_wrap = Some(true);
        title.expand_tabs(None);
        title
    }

    fn align_text(&self, console: &Console, title: &Text, width: usize, character: char) -> Text {
        let _ = console;
        let mut text = title.clone();
        text.truncate(width, None, false);
        let excess = width.saturating_sub(text.cell_len());
        if !text.style.is_null() {
            let style = text.style;
            text.stylize(style, 0, None);
        }
        if excess == 0 {
            return text;
        }
        let fill = |n: usize| {
            let fill = if self.blank_top {
                String::new()
            } else {
                character.to_string().repeat(n)
            };
            TextPart::Styled(fill, self.border_style)
        };
        let mut out = match self.title_align.as_str() {
            "center" => {
                let left = excess / 2;
                Text::assemble([fill(left), TextPart::Text(text), fill(excess - left)])
            }
            "right" => Text::assemble([fill(excess), TextPart::Text(text)]),
            _ => Text::assemble([TextPart::Text(text), fill(excess)]),
        };
        out.no_wrap = Some(true);
        out.end = String::new();
        out
    }
}

impl Renderable for Panel {
    fn render(&self, console: &Console, options: &RenderOptions) -> Vec<Segment> {
        let padded;
        let renderable: &dyn Renderable = if self.padding != (0, 0, 0, 0) {
            padded = Padding::new(self.renderable.as_ref(), self.padding);
            &padded
        } else {
            self.renderable.as_ref()
        };
        let style = self.style;
        let border_style = style.combine(&self.border_style);
        let width = options.max_width;
        let b = &self.box_style;
        let title = self.title.as_ref().map(|t| {
            let mut t = t.clone();
            t.stylize_before(border_style, 0, None);
            t
        });
        let mut child_width = if self.expand {
            width as isize - 2
        } else {
            Measurement::get(
                console,
                &options.update_width(width as isize - 2),
                renderable,
            )
            .maximum as isize
        };
        if let Some(title) = &title {
            child_width = (options.max_width as isize - 2)
                .min(child_width.max(title.cell_len() as isize + 2));
        }
        let width = child_width + 2;
        let child_options = options
            .update_width(child_width)
            .with_height(options.height.map(|h| h.saturating_sub(2)));
        let lines = console.render_lines(renderable, &child_options, Some(style), true, false);
        let mut out = Vec::new();
        let border = |s: String| Segment::styled(s, border_style);
        let blank_right = || " ".repeat(800);
        match &title {
            Some(title) if width > 4 && self.blank_top => {
                let title = self.align_text(console, title, (width - 4) as usize, b.top);
                out.extend(console.render(&title, &child_options.update_width(width - 4)));
                out.push(border(blank_right()));
            }
            _ if self.blank_top => out.push(border(blank_right())),
            Some(title) if width > 4 => {
                let title = self.align_text(console, title, (width - 4) as usize, b.top);
                out.push(border(format!("{}{}", b.top_left, b.top)));
                out.extend(console.render(&title, &child_options.update_width(width - 4)));
                out.push(border(format!("{}{}", b.top, b.top_right)));
            }
            _ => out.push(border(b.get_top(&[(width - 2).max(0) as usize]))),
        }
        out.push(Segment::line());
        for line in lines {
            out.push(border(b.mid_left.to_string()));
            out.extend(line);
            out.push(border(b.mid_right.to_string()));
            out.push(Segment::line());
        }
        out.push(border(b.get_bottom(&[(width - 2).max(0) as usize])));
        out.push(Segment::line());
        out
    }

    fn measure(&self, console: &Console, options: &RenderOptions) -> Measurement {
        let (_, right, _, left) = self.padding;
        let padding = left + right;
        let child_options = options.update_width(options.max_width as isize - padding as isize - 2);
        let mut items: Vec<&dyn Renderable> = vec![self.renderable.as_ref()];
        if let Some(title) = &self.title {
            items.push(title);
        }
        let width = measure_all(console, &child_options, &items).maximum + padding + 2;
        Measurement::new(width, width)
    }
}

/// Rich `Columns(renderables)` with the defaults rich-click uses (padding (0, 1), no
/// expand, row-first).
pub struct Columns(pub Vec<Node>);

impl Renderable for Columns {
    fn render(&self, console: &Console, options: &RenderOptions) -> Vec<Segment> {
        if self.0.is_empty() {
            return Vec::new();
        }
        let width_padding = 1;
        let max_width = options.max_width;
        let widths_of: Vec<usize> = self
            .0
            .iter()
            .map(|r| Measurement::get(console, options, r.as_ref()).maximum)
            .collect();
        let count = self.0.len();
        let mut column_count = count;
        while column_count > 1 {
            let mut widths: Vec<usize> = Vec::new();
            let mut column_no = 0;
            let mut shrunk = false;
            let padded = widths_of.iter().copied().chain(std::iter::repeat_n(
                0,
                (column_count - count % column_count) % column_count,
            ));
            for width in padded {
                if widths.len() <= column_no {
                    widths.push(0);
                }
                widths[column_no] = widths[column_no].max(width);
                let total: usize =
                    widths.iter().sum::<usize>() + width_padding * (widths.len() - 1);
                if total > max_width {
                    column_count = widths.len() - 1;
                    shrunk = true;
                    break;
                }
                column_no = (column_no + 1) % column_count;
            }
            if !shrunk {
                break;
            }
        }
        let mut cells: Vec<Cell> = self.0.iter().map(|r| Cell::Renderable(r.clone())).collect();
        if !count.is_multiple_of(column_count) {
            for _ in 0..column_count - count % column_count {
                cells.push(Cell::Str(String::new()));
            }
        }
        let mut table = Table::grid(0, 1);
        for row in cells.chunks(column_count.max(1)) {
            table.add_row(row.to_vec(), false);
        }
        console.render(&table, options)
    }
}
