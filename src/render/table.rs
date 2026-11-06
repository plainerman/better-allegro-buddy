//! Rich `Table` (rich/table.py) with Rich's column width algorithm (`_ratio.py`).
//!
//! [`Table::simple`] is `core.render_table`'s configuration (box.SIMPLE, show_edge=False,
//! pad_edge=False, bold title/header); [`Table::grid`] is `Table.grid(padding=...)`.
//! Every other Rich option used by rich-click's help panels is available as a public field.
//! Cells are [`Cell`]s: `Text` renders as is; `Str` goes through `Console.render_str` with
//! markup (and the column's `highlight`, False by default); any other renderable via
//! [`Cell::Renderable`].

use super::boxes::{BoxStyle, RowLevel, SIMPLE};
use super::console::{RenderStr, RichStr};
use super::padding::Padding;
use super::segment::{self, VerticalAlign};
use super::style::Style;
use super::text::{Overflow, Text};
use super::theme;
use super::{Console, Measurement, RenderOptions, Renderable, Segment};
use std::rc::Rc;

pub use super::text::Justify;

/// A table cell: a styled `Text`, a `str` that Rich renders with markup, or any renderable.
#[derive(Clone)]
pub enum Cell {
    Text(Text),
    Str(String),
    Renderable(Rc<dyn Renderable>),
}

impl std::fmt::Debug for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Cell::Text(t) => f.debug_tuple("Text").field(&t.plain()).finish(),
            Cell::Str(s) => f.debug_tuple("Str").field(s).finish(),
            Cell::Renderable(_) => f.write_str("Renderable(..)"),
        }
    }
}

impl Cell {
    /// Rich truthiness (`if self.title:`).
    fn is_truthy(&self) -> bool {
        match self {
            Cell::Text(t) => !t.is_empty(),
            Cell::Str(s) => !s.is_empty(),
            Cell::Renderable(_) => true,
        }
    }
}

impl Renderable for Cell {
    fn render(&self, console: &Console, options: &RenderOptions) -> Vec<Segment> {
        match self {
            Cell::Text(t) => t.render(console, options),
            Cell::Str(s) => RichStr(s).render(console, options),
            Cell::Renderable(r) => r.render(console, options),
        }
    }

    fn measure(&self, console: &Console, options: &RenderOptions) -> Measurement {
        match self {
            Cell::Text(t) => Renderable::measure(t, console, options),
            Cell::Str(s) => RichStr(s).measure(console, options),
            Cell::Renderable(r) => r.measure(console, options),
        }
    }
}

impl From<Text> for Cell {
    fn from(t: Text) -> Cell {
        Cell::Text(t)
    }
}

impl From<&str> for Cell {
    fn from(s: &str) -> Cell {
        Cell::Str(s.to_owned())
    }
}

impl From<String> for Cell {
    fn from(s: String) -> Cell {
        Cell::Str(s)
    }
}

/// Rich `Column`.
#[derive(Debug, Clone)]
pub struct Column {
    pub header: Cell,
    pub footer: Cell,
    pub header_style: Style,
    pub footer_style: Style,
    /// Column style (applied under the cell contents), e.g. `dim` in the grid.
    pub style: Style,
    pub justify: Justify,
    pub vertical: VerticalAlign,
    pub overflow: Overflow,
    pub width: Option<usize>,
    pub min_width: Option<usize>,
    pub max_width: Option<usize>,
    pub ratio: Option<usize>,
    pub no_wrap: bool,
    /// `None` = the table's `highlight` (Rich `add_column(highlight=None)`).
    pub highlight: Option<bool>,
    pub cells: Vec<Cell>,
}

impl Default for Column {
    fn default() -> Column {
        Column {
            header: Cell::Str(String::new()),
            footer: Cell::Str(String::new()),
            header_style: Style::NULL,
            footer_style: Style::NULL,
            style: Style::NULL,
            justify: Justify::Left,
            vertical: VerticalAlign::Top,
            overflow: Overflow::Ellipsis,
            width: None,
            min_width: None,
            max_width: None,
            ratio: None,
            no_wrap: false,
            highlight: None,
            cells: Vec::new(),
        }
    }
}

/// Rich `Row`: per-row style and the `end_section` flag (cells live in the columns).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Row {
    pub style: Option<Style>,
    pub end_section: bool,
}

/// Rich `Table`.
#[derive(Debug, Clone)]
pub struct Table {
    pub columns: Vec<Column>,
    pub rows: Vec<Row>,
    /// `str` titles go through markup with `title_style` as base; `Text` titles as is.
    pub title: Option<Cell>,
    pub caption: Option<Cell>,
    pub width: Option<usize>,
    pub min_width: Option<usize>,
    pub box_style: Option<BoxStyle>,
    /// (top, right, bottom, left) as Rich normalizes `padding`.
    pub padding: (usize, usize, usize, usize),
    pub collapse_padding: bool,
    pub pad_edge: bool,
    pub expand: bool,
    pub show_header: bool,
    pub show_footer: bool,
    pub show_edge: bool,
    pub show_lines: bool,
    pub leading: usize,
    pub style: Style,
    pub row_styles: Vec<Style>,
    pub header_style: Style,
    pub footer_style: Style,
    pub border_style: Style,
    /// `None` = Rich's `table.title` (italic).
    pub title_style: Option<Style>,
    /// `None` = Rich's `table.caption` (italic dim).
    pub caption_style: Option<Style>,
    pub title_justify: Justify,
    pub caption_justify: Justify,
    pub highlight: bool,
}

impl Default for Table {
    /// Rich `Table()` defaults (box HEAVY_HEAD, padding (0, 1), ...).
    fn default() -> Table {
        Table {
            columns: Vec::new(),
            rows: Vec::new(),
            title: None,
            caption: None,
            width: None,
            min_width: None,
            box_style: Some(*super::boxes::HEAVY_HEAD),
            padding: (0, 1, 0, 1),
            collapse_padding: false,
            pad_edge: true,
            expand: false,
            show_header: true,
            show_footer: false,
            show_edge: true,
            show_lines: false,
            leading: 0,
            style: Style::NULL,
            row_styles: Vec::new(),
            header_style: theme::style("table.header"),
            footer_style: theme::style("table.footer"),
            border_style: Style::NULL,
            title_style: None,
            caption_style: None,
            title_justify: Justify::Center,
            caption_justify: Justify::Center,
            highlight: false,
        }
    }
}

/// Rich `ratio_reduce(total, ratios, maximums, values)`.
pub fn ratio_reduce(
    total: isize,
    ratios: &[isize],
    maximums: &[isize],
    values: &[isize],
) -> Vec<isize> {
    let ratios: Vec<isize> = ratios
        .iter()
        .zip(maximums)
        .map(|(&r, &m)| if m != 0 { r } else { 0 })
        .collect();
    let mut total_ratio: isize = ratios.iter().sum();
    if total_ratio == 0 {
        return values.to_vec();
    }
    let mut total_remaining = total;
    let mut result = Vec::with_capacity(values.len());
    for ((&ratio, &maximum), &value) in ratios.iter().zip(maximums).zip(values) {
        if ratio != 0 && total_ratio > 0 {
            let share = (ratio as f64 * total_remaining as f64 / total_ratio as f64)
                .round_ties_even() as isize;
            let distributed = maximum.min(share);
            result.push(value - distributed);
            total_remaining -= distributed;
            total_ratio -= ratio;
        } else {
            result.push(value);
        }
    }
    result
}

/// Rich `ratio_distribute(total, ratios, minimums)`.
pub fn ratio_distribute(total: isize, ratios: &[isize], minimums: Option<&[isize]>) -> Vec<isize> {
    let ratios: Vec<isize> = match minimums {
        Some(mins) if !mins.is_empty() => ratios
            .iter()
            .zip(mins)
            .map(|(&r, &m)| if m != 0 { r } else { 0 })
            .collect(),
        _ => ratios.to_vec(),
    };
    let mut total_ratio: isize = ratios.iter().sum();
    assert!(total_ratio > 0, "Sum of ratios must be > 0");
    let mut total_remaining = total;
    let zeros = vec![0; ratios.len()];
    let minimums = minimums.unwrap_or(&zeros);
    let mut out = Vec::with_capacity(ratios.len());
    for (&ratio, &minimum) in ratios.iter().zip(minimums) {
        let distributed = if total_ratio > 0 {
            minimum
                .max((ratio as f64 * total_remaining as f64 / total_ratio as f64).ceil() as isize)
        } else {
            total_remaining
        };
        out.push(distributed);
        total_ratio -= ratio;
        total_remaining -= distributed;
    }
    out
}

/// One prepared cell (Rich `_Cell`): style, the (padded) renderable, vertical alignment.
struct CellSpec<'a> {
    style: Style,
    inner: &'a dyn Renderable,
    pad: Option<(usize, usize, usize, usize)>,
    vertical: VerticalAlign,
}

impl Renderable for CellSpec<'_> {
    fn render(&self, console: &Console, options: &RenderOptions) -> Vec<Segment> {
        match self.pad {
            Some(pad) => Padding::new(self.inner, pad).render(console, options),
            None => self.inner.render(console, options),
        }
    }

    fn measure(&self, console: &Console, options: &RenderOptions) -> Measurement {
        match self.pad {
            Some(pad) => Padding::new(self.inner, pad).measure(console, options),
            None => self.inner.measure(console, options),
        }
    }
}

impl Table {
    /// Rich `Table()` with default settings.
    pub fn new() -> Table {
        Table::default()
    }

    /// The `core.render_table` configuration:
    /// `Table(title=title, title_style="bold", box=box.SIMPLE, show_edge=False,
    /// pad_edge=False, header_style="bold")`.
    pub fn simple(title: Option<&str>) -> Table {
        Table {
            title: title.map(Cell::from),
            title_style: Some(theme::style("bold")),
            box_style: Some(*SIMPLE),
            show_edge: false,
            pad_edge: false,
            header_style: theme::style("bold"),
            ..Table::default()
        }
    }

    /// `Table.grid(padding=(vertical, horizontal))` (no box/header/edge, collapse_padding).
    pub fn grid(vertical: usize, horizontal: usize) -> Table {
        Table {
            box_style: None,
            padding: (vertical, horizontal, vertical, horizontal),
            collapse_padding: true,
            show_header: false,
            show_footer: false,
            show_edge: false,
            pad_edge: false,
            expand: false,
            ..Table::default()
        }
    }

    /// `add_column(header, justify=justify)`.
    pub fn add_column(&mut self, header: &str, justify: Justify) -> &mut Table {
        self.add_column_spec(Column {
            header: Cell::from(header),
            justify,
            ..Column::default()
        })
    }

    /// `add_column(**spec)`, e.g. the grid's `style="dim", no_wrap=True` column.
    pub fn add_column_spec(&mut self, column: Column) -> &mut Table {
        self.columns.push(column);
        self
    }

    /// `add_row(*cells, end_section=end_section)`.
    pub fn add_row(&mut self, cells: Vec<Cell>, end_section: bool) -> &mut Table {
        self.add_row_styled(cells, None, end_section)
    }

    /// `add_row(*cells, style=style, end_section=end_section)`.
    pub fn add_row_styled(
        &mut self,
        mut cells: Vec<Cell>,
        style: Option<Style>,
        end_section: bool,
    ) -> &mut Table {
        while cells.len() < self.columns.len() {
            cells.push(Cell::Str(String::new()));
        }
        for (index, cell) in cells.into_iter().enumerate() {
            if index == self.columns.len() {
                let mut column = Column {
                    highlight: Some(self.highlight),
                    ..Column::default()
                };
                for _ in 0..self.rows.len() {
                    column.cells.push(Cell::Text(Text::new("")));
                }
                self.columns.push(column);
            }
            self.columns[index].cells.push(cell);
        }
        self.rows.push(Row { style, end_section });
        self
    }

    /// Rich `add_section()`.
    pub fn add_section(&mut self) {
        if let Some(row) = self.rows.last_mut() {
            row.end_section = true;
        }
    }

    fn expand(&self) -> bool {
        self.expand || self.width.is_some()
    }

    fn extra_width(&self) -> usize {
        let mut width = 0;
        if self.box_style.is_some() && self.show_edge {
            width += 2;
        }
        if self.box_style.is_some() {
            width += self.columns.len().saturating_sub(1);
        }
        width
    }

    fn get_row_style(&self, index: usize) -> Style {
        let mut style = Style::NULL;
        if !self.row_styles.is_empty() {
            style = style.combine(&self.row_styles[index % self.row_styles.len()]);
        }
        if let Some(row_style) = self.rows[index].style {
            style = style.combine(&row_style);
        }
        style
    }

    fn get_padding_width(&self, column_index: usize) -> usize {
        let (_, mut pad_right, _, mut pad_left) = self.padding;
        if self.collapse_padding {
            pad_left = 0;
            pad_right = pad_right.abs_diff(pad_left);
        }
        if !self.pad_edge {
            if column_index == 0 {
                pad_left = 0;
            }
            if column_index + 1 == self.columns.len() {
                pad_right = 0;
            }
        }
        pad_left + pad_right
    }

    fn get_cells<'a>(&'a self, column_index: usize) -> Vec<CellSpec<'a>> {
        let column = &self.columns[column_index];
        let (top, right, bottom, left) = self.padding;
        let any_padding = top + right + bottom + left > 0;
        let first_column = column_index == 0;
        let last_column = column_index + 1 == self.columns.len();
        let get_padding = |first_row: bool, last_row: bool| {
            let (mut top, mut right, mut bottom, mut left) = (top, right, bottom, left);
            if self.collapse_padding {
                if !first_column {
                    left = left.saturating_sub(right);
                }
                if !last_row {
                    bottom = top.saturating_sub(bottom);
                }
            }
            if !self.pad_edge {
                if first_column {
                    left = 0;
                }
                if last_column {
                    right = 0;
                }
                if first_row {
                    top = 0;
                }
                if last_row {
                    bottom = 0;
                }
            }
            (top, right, bottom, left)
        };
        let mut raw: Vec<(Style, &Cell)> = Vec::with_capacity(column.cells.len() + 2);
        if self.show_header {
            raw.push((
                self.header_style.combine(&column.header_style),
                &column.header,
            ));
        }
        for cell in &column.cells {
            raw.push((column.style, cell));
        }
        if self.show_footer {
            raw.push((
                self.footer_style.combine(&column.footer_style),
                &column.footer,
            ));
        }
        let count = raw.len();
        raw.into_iter()
            .enumerate()
            .map(|(i, (style, cell))| CellSpec {
                style,
                inner: cell,
                pad: any_padding.then(|| get_padding(i == 0, i + 1 == count)),
                vertical: column.vertical,
            })
            .collect()
    }

    fn measure_column(
        &self,
        console: &Console,
        options: &RenderOptions,
        index: usize,
    ) -> Measurement {
        let column = &self.columns[index];
        let max_width = options.max_width;
        if max_width < 1 {
            return Measurement::new(0, 0);
        }
        let padding_width = self.get_padding_width(index);
        if let Some(width) = column.width {
            return Measurement::new(width + padding_width, width + padding_width)
                .with_maximum(max_width);
        }
        let mut minimum: Option<usize> = None;
        let mut maximum: Option<usize> = None;
        for cell in self.get_cells(index) {
            let m = Measurement::get(console, options, &cell);
            minimum = Some(minimum.map_or(m.minimum, |v| v.max(m.minimum)));
            maximum = Some(maximum.map_or(m.maximum, |v| v.max(m.maximum)));
        }
        Measurement::new(minimum.unwrap_or(1), maximum.unwrap_or(max_width))
            .with_maximum(max_width)
            .clamp(
                column.min_width.map(|w| w + padding_width),
                column.max_width.map(|w| w + padding_width),
            )
    }

    fn collapse_widths(mut widths: Vec<isize>, wrapable: &[bool], max_width: isize) -> Vec<isize> {
        let mut total_width: isize = widths.iter().sum();
        let mut excess_width = total_width - max_width;
        if wrapable.iter().any(|&w| w) {
            while total_width != 0 && excess_width > 0 {
                let max_column = widths
                    .iter()
                    .zip(wrapable)
                    .filter(|(_, w)| **w)
                    .map(|(&width, _)| width)
                    .max()
                    .unwrap_or(0);
                let second_max_column = widths
                    .iter()
                    .zip(wrapable)
                    .map(|(&width, &w)| if w && width != max_column { width } else { 0 })
                    .max()
                    .unwrap_or(0);
                let column_difference = max_column - second_max_column;
                let ratios: Vec<isize> = widths
                    .iter()
                    .zip(wrapable)
                    .map(|(&width, &w)| isize::from(width == max_column && w))
                    .collect();
                if !ratios.iter().any(|&r| r != 0) || column_difference == 0 {
                    break;
                }
                let max_reduce = vec![excess_width.min(column_difference); widths.len()];
                widths = ratio_reduce(excess_width, &ratios, &max_reduce, &widths);
                total_width = widths.iter().sum();
                excess_width = total_width - max_width;
            }
        }
        widths
    }

    fn calculate_column_widths(&self, console: &Console, options: &RenderOptions) -> Vec<usize> {
        let max_width = options.max_width as isize;
        let width_ranges: Vec<Measurement> = (0..self.columns.len())
            .map(|i| self.measure_column(console, options, i))
            .collect();
        let mut widths: Vec<isize> = width_ranges
            .iter()
            .map(|r| {
                if r.maximum == 0 {
                    1
                } else {
                    r.maximum as isize
                }
            })
            .collect();
        let extra_width = self.extra_width() as isize;
        if self.expand() {
            let ratios: Vec<isize> = self
                .columns
                .iter()
                .filter_map(|c| c.ratio.map(|r| r as isize))
                .collect();
            if ratios.iter().any(|&r| r != 0) {
                let fixed_widths: Vec<isize> = width_ranges
                    .iter()
                    .zip(&self.columns)
                    .map(|(r, c)| {
                        if c.ratio.is_some() {
                            0
                        } else {
                            r.maximum as isize
                        }
                    })
                    .collect();
                let flex_minimum: Vec<isize> = self
                    .columns
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.ratio.is_some())
                    .map(|(i, c)| {
                        (c.width.filter(|&w| w != 0).unwrap_or(1) + self.get_padding_width(i))
                            as isize
                    })
                    .collect();
                let flexible_width = max_width - fixed_widths.iter().sum::<isize>();
                let flex_widths = ratio_distribute(flexible_width, &ratios, Some(&flex_minimum));
                let mut iter = flex_widths.into_iter();
                for (index, column) in self.columns.iter().enumerate() {
                    if column.ratio.is_some() {
                        widths[index] = fixed_widths[index] + iter.next().unwrap();
                    }
                }
            }
        }
        let mut table_width: isize = widths.iter().sum();
        if table_width > max_width {
            let wrapable: Vec<bool> = self
                .columns
                .iter()
                .map(|c| c.width.is_none() && !c.no_wrap)
                .collect();
            widths = Self::collapse_widths(widths, &wrapable, max_width);
            table_width = widths.iter().sum();
            if table_width > max_width {
                let excess_width = table_width - max_width;
                widths = ratio_reduce(excess_width, &vec![1; widths.len()], &widths, &widths);
                table_width = widths.iter().sum();
            }
            widths = widths
                .iter()
                .enumerate()
                .map(|(i, &w)| {
                    self.measure_column(console, &options.update_width(w), i)
                        .maximum as isize
                })
                .collect();
        }
        let below_min = self
            .min_width
            .is_some_and(|m| table_width < m as isize - extra_width);
        if (table_width < max_width && self.expand()) || below_min {
            let target = match self.min_width {
                None => max_width,
                Some(m) => (m as isize - extra_width).min(max_width),
            };
            let pad_widths = ratio_distribute(target - table_width, &widths, None);
            widths = widths.iter().zip(pad_widths).map(|(w, p)| w + p).collect();
        }
        widths.into_iter().map(|w| w.max(0) as usize).collect()
    }

    fn render_annotation(
        &self,
        console: &Console,
        options: &RenderOptions,
        annotation: &Cell,
        style: Style,
        justify: Justify,
    ) -> Vec<Segment> {
        let options = RenderOptions {
            justify: Some(justify),
            ..options.clone()
        };
        match annotation {
            Cell::Str(s) => {
                let text = console.render_str(
                    s,
                    RenderStr {
                        style,
                        highlight: Some(false),
                        ..RenderStr::default()
                    },
                );
                console.render(&text, &options)
            }
            Cell::Text(t) => console.render(t, &options),
            Cell::Renderable(r) => console.render(r.as_ref(), &options),
        }
    }

    fn render_body(
        &self,
        console: &Console,
        options: &RenderOptions,
        widths: &[usize],
    ) -> Vec<Segment> {
        let mut out = Vec::new();
        let border_style = Some(self.style.combine(&self.border_style));
        let column_cells: Vec<Vec<CellSpec>> =
            (0..self.columns.len()).map(|i| self.get_cells(i)).collect();
        let row_count = column_cells.iter().map(Vec::len).min().unwrap_or(0);
        let bx = self.box_style.map(|b| {
            if self.show_header {
                b
            } else {
                b.get_plain_headed_box()
            }
        });
        let seg = |text: String| Segment::new(text, border_style);
        let box_segments = bx.map(|b| {
            [
                (
                    seg(b.head_left.into()),
                    seg(b.head_right.into()),
                    seg(b.head_vertical.into()),
                ),
                (
                    seg(b.mid_left.into()),
                    seg(b.mid_right.into()),
                    seg(b.mid_vertical.into()),
                ),
                (
                    seg(b.foot_left.into()),
                    seg(b.foot_right.into()),
                    seg(b.foot_vertical.into()),
                ),
            ]
        });
        if let Some(b) = bx
            && self.show_edge
        {
            out.push(seg(b.get_top(widths)));
            out.push(Segment::line());
        }
        for index in 0..row_count {
            let first = index == 0;
            let last = index + 1 == row_count;
            let header_row = first && self.show_header;
            let footer_row = last && self.show_footer;
            let row = (!header_row && !footer_row)
                .then(|| &self.rows[index - usize::from(self.show_header)]);
            let row_style = if header_row || footer_row {
                Style::NULL
            } else {
                self.get_row_style(if self.show_header { index - 1 } else { index })
            };
            let mut max_height = 1;
            let mut cells: Vec<Vec<Vec<Segment>>> = Vec::with_capacity(self.columns.len());
            for ((&width, column), specs) in widths.iter().zip(&self.columns).zip(&column_cells) {
                let cell = &specs[index];
                let render_options = RenderOptions {
                    min_width: width,
                    max_width: width,
                    justify: Some(column.justify),
                    no_wrap: Some(column.no_wrap),
                    overflow: Some(column.overflow),
                    height: None,
                    highlight: Some(column.highlight.unwrap_or(self.highlight)),
                    ..options.clone()
                };
                let lines = console.render_lines(
                    cell,
                    &render_options,
                    Some(cell.style.combine(&row_style)),
                    true,
                    false,
                );
                max_height = max_height.max(lines.len());
                cells.push(lines);
            }
            let row_height = cells.iter().map(Vec::len).max().unwrap_or(0);
            let shaped: Vec<Vec<Vec<Segment>>> = cells
                .into_iter()
                .zip(widths)
                .zip(&column_cells)
                .map(|((lines, &width), specs)| {
                    let cell = &specs[index];
                    let vertical = if header_row {
                        VerticalAlign::Bottom
                    } else if footer_row {
                        VerticalAlign::Top
                    } else {
                        cell.vertical
                    };
                    let style = Some(cell.style.combine(&row_style));
                    segment::set_shape(
                        segment::align_lines(lines, width, row_height, style, vertical),
                        width,
                        Some(max_height),
                        None,
                    )
                })
                .collect();
            if let (Some(b), Some(box_segments)) = (bx, &box_segments) {
                if last && self.show_footer {
                    out.push(seg(b.get_row(widths, RowLevel::Foot, self.show_edge)));
                    out.push(Segment::line());
                }
                let (left, right, divider) = &box_segments[if first {
                    0
                } else if last {
                    2
                } else {
                    1
                }];
                let divider = if !divider.text.trim().is_empty() {
                    divider.clone()
                } else {
                    let own = divider.style.unwrap_or(Style::NULL);
                    Segment::new(
                        divider.text.clone(),
                        Some(row_style.background_style().combine(&own)),
                    )
                };
                for line_no in 0..max_height {
                    if self.show_edge {
                        out.push(left.clone());
                    }
                    let count = shaped.len();
                    for (i, cell) in shaped.iter().enumerate() {
                        out.extend(cell[line_no].iter().cloned());
                        if i + 1 < count {
                            out.push(divider.clone());
                        }
                    }
                    if self.show_edge {
                        out.push(right.clone());
                    }
                    out.push(Segment::line());
                }
            } else {
                for line_no in 0..max_height {
                    for cell in &shaped {
                        out.extend(cell[line_no].iter().cloned());
                    }
                    out.push(Segment::line());
                }
            }
            if let Some(b) = bx {
                if first && self.show_header {
                    out.push(seg(b.get_row(widths, RowLevel::Head, self.show_edge)));
                    out.push(Segment::line());
                }
                let end_section = row.is_some_and(|r| r.end_section);
                if (self.show_lines || self.leading > 0 || end_section)
                    && !last
                    && !(self.show_footer && index + 2 >= row_count)
                    && !(self.show_header && header_row)
                {
                    if self.leading > 0 {
                        out.push(seg(b
                            .get_row(widths, RowLevel::Mid, self.show_edge)
                            .repeat(self.leading)));
                    } else {
                        out.push(seg(b.get_row(widths, RowLevel::Row, self.show_edge)));
                    }
                    out.push(Segment::line());
                }
            }
        }
        if let Some(b) = bx
            && self.show_edge
        {
            out.push(seg(b.get_bottom(widths)));
            out.push(Segment::line());
        }
        out
    }
}

impl Renderable for Table {
    fn render(&self, console: &Console, options: &RenderOptions) -> Vec<Segment> {
        if self.columns.is_empty() {
            return vec![Segment::line()];
        }
        let max_width = self.width.unwrap_or(options.max_width);
        let extra_width = self.extra_width();
        let widths = self.calculate_column_widths(
            console,
            &options.update_width(max_width as isize - extra_width as isize),
        );
        let table_width = widths.iter().sum::<usize>() + extra_width;
        let render_options = RenderOptions {
            min_width: table_width,
            max_width: table_width,
            highlight: Some(self.highlight),
            height: None,
            ..options.clone()
        };
        let mut out = Vec::new();
        if let Some(title) = self.title.as_ref().filter(|t| t.is_truthy()) {
            let style = self
                .title_style
                .unwrap_or_else(|| theme::style("table.title"));
            out.extend(self.render_annotation(
                console,
                &render_options,
                title,
                style,
                self.title_justify,
            ));
        }
        out.extend(self.render_body(console, &render_options, &widths));
        if let Some(caption) = self.caption.as_ref().filter(|t| t.is_truthy()) {
            let style = self
                .caption_style
                .unwrap_or_else(|| theme::style("table.caption"));
            out.extend(self.render_annotation(
                console,
                &render_options,
                caption,
                style,
                self.caption_justify,
            ));
        }
        out
    }

    fn measure(&self, console: &Console, options: &RenderOptions) -> Measurement {
        let max_width = self.width.unwrap_or(options.max_width) as isize;
        let extra_width = self.extra_width();
        let max_width: usize = self
            .calculate_column_widths(
                console,
                &options.update_width(max_width - extra_width as isize),
            )
            .iter()
            .sum();
        let measurements: Vec<Measurement> = (0..self.columns.len())
            .map(|i| self.measure_column(console, &options.update_width(max_width as isize), i))
            .collect();
        let minimum = measurements.iter().map(|m| m.minimum).sum::<usize>() + extra_width;
        let maximum = match self.width {
            None => measurements.iter().map(|m| m.maximum).sum::<usize>() + extra_width,
            Some(w) => w,
        };
        Measurement::new(minimum, maximum).clamp(self.min_width, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratios() {
        assert_eq!(
            ratio_reduce(3, &[1, 1, 1], &[10, 10, 10], &[10, 10, 10]),
            vec![9, 9, 9]
        );
        assert_eq!(ratio_distribute(10, &[1, 1, 1], None), vec![4, 3, 3]);
        assert_eq!(ratio_reduce(5, &[1, 0], &[5, 5], &[20, 10]), vec![15, 10]);
    }
}
