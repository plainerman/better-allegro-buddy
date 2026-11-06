//! Rich `Text` (rich/text.py): a string with style spans, plus Rich's word wrapping
//! (`rich/_wrap.py`) and `Lines.justify` (`rich/containers.py`).
//!
//! * `Text(plain, style=base)`: `style` is the BASE style applied under all spans; it also
//!   styles the padding added when justifying. Constructing strips control codes (BEL, BS,
//!   VT, FF, CR).
//! * Span styles are resolved [`Style`]s (Rich resolves style names at render time with the
//!   console theme; bab only ever uses the default theme, see [`super::theme`]).
//! * Offsets are code points, like Rich; `len()` is code points, not cells.
//! * `Console.print(text)` re-joins the text (Rich `_collect_renderables`), which drops the
//!   text's own `justify`/`overflow`/`no_wrap`/`end`; see [`super::console`].

use super::cells::{cell_len, chop_cells, set_cell_size};
use super::style::{Style, py_isspace};
use super::theme::Theme;
use super::{Console, Measurement, RenderOptions, Renderable, Segment};

/// Rich `JustifyMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Justify {
    #[default]
    Default,
    Left,
    Center,
    Right,
    Full,
}

/// Rich `OverflowMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Overflow {
    #[default]
    Fold,
    Crop,
    Ellipsis,
    Ignore,
}

/// A styled range in code points `[start, end)` (Rich `Span`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub style: Style,
}

impl Span {
    pub fn new(start: usize, end: usize, style: Style) -> Span {
        Span { start, end, style }
    }
}

/// Rich `Text`.
#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    plain: String,
    length: usize,
    spans: Vec<Span>,
    /// Base style (Rich `Text.style`).
    pub style: Style,
    pub justify: Option<Justify>,
    pub overflow: Option<Overflow>,
    pub no_wrap: Option<bool>,
    pub end: String,
    pub tab_size: Option<usize>,
}

impl Default for Text {
    fn default() -> Text {
        Text {
            plain: String::new(),
            length: 0,
            spans: Vec::new(),
            style: Style::NULL,
            justify: None,
            overflow: None,
            no_wrap: None,
            end: "\n".to_owned(),
            tab_size: None,
        }
    }
}

/// One argument of [`Text::assemble`]: `str`, `(str, style)` or `Text`.
#[derive(Debug, Clone, PartialEq)]
pub enum TextPart {
    Plain(String),
    Styled(String, Style),
    Text(Text),
}

/// Rich `strip_control_codes`: removes BEL, BS, VT, FF and CR.
pub fn strip_control_codes(text: &str) -> std::borrow::Cow<'_, str> {
    const STRIP: [char; 5] = ['\x07', '\x08', '\x0b', '\x0c', '\x0d'];
    if text.contains(STRIP) {
        std::borrow::Cow::Owned(text.chars().filter(|c| !STRIP.contains(c)).collect())
    } else {
        std::borrow::Cow::Borrowed(text)
    }
}

fn char_count(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.chars().count()
    }
}

/// Byte offset of the `index`-th code point of `s` (clamped to the end).
fn byte_at(s: &str, index: usize) -> usize {
    if s.is_ascii() {
        return index.min(s.len());
    }
    s.char_indices().nth(index).map_or(s.len(), |(b, _)| b)
}

/// `s[start:end]` in code points.
fn char_slice(s: &str, start: usize, end: usize) -> &str {
    if s.is_ascii() {
        let end = end.min(s.len());
        return &s[start.min(end)..end];
    }
    let a = byte_at(s, start);
    let b = byte_at(s, end).max(a);
    &s[a..b]
}

/// Python `str.rstrip()`.
fn py_rstrip(s: &str) -> &str {
    s.trim_end_matches(py_isspace)
}

/// Python `str.splitlines()`.
pub(crate) fn py_splitlines(s: &str) -> Vec<&str> {
    let is_break = |c: char| {
        matches!(
            c,
            '\n' | '\r'
                | '\x0b'
                | '\x0c'
                | '\x1c'
                | '\x1d'
                | '\x1e'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        )
    };
    let mut lines = Vec::new();
    let mut rest = s;
    while !rest.is_empty() {
        match rest.find(is_break) {
            Some(i) => {
                lines.push(&rest[..i]);
                let skip = if rest[i..].starts_with("\r\n") {
                    2
                } else {
                    rest[i..].chars().next().unwrap().len_utf8()
                };
                rest = &rest[i + skip..];
            }
            None => {
                lines.push(rest);
                break;
            }
        }
    }
    lines
}

impl Text {
    /// `Text(plain)` (control codes stripped).
    pub fn new(plain: impl Into<String>) -> Text {
        Text::with_style(plain, Style::NULL)
    }

    /// `Text(plain, style=definition)`; `definition` is a Rich style string or theme name
    /// (invalid definitions render unstyled, as in Rich).
    pub fn styled(plain: impl Into<String>, definition: &str) -> Text {
        Text::with_style(plain, super::theme::style(definition))
    }

    /// `Text(plain, style=style)`.
    pub fn with_style(plain: impl Into<String>, style: Style) -> Text {
        let plain: String = plain.into();
        let plain = match strip_control_codes(&plain) {
            std::borrow::Cow::Borrowed(_) => plain,
            std::borrow::Cow::Owned(s) => s,
        };
        Text {
            length: char_count(&plain),
            plain,
            style,
            ..Text::default()
        }
    }

    /// `Text.assemble(*parts)`.
    pub fn assemble<I: IntoIterator<Item = TextPart>>(parts: I) -> Text {
        let mut text = Text::default();
        for part in parts {
            match part {
                TextPart::Plain(s) => {
                    text.append_styled(&s, Style::NULL);
                }
                TextPart::Styled(s, style) => {
                    text.append_styled(&s, style);
                }
                TextPart::Text(t) => {
                    text.append_text(&t);
                }
            }
        }
        text
    }

    /// Rich `blank_copy(plain)`: same attributes, no spans.
    pub fn blank_copy(&self, plain: &str) -> Text {
        Text {
            justify: self.justify,
            overflow: self.overflow,
            no_wrap: self.no_wrap,
            end: self.end.clone(),
            tab_size: self.tab_size,
            ..Text::with_style(plain, self.style)
        }
    }

    /// `self.join(items)`: items separated by copies of `self`.
    pub fn join<I: IntoIterator<Item = Text>>(&self, items: I) -> Text {
        let mut new_text = self.blank_copy("");
        let mut offset = 0;
        let mut push = |new_text: &mut Text, text: &Text| {
            new_text.plain.push_str(&text.plain);
            if !text.style.is_null() {
                new_text
                    .spans
                    .push(Span::new(offset, offset + text.length, text.style));
            }
            new_text.spans.extend(
                text.spans
                    .iter()
                    .map(|s| Span::new(offset + s.start, offset + s.end, s.style)),
            );
            offset += text.length;
        };
        let mut first = true;
        for item in items {
            if !first && !self.plain.is_empty() {
                push(&mut new_text, self);
            }
            first = false;
            push(&mut new_text, &item);
        }
        new_text.length = offset;
        new_text
    }

    /// The plain string.
    pub fn plain(&self) -> &str {
        &self.plain
    }

    /// Rich `plain` setter: strips control codes and trims spans when shorter.
    pub fn set_plain(&mut self, new_text: &str) {
        if new_text != self.plain {
            let sanitized = strip_control_codes(new_text).into_owned();
            let old_length = self.length;
            self.length = char_count(&sanitized);
            self.plain = sanitized;
            if old_length > self.length {
                self.trim_spans();
            }
        }
    }

    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    pub fn spans_mut(&mut self) -> &mut Vec<Span> {
        &mut self.spans
    }

    /// `len(text)`: code points.
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Terminal cells of the plain text (Rich `Text.cell_len`).
    pub fn cell_len(&self) -> usize {
        cell_len(&self.plain)
    }

    /// `append(text, style)` with a style definition (`None`/empty: no span).
    pub fn append(&mut self, text: &str, style: Option<&str>) -> &mut Text {
        let style = style.map_or(Style::NULL, super::theme::style);
        self.append_styled(text, style)
    }

    /// `append(text, style)` with a parsed style.
    pub fn append_styled(&mut self, text: &str, style: Style) -> &mut Text {
        if !text.is_empty() {
            let sanitized = strip_control_codes(text);
            let offset = self.length;
            let length = char_count(&sanitized);
            self.plain.push_str(&sanitized);
            if !style.is_null() {
                self.spans.push(Span::new(offset, offset + length, style));
            }
            self.length += length;
        }
        self
    }

    /// `append_text(other)` / `append(other: Text)`.
    pub fn append_text(&mut self, other: &Text) -> &mut Text {
        let offset = self.length;
        if !other.style.is_null() {
            self.spans
                .push(Span::new(offset, offset + other.length, other.style));
        }
        self.plain.push_str(&other.plain);
        self.spans.extend(
            other
                .spans
                .iter()
                .map(|s| Span::new(s.start + offset, s.end + offset, s.style)),
        );
        self.length += other.length;
        self
    }

    /// `copy_styles(other)`: append the other text's spans.
    pub fn copy_styles(&mut self, other: &Text) {
        self.spans.extend_from_slice(&other.spans);
    }

    fn clamp_range(&self, start: isize, end: Option<isize>) -> Option<(usize, usize)> {
        let length = self.length as isize;
        let start = if start < 0 { length + start } else { start };
        let end = match end {
            None => length,
            Some(e) if e < 0 => length + e,
            Some(e) => e,
        };
        if start >= length || end <= start {
            return None;
        }
        Some((start.max(0) as usize, end.min(length) as usize))
    }

    /// `stylize(style, start, end)`: add a span (code points; negative = from the end).
    pub fn stylize(&mut self, style: Style, start: isize, end: Option<isize>) -> &mut Text {
        if !style.is_null()
            && let Some((start, end)) = self.clamp_range(start, end)
        {
            self.spans.push(Span::new(start, end, style));
        }
        self
    }

    /// `stylize_before(style, start, end)`: insert a span below the existing ones.
    pub fn stylize_before(&mut self, style: Style, start: isize, end: Option<isize>) -> &mut Text {
        if !style.is_null()
            && let Some((start, end)) = self.clamp_range(start, end)
        {
            self.spans.insert(0, Span::new(start, end, style));
        }
        self
    }

    /// Rich `highlight_regex(re, style_prefix=prefix)`: a span per named group (in pattern
    /// order), styled `prefix + group name` resolved with `theme`.
    pub fn highlight_regex(
        &mut self,
        re: &fancy_regex::Regex,
        style_prefix: &str,
        theme: &Theme,
    ) -> usize {
        let names: Vec<(usize, Style)> = re
            .capture_names()
            .enumerate()
            .filter_map(|(i, name)| {
                name.map(|n| (i, theme.get_style_or_null(&format!("{style_prefix}{n}"))))
            })
            .collect();
        let mut new_spans = Vec::new();
        let mut count = 0;
        let mut position = 0;
        let plain = self.plain.as_str();
        while position <= plain.len() {
            let Ok(Some(caps)) = re.captures_from_pos(plain, position) else {
                break;
            };
            let whole = caps.get(0).unwrap();
            count += 1;
            for &(index, style) in &names {
                if let Some(m) = caps.get(index)
                    && m.end() > m.start()
                {
                    let start = char_count(&plain[..m.start()]);
                    let end = start + char_count(m.as_str());
                    new_spans.push(Span::new(start, end, style));
                }
            }
            position = if whole.end() > whole.start() {
                whole.end()
            } else {
                match plain[whole.end()..].chars().next() {
                    Some(c) => whole.end() + c.len_utf8(),
                    None => break,
                }
            };
        }
        self.spans.extend(new_spans);
        count
    }

    /// Rich `highlight_regex(re, style=style)`: a span over every whole match.
    pub fn highlight_regex_style(&mut self, re: &fancy_regex::Regex, style: Style) -> usize {
        let mut new_spans = Vec::new();
        let mut count = 0;
        for m in re.find_iter(&self.plain).flatten() {
            count += 1;
            if m.end() > m.start() && !style.is_null() {
                let start = char_count(&self.plain[..m.start()]);
                new_spans.push(Span::new(start, start + char_count(m.as_str()), style));
            }
        }
        self.spans.extend(new_spans);
        count
    }

    /// Rich `highlight_words(words, style, case_sensitive=True)`.
    pub fn highlight_words(&mut self, words: &[&str], style: Style) -> usize {
        let pattern = words
            .iter()
            .map(|w| regex::escape(w))
            .collect::<Vec<_>>()
            .join("|");
        let Ok(re) = regex::Regex::new(&pattern) else {
            return 0;
        };
        let mut count = 0;
        let mut new_spans = Vec::new();
        for m in re.find_iter(&self.plain) {
            let start = char_count(&self.plain[..m.start()]);
            new_spans.push(Span::new(start, start + char_count(m.as_str()), style));
            count += 1;
        }
        self.spans.extend(new_spans);
        count
    }

    /// Rich `get_style_at_offset(console, offset)`.
    pub fn get_style_at_offset(&self, offset: isize) -> Style {
        let offset = if offset < 0 {
            self.length as isize + offset
        } else {
            offset
        };
        let mut style = self.style;
        for span in &self.spans {
            if (span.end as isize) > offset && offset >= span.start as isize {
                style = style.combine(&span.style);
            }
        }
        style
    }

    /// Rich `extend_style(spaces)`: extend the text (and spans touching its end) by spaces.
    pub fn extend_style(&mut self, spaces: usize) {
        if spaces == 0 {
            return;
        }
        if !self.spans.is_empty() {
            let end_offset = self.length;
            for span in &mut self.spans {
                if span.end >= end_offset {
                    span.end += spaces;
                }
            }
            self.plain.extend(std::iter::repeat_n(' ', spaces));
            self.length += spaces;
        } else {
            let new = format!("{}{}", self.plain, " ".repeat(spaces));
            self.set_plain(&new);
        }
    }

    /// `rstrip()`.
    pub fn rstrip(&mut self) {
        let stripped = py_rstrip(&self.plain).to_owned();
        self.set_plain(&stripped);
    }

    /// Rich `rstrip_end(size)`: remove trailing whitespace beyond `size` code points.
    pub fn rstrip_end(&mut self, size: usize) {
        if self.length > size {
            let excess = self.length - size;
            let trimmed = py_rstrip(&self.plain);
            let whitespace = char_count(&self.plain[trimmed.len()..]);
            if whitespace > 0 {
                self.right_crop(whitespace.min(excess));
            }
        }
    }

    /// Rich `set_length(new_length)`: pad or crop to a number of code points.
    pub fn set_length(&mut self, new_length: usize) {
        if self.length < new_length {
            self.pad_right(new_length - self.length, ' ');
        } else if self.length > new_length {
            self.right_crop(self.length - new_length);
        }
    }

    fn trim_spans(&mut self) {
        let max_offset = self.length;
        self.spans.retain(|s| s.start < max_offset);
        for span in &mut self.spans {
            if span.end >= max_offset {
                span.end = span.end.min(max_offset);
            }
        }
    }

    /// Rich `right_crop(amount)`.
    pub fn right_crop(&mut self, amount: usize) {
        let max_offset = self.length as isize - amount as isize;
        self.spans.retain(|s| (s.start as isize) < max_offset);
        for span in &mut self.spans {
            if span.end as isize >= max_offset {
                span.end = (span.end as isize).min(max_offset) as usize;
            }
        }
        if amount > 0 {
            let cut = byte_at(&self.plain, self.length.saturating_sub(amount));
            self.plain.truncate(cut);
            self.length = self.length.saturating_sub(amount);
        }
    }

    /// Rich `truncate(max_width, overflow=, pad=)`.
    pub fn truncate(&mut self, max_width: usize, overflow: Option<Overflow>, pad: bool) {
        let overflow = overflow.or(self.overflow).unwrap_or_default();
        if overflow != Overflow::Ignore {
            let length = cell_len(&self.plain);
            if length > max_width {
                let new = if overflow == Overflow::Ellipsis {
                    let mut s = if max_width == 0 {
                        // Python `set_cell_size(text, -1)`: ASCII text loses its last char.
                        if self.plain.bytes().all(|b| (0x20..0x7f).contains(&b)) {
                            self.plain[..self.plain.len().saturating_sub(1)].to_owned()
                        } else {
                            String::new()
                        }
                    } else {
                        set_cell_size(&self.plain, max_width - 1)
                    };
                    s.push('…');
                    s
                } else {
                    set_cell_size(&self.plain, max_width)
                };
                self.set_plain(&new);
            }
            if pad && length < max_width {
                let spaces = max_width - length;
                self.plain.extend(std::iter::repeat_n(' ', spaces));
                self.length += spaces;
            }
        }
    }

    /// Rich `pad(count, character)`: both sides.
    pub fn pad(&mut self, count: usize, character: char) {
        if count > 0 {
            let pad: String = std::iter::repeat_n(character, count).collect();
            let new = format!("{pad}{}{pad}", self.plain);
            self.set_plain(&new);
            for span in &mut self.spans {
                span.start += count;
                span.end += count;
            }
        }
    }

    /// Rich `pad_left(count, character)`.
    pub fn pad_left(&mut self, count: usize, character: char) {
        if count > 0 {
            let pad: String = std::iter::repeat_n(character, count).collect();
            let new = format!("{pad}{}", self.plain);
            self.set_plain(&new);
            for span in &mut self.spans {
                span.start += count;
                span.end += count;
            }
        }
    }

    /// Rich `pad_right(count, character)`.
    pub fn pad_right(&mut self, count: usize, character: char) {
        if count > 0 {
            let pad: String = std::iter::repeat_n(character, count).collect();
            let new = format!("{}{pad}", self.plain);
            self.set_plain(&new);
        }
    }

    /// Rich `align(align, width, character)`.
    pub fn align(&mut self, align: Justify, width: usize, character: char) {
        self.truncate(width, None, false);
        let excess = width.saturating_sub(cell_len(&self.plain));
        if excess > 0 {
            match align {
                Justify::Left => self.pad_right(excess, character),
                Justify::Center => {
                    let left = excess / 2;
                    self.pad_left(left, character);
                    self.pad_right(excess - left, character);
                }
                _ => self.pad_left(excess, character),
            }
        }
    }

    /// Rich `divide(offsets)`: split at code point offsets, keeping spans.
    pub fn divide(&self, offsets: &[usize]) -> Vec<Text> {
        if offsets.is_empty() {
            return vec![self.clone()];
        }
        let text_length = self.length;
        let mut divide_offsets = Vec::with_capacity(offsets.len() + 2);
        divide_offsets.push(0);
        divide_offsets.extend_from_slice(offsets);
        divide_offsets.push(text_length);
        let line_ranges: Vec<(usize, usize)> =
            divide_offsets.windows(2).map(|w| (w[0], w[1])).collect();
        let mut new_lines: Vec<Text> = line_ranges
            .iter()
            .map(|&(start, end)| Text {
                justify: self.justify,
                overflow: self.overflow,
                ..Text::with_style(char_slice(&self.plain, start, end), self.style)
            })
            .collect();
        if self.spans.is_empty() {
            return new_lines;
        }
        let line_count = line_ranges.len() as isize;
        for span in &self.spans {
            let (span_start, span_end) = (span.start as isize, span.end as isize);
            let mut lower: isize = 0;
            let mut upper: isize = line_count;
            let mut start_line_no = (lower + upper).div_euclid(2);
            let (mut line_start, mut line_end);
            loop {
                let range = line_ranges[start_line_no as usize];
                line_start = range.0 as isize;
                line_end = range.1 as isize;
                if span_start < line_start {
                    upper = start_line_no - 1;
                } else if span_start > line_end {
                    lower = start_line_no + 1;
                } else {
                    break;
                }
                start_line_no = (lower + upper).div_euclid(2);
            }
            let end_line_no = if span_end < line_end {
                start_line_no
            } else {
                let mut end_line_no = start_line_no;
                lower = start_line_no;
                upper = line_count;
                loop {
                    let range = line_ranges[end_line_no as usize];
                    let (ls, le) = (range.0 as isize, range.1 as isize);
                    if span_end < ls {
                        upper = end_line_no - 1;
                    } else if span_end > le {
                        lower = end_line_no + 1;
                    } else {
                        break;
                    }
                    end_line_no = (lower + upper).div_euclid(2);
                }
                end_line_no
            };
            for line_no in start_line_no..=end_line_no {
                let (ls, le) = line_ranges[line_no as usize];
                let (ls, le) = (ls as isize, le as isize);
                let new_start = (span_start - ls).max(0);
                let new_end = (span_end - ls).min(le - ls);
                if new_end > new_start {
                    new_lines[line_no as usize].spans.push(Span::new(
                        new_start as usize,
                        new_end as usize,
                        span.style,
                    ));
                }
            }
        }
        new_lines
    }

    /// Rich `split(separator, include_separator=, allow_blank=)`.
    pub fn split(&self, separator: &str, include_separator: bool, allow_blank: bool) -> Vec<Text> {
        assert!(!separator.is_empty());
        if !self.plain.contains(separator) {
            return vec![self.clone()];
        }
        let sep_len = char_count(separator);
        let mut matches = Vec::new();
        let mut char_pos = 0;
        let mut byte_pos = 0;
        for (index, _) in self.plain.match_indices(separator) {
            char_pos += char_count(&self.plain[byte_pos..index]);
            byte_pos = index;
            matches.push(char_pos);
        }
        let mut lines = if include_separator {
            let offsets: Vec<usize> = matches.iter().map(|&s| s + sep_len).collect();
            self.divide(&offsets)
        } else {
            let offsets: Vec<usize> = matches.iter().flat_map(|&s| [s, s + sep_len]).collect();
            self.divide(&offsets)
                .into_iter()
                .filter(|line| line.plain != separator)
                .collect()
        };
        if !allow_blank && self.plain.ends_with(separator) {
            lines.pop();
        }
        lines
    }

    /// Rich `expand_tabs(tab_size)`.
    pub fn expand_tabs(&mut self, tab_size: Option<usize>) {
        if !self.plain.contains('\t') {
            return;
        }
        let tab_size = tab_size.or(self.tab_size).unwrap_or(8);
        let mut new_text = Vec::new();
        for line in self.split("\n", true, false) {
            if !line.plain.contains('\t') {
                new_text.push(line);
                continue;
            }
            let mut cell_position = 0;
            for mut part in line.split("\t", true, false) {
                if part.plain.ends_with('\t') {
                    part.plain.pop();
                    part.plain.push(' ');
                    cell_position += part.cell_len();
                    let remainder = if tab_size == 0 {
                        0
                    } else {
                        cell_position % tab_size
                    };
                    if remainder > 0 {
                        let spaces = tab_size - remainder;
                        part.extend_style(spaces);
                        cell_position += spaces;
                    }
                } else {
                    cell_position += part.cell_len();
                }
                new_text.push(part);
            }
        }
        let result = Text::new("").join(new_text);
        self.plain = result.plain;
        self.length = result.length;
        self.spans = result.spans;
    }

    /// Rich `Text.wrap(console, width, justify=, overflow=, tab_size=, no_wrap=)`.
    pub fn wrap(
        &self,
        width: usize,
        justify: Option<Justify>,
        overflow: Option<Overflow>,
        tab_size: usize,
        no_wrap: Option<bool>,
    ) -> Vec<Text> {
        let wrap_justify = justify.or(self.justify).unwrap_or_default();
        let wrap_overflow = overflow.or(self.overflow).unwrap_or_default();
        let no_wrap =
            no_wrap.or(self.no_wrap).unwrap_or(false) || overflow == Some(Overflow::Ignore);
        let mut lines = Vec::new();
        for mut line in self.split("\n", false, true) {
            if line.plain.contains('\t') {
                line.expand_tabs(Some(tab_size));
            }
            let mut new_lines = if no_wrap {
                if overflow == Some(Overflow::Ignore) {
                    lines.push(line);
                    continue;
                }
                vec![line]
            } else {
                let offsets = divide_line(&line.plain, width, wrap_overflow == Overflow::Fold);
                let mut new_lines = line.divide(&offsets);
                for l in &mut new_lines {
                    l.rstrip_end(width);
                }
                new_lines
            };
            justify_lines(&mut new_lines, width, wrap_justify, wrap_overflow);
            for l in &mut new_lines {
                l.truncate(width, Some(wrap_overflow), false);
            }
            lines.extend(new_lines);
        }
        lines
    }

    /// Rich `Text.render(console, end)`: segments per style run.
    pub fn render_with_end(&self, end: &str) -> Vec<Segment> {
        let mut segments = Vec::new();
        if self.spans.is_empty() {
            segments.push(Segment::plain(self.plain.as_str()));
            if !end.is_empty() {
                segments.push(Segment::plain(end));
            }
            return segments;
        }
        let mut style_map = Vec::with_capacity(self.spans.len() + 1);
        style_map.push(self.style);
        style_map.extend(self.spans.iter().map(|s| s.style));
        let mut events: Vec<(usize, bool, usize)> = Vec::with_capacity(self.spans.len() * 2 + 2);
        events.push((0, false, 0));
        events.extend(
            self.spans
                .iter()
                .enumerate()
                .map(|(i, s)| (s.start, false, i + 1)),
        );
        events.extend(
            self.spans
                .iter()
                .enumerate()
                .map(|(i, s)| (s.end, true, i + 1)),
        );
        events.push((self.length, true, 0));
        events.sort_by_key(|&(offset, leaving, _)| (offset, leaving));
        let ascii = self.plain.is_ascii();
        let byte_offsets: Vec<usize> = if ascii {
            Vec::new()
        } else {
            let mut v: Vec<usize> = self.plain.char_indices().map(|(b, _)| b).collect();
            v.push(self.plain.len());
            v
        };
        let byte = |i: usize| {
            if ascii {
                i.min(self.plain.len())
            } else {
                byte_offsets[i.min(self.length)]
            }
        };
        let mut stack: Vec<usize> = Vec::new();
        for window in events.windows(2) {
            let (offset, leaving, style_id) = window[0];
            let next_offset = window[1].0;
            if leaving {
                if let Some(pos) = stack.iter().position(|&id| id == style_id) {
                    stack.remove(pos);
                }
            } else {
                stack.push(style_id);
            }
            if next_offset > offset {
                let mut ids = stack.clone();
                ids.sort_unstable();
                let style = ids
                    .iter()
                    .fold(Style::NULL, |acc, &id| acc.combine(&style_map[id]));
                segments.push(Segment::styled(
                    &self.plain[byte(offset)..byte(next_offset)],
                    style,
                ));
            }
        }
        if !end.is_empty() {
            segments.push(Segment::plain(end));
        }
        segments
    }

    /// Rich `Text.render(console, end="")` for one line.
    pub fn render_segments(&self) -> Vec<Segment> {
        self.render_with_end("")
    }

    /// Rich `Text.__rich_measure__`.
    pub fn measure_text(&self) -> Measurement {
        let max_text_width = py_splitlines(&self.plain)
            .into_iter()
            .map(cell_len)
            .max()
            .unwrap_or(0);
        let min_text_width = self
            .plain
            .split(py_isspace)
            .filter(|w| !w.is_empty())
            .map(cell_len)
            .max()
            .unwrap_or(max_text_width);
        Measurement {
            minimum: min_text_width,
            maximum: max_text_width,
        }
    }
}

/// Rich `Lines.justify(console, width, justify, overflow)`.
pub fn justify_lines(lines: &mut [Text], width: usize, justify: Justify, overflow: Overflow) {
    match justify {
        Justify::Default => {}
        Justify::Left => {
            for line in lines.iter_mut() {
                line.truncate(width, Some(overflow), true);
            }
        }
        Justify::Center => {
            for line in lines.iter_mut() {
                line.rstrip();
                line.truncate(width, Some(overflow), false);
                line.pad_left(width.saturating_sub(line.cell_len()) / 2, ' ');
                line.pad_right(width.saturating_sub(line.cell_len()), ' ');
            }
        }
        Justify::Right => {
            for line in lines.iter_mut() {
                line.rstrip();
                line.truncate(width, Some(overflow), false);
                line.pad_left(width.saturating_sub(line.cell_len()), ' ');
            }
        }
        Justify::Full => {
            let count = lines.len();
            for line in lines.iter_mut().take(count.saturating_sub(1)) {
                let words = line.split(" ", false, false);
                let words_size: usize = words.iter().map(Text::cell_len).sum();
                let mut num_spaces = words.len().saturating_sub(1);
                let mut spaces = vec![1usize; num_spaces];
                let mut index = 0;
                if !spaces.is_empty() {
                    while words_size + num_spaces < width {
                        let len = spaces.len();
                        spaces[len - index - 1] += 1;
                        num_spaces += 1;
                        index = (index + 1) % len;
                    }
                }
                let mut tokens = Vec::with_capacity(words.len() * 2);
                for (index, word) in words.iter().enumerate() {
                    tokens.push(word.clone());
                    if index < spaces.len() {
                        let style = word.get_style_at_offset(-1);
                        let next_style = words[index + 1].get_style_at_offset(0);
                        let space_style = if style == next_style {
                            style
                        } else {
                            line.style
                        };
                        tokens.push(Text::with_style(" ".repeat(spaces[index]), space_style));
                    }
                }
                *line = Text::new("").join(tokens);
            }
        }
    }
}

/// Rich `_wrap.words`: `\s*\S+\s*` runs as (start, end) code point offsets + text.
fn words(text: &str) -> Vec<(usize, usize, &str)> {
    let mut out = Vec::new();
    let mut chars = text.char_indices().peekable();
    let mut char_index = 0;
    while let Some(&(start_byte, _)) = chars.peek() {
        let start = char_index;
        let mut seen_word = false;
        // leading whitespace
        while let Some(&(_, c)) = chars.peek() {
            if py_isspace(c) {
                chars.next();
                char_index += 1;
            } else {
                break;
            }
        }
        while let Some(&(_, c)) = chars.peek() {
            if !py_isspace(c) {
                seen_word = true;
                chars.next();
                char_index += 1;
            } else {
                break;
            }
        }
        if !seen_word {
            break;
        }
        while let Some(&(_, c)) = chars.peek() {
            if py_isspace(c) {
                chars.next();
                char_index += 1;
            } else {
                break;
            }
        }
        let end_byte = chars.peek().map_or(text.len(), |&(b, _)| b);
        out.push((start, char_index, &text[start_byte..end_byte]));
    }
    out
}

/// Rich `_wrap.divide_line(text, width, fold)`: code point offsets to break the line at.
pub fn divide_line(text: &str, width: usize, fold: bool) -> Vec<usize> {
    let mut break_positions = Vec::new();
    let mut cell_offset: usize = 0;
    for (start, _end, word) in words(text) {
        let mut start = start;
        let word_length = cell_len(py_rstrip(word));
        let remaining_space = width as isize - cell_offset as isize;
        if remaining_space >= word_length as isize {
            cell_offset += cell_len(word);
        } else if word_length > width {
            if fold {
                let folded = chop_cells(word, width);
                let count = folded.len();
                for (i, line) in folded.iter().enumerate() {
                    if start > 0 {
                        break_positions.push(start);
                    }
                    if i == count - 1 {
                        cell_offset = cell_len(line);
                    } else {
                        start += char_count(line);
                    }
                }
            } else {
                if start > 0 {
                    break_positions.push(start);
                }
                cell_offset = cell_len(word);
            }
        } else if cell_offset > 0 && start > 0 {
            break_positions.push(start);
            cell_offset = cell_len(word);
        }
    }
    break_positions
}

impl Renderable for Text {
    fn render(&self, console: &Console, options: &RenderOptions) -> Vec<Segment> {
        let tab_size = self.tab_size.unwrap_or(console.tab_size());
        let justify = self.justify.or(options.justify).unwrap_or_default();
        let overflow = self.overflow.or(options.overflow).unwrap_or_default();
        let lines = self.wrap(
            options.max_width,
            Some(justify),
            Some(overflow),
            if tab_size == 0 { 8 } else { tab_size },
            Some(self.no_wrap.or(options.no_wrap).unwrap_or(false)),
        );
        Text::new("\n").join(lines).render_with_end(&self.end)
    }

    fn measure(&self, _console: &Console, _options: &RenderOptions) -> Measurement {
        self.measure_text()
    }

    fn as_text(&self) -> Option<&Text> {
        Some(self)
    }
}

impl From<&str> for Text {
    fn from(s: &str) -> Text {
        Text::new(s)
    }
}

impl From<String> for Text {
    fn from(s: String) -> Text {
        Text::new(s)
    }
}

impl From<&String> for Text {
    fn from(s: &String) -> Text {
        Text::new(s.as_str())
    }
}

impl From<&str> for TextPart {
    fn from(s: &str) -> TextPart {
        TextPart::Plain(s.to_owned())
    }
}

impl From<String> for TextPart {
    fn from(s: String) -> TextPart {
        TextPart::Plain(s)
    }
}

impl From<Text> for TextPart {
    fn from(t: Text) -> TextPart {
        TextPart::Text(t)
    }
}

/// `("text", "style definition")`.
impl From<(&str, &str)> for TextPart {
    fn from((s, style): (&str, &str)) -> TextPart {
        TextPart::Styled(s.to_owned(), super::theme::style(style))
    }
}

/// `(f"...", "style definition")`.
impl From<(String, &str)> for TextPart {
    fn from((s, style): (String, &str)) -> TextPart {
        TextPart::Styled(s, super::theme::style(style))
    }
}

/// `(text, Style)`.
impl From<(&str, Style)> for TextPart {
    fn from((s, style): (&str, Style)) -> TextPart {
        TextPart::Styled(s.to_owned(), style)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divide_line_matches_rich() {
        assert_eq!(divide_line("foo bar baz", 7, true), vec![8]);
        assert_eq!(divide_line("abcdefghij", 4, true), vec![4, 8]);
        assert_eq!(divide_line("  x", 1, true), vec![1, 2]);
        assert_eq!(divide_line("ab abcdefghij", 4, false), vec![3]);
    }

    #[test]
    fn join_and_spans() {
        let parts = vec![Text::styled("a", "bold"), Text::new("b")];
        let joined = Text::new(", ").join(parts);
        assert_eq!(joined.plain(), "a, b");
        assert_eq!(joined.spans(), &[Span::new(0, 1, Style::of("bold"))]);
    }

    #[test]
    fn render_runs() {
        let mut text = Text::new("hello world");
        text.stylize(Style::of("bold"), 0, Some(5));
        let segs = text.render_segments();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "hello");
        assert_eq!(segs[1].text, " world");
    }

    #[test]
    fn splitlines() {
        assert_eq!(py_splitlines("a\nb\r\nc\n"), vec!["a", "b", "c"]);
        assert_eq!(py_splitlines(""), Vec::<&str>::new());
    }
}
