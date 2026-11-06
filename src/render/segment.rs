//! Rich `Segment` (rich/segment.py): a run of text with one style, plus the line helpers
//! renderables need. Segment boundaries matter for the output bytes (each styled segment
//! is rendered as its own `ESC[..m..ESC[0m`), so these helpers split exactly like Rich.

use super::cells::{cell_len, char_width, set_cell_size};
use super::style::Style;

/// Text with an optional style. `None` and a null style both render as plain text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub text: String,
    pub style: Option<Style>,
}

/// Rich truthiness of a segment style (`if style:`).
pub fn style_truthy(style: Option<Style>) -> bool {
    style.is_some_and(|s| !s.is_null())
}

impl Segment {
    pub fn new(text: impl Into<String>, style: Option<Style>) -> Self {
        Segment {
            text: text.into(),
            style,
        }
    }

    pub fn plain(text: impl Into<String>) -> Self {
        Segment {
            text: text.into(),
            style: None,
        }
    }

    pub fn styled(text: impl Into<String>, style: Style) -> Self {
        Segment {
            text: text.into(),
            style: Some(style),
        }
    }

    /// `Segment.line()`.
    pub fn line() -> Self {
        Segment::plain("\n")
    }

    /// Terminal cells this segment occupies (`cell_length`).
    pub fn cell_len(&self) -> usize {
        cell_len(&self.text)
    }

    /// Rich `Segment.split_cells(cut)`.
    pub fn split_cells(&self, cut: usize) -> (Segment, Segment) {
        let text = &self.text;
        let style = self.style;
        if text.bytes().all(|b| (0x20..0x7f).contains(&b)) {
            if cut >= text.len() {
                return (self.clone(), Segment::new("", style));
            }
            return (
                Segment::new(&text[..cut], style),
                Segment::new(&text[cut..], style),
            );
        }
        let cell_length = self.cell_len();
        if cut >= cell_length {
            return (self.clone(), Segment::new("", style));
        }
        let chars: Vec<char> = text.chars().collect();
        let collect = |a: usize, b: usize| chars[a..b].iter().collect::<String>();
        let mut pos = ((cut as f64 / cell_length as f64) * chars.len() as f64) as usize;
        loop {
            let before = collect(0, pos);
            let cell_pos = cell_len(&before);
            let out_by = cell_pos as isize - cut as isize;
            if out_by == 0 {
                return (
                    Segment::new(before, style),
                    Segment::new(collect(pos, chars.len()), style),
                );
            }
            if out_by == -1 && char_width(chars[pos]) == 2 {
                return (
                    Segment::new(collect(0, pos) + " ", style),
                    Segment::new(" ".to_owned() + &collect(pos + 1, chars.len()), style),
                );
            }
            if out_by == 1 && char_width(chars[pos - 1]) == 2 {
                return (
                    Segment::new(collect(0, pos - 1) + " ", style),
                    Segment::new(" ".to_owned() + &collect(pos, chars.len()), style),
                );
            }
            if cell_pos < cut {
                pos += 1;
            } else {
                pos -= 1;
            }
        }
    }
}

/// Rich `Segment.apply_style(segments, style, post_style)`.
pub fn apply_style(
    segments: Vec<Segment>,
    style: Option<Style>,
    post_style: Option<Style>,
) -> Vec<Segment> {
    let style = style.filter(|s| !s.is_null());
    let post_style = post_style.filter(|s| !s.is_null());
    if style.is_none() && post_style.is_none() {
        return segments;
    }
    segments
        .into_iter()
        .map(|mut segment| {
            if let Some(style) = style {
                segment.style = Some(match segment.style {
                    Some(own) => style.combine(&own),
                    None => style,
                });
            }
            if let Some(post) = post_style {
                segment.style = Some(match segment.style {
                    Some(own) if !own.is_null() => own.combine(&post),
                    _ => post,
                });
            }
            segment
        })
        .collect()
}

/// Rich `Segment.split_lines(segments)`.
pub fn split_lines(segments: Vec<Segment>) -> Vec<Vec<Segment>> {
    split_lines_terminator(segments)
        .into_iter()
        .map(|(line, _)| line)
        .collect()
}

/// Rich `Segment.split_lines_terminator(segments)`: lines plus "ended with `\n`".
pub fn split_lines_terminator(segments: Vec<Segment>) -> Vec<(Vec<Segment>, bool)> {
    let mut lines = Vec::new();
    let mut line: Vec<Segment> = Vec::new();
    for segment in segments {
        if segment.text.contains('\n') {
            let mut parts = segment.text.split('\n').peekable();
            while let Some(part) = parts.next() {
                if !part.is_empty() {
                    line.push(Segment::new(part, segment.style));
                }
                if parts.peek().is_some() {
                    lines.push((std::mem::take(&mut line), true));
                }
            }
        } else {
            line.push(segment);
        }
    }
    if !line.is_empty() {
        lines.push((line, false));
    }
    lines
}

/// Rich `Segment.split_and_crop_lines(segments, length, style, pad, include_new_lines)`.
pub fn split_and_crop_lines(
    segments: Vec<Segment>,
    length: usize,
    style: Option<Style>,
    pad: bool,
    include_new_lines: bool,
) -> Vec<Vec<Segment>> {
    let mut lines = Vec::new();
    for (line, terminated) in split_lines_terminator(segments) {
        let mut cropped = adjust_line_length(line, length, style, pad);
        if terminated && include_new_lines {
            cropped.push(Segment::line());
        }
        lines.push(cropped);
    }
    lines
}

/// Rich `Segment.adjust_line_length(line, length, style, pad=True)`: crop or pad (with
/// spaces in `style`) a line to exactly `length` cells.
pub fn adjust_line_length(
    mut line: Vec<Segment>,
    length: usize,
    style: Option<Style>,
    pad: bool,
) -> Vec<Segment> {
    let line_length = line_cell_len(&line);
    if line_length < length {
        if pad {
            line.push(Segment::new(" ".repeat(length - line_length), style));
        }
        line
    } else if line_length > length {
        let mut new_line = Vec::with_capacity(line.len());
        let mut line_length = 0;
        for segment in line {
            let segment_length = segment.cell_len();
            if line_length + segment_length < length {
                line_length += segment_length;
                new_line.push(segment);
            } else {
                let text = set_cell_size(&segment.text, length - line_length);
                new_line.push(Segment::new(text, segment.style));
                break;
            }
        }
        new_line
    } else {
        line
    }
}

/// Total cells of a line (`Segment.get_line_length`).
pub fn line_cell_len(line: &[Segment]) -> usize {
    line.iter().map(Segment::cell_len).sum()
}

/// Rich `Segment.get_shape(lines)` -> (max width, height).
pub fn get_shape(lines: &[Vec<Segment>]) -> (usize, usize) {
    let width = lines.iter().map(|l| line_cell_len(l)).max().unwrap_or(0);
    (width, lines.len())
}

/// Rich `Segment.set_shape(lines, width, height, style, new_lines=False)`.
pub fn set_shape(
    lines: Vec<Vec<Segment>>,
    width: usize,
    height: Option<usize>,
    style: Option<Style>,
) -> Vec<Vec<Segment>> {
    let height = match height {
        Some(h) if h > 0 => h,
        _ => lines.len(),
    };
    // Rich slices to `height` and then (by accident) re-adjusts all lines, so extra lines
    // are kept; replicate that.
    let mut shaped: Vec<Vec<Segment>> = lines
        .into_iter()
        .map(|line| adjust_line_length(line, width, style, true))
        .collect();
    while shaped.len() < height {
        shaped.push(vec![Segment::new(" ".repeat(width), style)]);
    }
    shaped
}

/// Vertical alignment of a cell inside a row (Rich `VerticalAlignMethod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VerticalAlign {
    #[default]
    Top,
    Middle,
    Bottom,
}

/// Rich `Segment.align_top/align_middle/align_bottom(lines, width, height, style)`.
pub fn align_lines(
    mut lines: Vec<Vec<Segment>>,
    width: usize,
    height: usize,
    style: Option<Style>,
    vertical: VerticalAlign,
) -> Vec<Vec<Segment>> {
    if lines.len() == height {
        return lines;
    }
    let extra = height as isize - lines.len() as isize;
    lines.truncate(height);
    if extra <= 0 {
        return lines;
    }
    let extra = extra as usize;
    let blank = || vec![Segment::new(" ".repeat(width), style)];
    let (top, bottom) = match vertical {
        VerticalAlign::Top => (0, extra),
        VerticalAlign::Bottom => (extra, 0),
        VerticalAlign::Middle => (extra / 2, extra - extra / 2),
    };
    let mut out = Vec::with_capacity(height);
    out.extend((0..top).map(|_| blank()));
    out.extend(lines);
    out.extend((0..bottom).map(|_| blank()));
    out
}

/// Rich `Segment.simplify(segments)`: merge neighbours with equal styles.
pub fn simplify(segments: Vec<Segment>) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::with_capacity(segments.len());
    for segment in segments {
        match out.last_mut() {
            Some(last) if last.style == segment.style => last.text.push_str(&segment.text),
            _ => out.push(segment),
        }
    }
    out
}

/// Rich `Segment.divide(segments, cuts)`: split a line at the given cell offsets.
pub fn divide(segments: &[Segment], cuts: &[usize]) -> Vec<Vec<Segment>> {
    let mut result = Vec::new();
    let mut split: Vec<Segment> = Vec::new();
    let mut cuts = cuts.iter().copied();
    let mut cut;
    loop {
        match cuts.next() {
            None => return result,
            Some(0) => result.push(Vec::new()),
            Some(c) => {
                cut = c;
                break;
            }
        }
    }
    let mut pos = 0;
    for segment in segments {
        let mut segment = segment.clone();
        while !segment.text.is_empty() {
            let end_pos = pos + cell_len(&segment.text);
            if end_pos < cut {
                split.push(segment);
                pos = end_pos;
                break;
            }
            if end_pos == cut {
                split.push(segment);
                result.push(std::mem::take(&mut split));
                pos = end_pos;
                match cuts.next() {
                    None => return result,
                    Some(c) => cut = c,
                }
                break;
            }
            let (before, after) = segment.split_cells(cut - pos);
            split.push(before);
            result.push(std::mem::take(&mut split));
            segment = after;
            pos = cut;
            match cuts.next() {
                None => {
                    if !split.is_empty() {
                        result.push(split);
                    }
                    return result;
                }
                Some(c) => cut = c,
            }
        }
    }
    result.push(split);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_and_pad() {
        let line = vec![
            Segment::plain("abc"),
            Segment::styled("def", Style::of("bold")),
        ];
        let cropped = adjust_line_length(line.clone(), 4, None, true);
        assert_eq!(
            cropped,
            vec![
                Segment::plain("abc"),
                Segment::styled("d", Style::of("bold"))
            ]
        );
        let padded = adjust_line_length(line, 8, None, true);
        assert_eq!(padded.last().unwrap().text, "  ");
    }

    #[test]
    fn split_wide() {
        let (a, b) = Segment::plain("中文字").split_cells(3);
        assert_eq!((a.text.as_str(), b.text.as_str()), ("中 ", " 字"));
    }

    #[test]
    fn terminator() {
        let lines = split_lines_terminator(vec![Segment::plain("a\nb\n"), Segment::plain("c")]);
        assert_eq!(lines.len(), 3);
        assert!(lines[0].1 && lines[1].1 && !lines[2].1);
    }
}
