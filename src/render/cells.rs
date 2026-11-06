//! Terminal cell widths, ported from Rich 15 `rich.cells` with Rich's own width table of
//! Unicode 17.0.0 (`rich._unicode_data`, generated into [`super::cell_tables`]). Known
//! divergence: Rich picks an older table when `UNICODE_VERSION` is set; bab ignores it.
//! Control characters count 0; zero-width joiners and VS16 are handled like Rich's
//! `_cell_len` / `split_graphemes`.

use super::cell_tables::{WIDE, WIDTHS};
use super::data::NARROW_TO_WIDE;

const ZWJ: char = '\u{200d}';
const VS16: char = '\u{fe0f}';

/// Width of `cp` in a table of flat `start | WIDE?, end` pairs (Rich's binary search).
fn table_width(table: &[u32], cp: u32) -> usize {
    let ranges = table.len() / 2;
    if cp > table[table.len() - 1] {
        return 1;
    }
    let (mut low, mut high) = (0, ranges);
    while low < high {
        let mid = (low + high) / 2;
        let (start, end) = (table[2 * mid] & !WIDE, table[2 * mid + 1]);
        if cp < start {
            high = mid;
        } else if cp > end {
            low = mid + 1;
        } else {
            return if table[2 * mid] & WIDE != 0 { 2 } else { 0 };
        }
    }
    1
}

/// Rich `get_character_cell_size(ch)`.
pub fn char_width(ch: char) -> usize {
    let cp = ch as u32;
    if (cp != 0 && cp < 32) || (0x7f..0xa0).contains(&cp) {
        return 0;
    }
    if (0x20..0x7f).contains(&cp) {
        return 1;
    }
    table_width(WIDTHS, cp)
}

fn is_narrow_to_wide(ch: char) -> bool {
    NARROW_TO_WIDE.binary_search(&ch).is_ok()
}

/// Rich `_is_single_cell_widths` (only the ASCII part; the other ranges Rich lists are all
/// width 1 in its table too, so the slow path gives the same answer).
fn is_ascii_printable(text: &str) -> bool {
    text.bytes().all(|b| (0x20..0x7f).contains(&b))
}

/// Rich `cell_len(text)`.
pub fn cell_len(text: &str) -> usize {
    if is_ascii_printable(text) {
        return text.len();
    }
    if !text.contains([ZWJ, VS16]) {
        return text.chars().map(char_width).sum();
    }
    let mut total = 0;
    let mut last_measured: Option<char> = None;
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch == ZWJ {
            // Rich skips the joined character too.
            chars.next();
            continue;
        }
        if ch == VS16 {
            if let Some(last) = last_measured.take() {
                total += usize::from(is_narrow_to_wide(last));
            }
            continue;
        }
        let width = char_width(ch);
        if width > 0 {
            last_measured = Some(ch);
            total += width;
        }
    }
    total
}

/// One grapheme: `[start, end)` in chars and its cell width (Rich `CellSpan`).
type CellSpan = (usize, usize, usize);

/// Rich `split_graphemes(text)` over `chars`.
fn split_graphemes(chars: &[char]) -> (Vec<CellSpan>, usize) {
    let count = chars.len();
    let mut index = 0;
    let mut last_measured: Option<char> = None;
    let mut total = 0;
    let mut spans: Vec<CellSpan> = Vec::with_capacity(count);
    while index < count {
        let ch = chars[index];
        if ch == ZWJ || ch == VS16 {
            if spans.is_empty() {
                spans.push((index, index + 1, 0));
                index += 1;
                continue;
            }
            if ch == ZWJ {
                index += if index < count - 1 { 2 } else { 1 };
                let last = spans.last_mut().unwrap();
                last.1 = index;
            } else {
                index += 1;
                let last = spans.last_mut().unwrap();
                if let Some(measured) = last_measured
                    && is_narrow_to_wide(measured)
                {
                    last_measured = None;
                    last.2 += 1;
                    total += 1;
                }
                last.1 = index;
            }
            continue;
        }
        let width = char_width(ch);
        if width > 0 {
            last_measured = Some(ch);
            spans.push((index, index + 1, width));
            total += width;
        } else if let Some(last) = spans.last_mut() {
            index += 1;
            last.1 = index;
            continue;
        } else {
            spans.push((index, index + 1, 0));
        }
        index += 1;
    }
    (spans, total)
}

/// Rich `_split_text(text, cell_position)`: split at a cell offset; a wide character
/// straddling the cut becomes a space on both sides.
fn split_text_slow(text: &str, cell_position: usize) -> (String, String) {
    if cell_position == 0 {
        return (String::new(), text.to_owned());
    }
    let chars: Vec<char> = text.chars().collect();
    let (spans, cell_length) = split_graphemes(&chars);
    let collect = |a: usize, b: usize| chars[a..b].iter().collect::<String>();
    // Rich guesses a start offset and walks; the walk is deterministic, so a linear scan
    // from the start gives the same answer.
    let _ = cell_length;
    let mut left_size = 0;
    for &(start, end, size) in &spans {
        if left_size == cell_position {
            return (collect(0, start), collect(start, chars.len()));
        }
        if left_size + size > cell_position {
            return (
                collect(0, start) + " ",
                " ".to_owned() + &collect(end, chars.len()),
            );
        }
        left_size += size;
    }
    (text.to_owned(), String::new())
}

/// Rich `split_text(text, cell_position)`.
pub fn split_text(text: &str, cell_position: usize) -> (String, String) {
    if is_ascii_printable(text) {
        let cut = cell_position.min(text.len());
        return (text[..cut].to_owned(), text[cut..].to_owned());
    }
    split_text_slow(text, cell_position)
}

/// Rich `set_cell_size(text, total)`: crop (never splitting a wide char; pad with a space
/// if needed) or pad with spaces to exactly `total` cells.
pub fn set_cell_size(text: &str, total: usize) -> String {
    if is_ascii_printable(text) {
        let size = text.len();
        if size < total {
            let mut out = String::with_capacity(total);
            out.push_str(text);
            out.extend(std::iter::repeat_n(' ', total - size));
            return out;
        }
        return text[..total].to_owned();
    }
    if total == 0 {
        return String::new();
    }
    let size = cell_len(text);
    if size == total {
        return text.to_owned();
    }
    if size < total {
        let mut out = String::with_capacity(text.len() + total - size);
        out.push_str(text);
        out.extend(std::iter::repeat_n(' ', total - size));
        return out;
    }
    split_text_slow(text, total).0
}

/// Rich `chop_cells(text, width)`: split into pieces of at most `width` cells.
pub fn chop_cells(text: &str, width: usize) -> Vec<String> {
    if is_ascii_printable(text) {
        if width == 0 {
            // Python's range() with step 0 raises; Rich never calls this with width 0.
            return vec![text.to_owned()];
        }
        return text
            .as_bytes()
            .chunks(width)
            .map(|c| String::from_utf8(c.to_vec()).unwrap())
            .collect();
    }
    let chars: Vec<char> = text.chars().collect();
    let (spans, _) = split_graphemes(&chars);
    let mut line_size = 0;
    let mut lines = Vec::new();
    let mut line_offset = 0;
    for &(start, _end, size) in &spans {
        if line_size + size > width {
            lines.push(chars[line_offset..start].iter().collect());
            line_offset = start;
            line_size = 0;
        }
        line_size += size;
    }
    if line_size > 0 {
        lines.push(chars[line_offset..].iter().collect());
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_widths() {
        // U+1FAE0 MELTING FACE (Unicode 14), U+231A WATCH (wide since Unicode 9)
        assert_eq!(table_width(WIDTHS, 0x1fae0), 2);
        assert_eq!(table_width(WIDTHS, 0x231a), 2);
        assert_eq!(table_width(WIDTHS, 0x301), 0);
        assert_eq!(table_width(WIDTHS, 0x10ffff), 1);
        assert_eq!(table_width(WIDTHS, 0), 0);
    }

    #[test]
    fn widths_of_glyphs_bab_prints() {
        for ch in [
            'ⓘ', '·', '─', '│', '✓', '✗', '…', '→', '≤', '⠋', '⚠', '●', '•',
        ] {
            assert_eq!(char_width(ch), 1, "{ch}");
        }
        assert_eq!(char_width('中'), 2);
        assert_eq!(char_width('\u{301}'), 0);
        assert_eq!(char_width('\x1b'), 0);
        assert_eq!(cell_len("a中b"), 4);
        assert_eq!(cell_len("ⓘ hint"), 6);
    }

    #[test]
    fn zwj_and_vs16() {
        assert_eq!(cell_len("👩\u{200d}💻"), 2);
        assert_eq!(cell_len("\u{2764}\u{fe0f}"), 2);
        assert_eq!(cell_len("\u{2764}"), 1);
    }

    #[test]
    fn set_cell_size_and_chop() {
        assert_eq!(set_cell_size("abc", 5), "abc  ");
        assert_eq!(set_cell_size("abcdef", 3), "abc");
        assert_eq!(set_cell_size("中文字", 3), "中 ");
        assert_eq!(set_cell_size("中文字", 4), "中文");
        assert_eq!(chop_cells("abcdefg", 3), ["abc", "def", "g"]);
        assert_eq!(chop_cells("中文字x", 3), ["中", "文", "字x"]);
        assert_eq!(split_text("中文", 1), (" ".to_owned(), " 文".to_owned()));
    }
}
