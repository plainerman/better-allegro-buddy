//! Rich `Box` (rich/box.py): the characters tables (and panels) are drawn with.

use super::data::{BOXES, PLAIN_HEADED_SUBSTITUTIONS};

/// Rich `Box`, parsed from Rich's 8-line definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoxStyle {
    pub ascii: bool,
    pub top_left: char,
    pub top: char,
    pub top_divider: char,
    pub top_right: char,
    pub head_left: char,
    pub head_vertical: char,
    pub head_right: char,
    pub head_row_left: char,
    pub head_row_horizontal: char,
    pub head_row_cross: char,
    pub head_row_right: char,
    pub mid_left: char,
    pub mid_vertical: char,
    pub mid_right: char,
    pub row_left: char,
    pub row_horizontal: char,
    pub row_cross: char,
    pub row_right: char,
    pub foot_row_left: char,
    pub foot_row_horizontal: char,
    pub foot_row_cross: char,
    pub foot_row_right: char,
    pub foot_left: char,
    pub foot_vertical: char,
    pub foot_right: char,
    pub bottom_left: char,
    pub bottom: char,
    pub bottom_divider: char,
    pub bottom_right: char,
}

/// Which separator row [`BoxStyle::get_row`] draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowLevel {
    Head,
    Row,
    Foot,
    Mid,
}

impl BoxStyle {
    /// `Box(definition)`: 8 lines of 4 characters.
    pub fn parse(definition: &str, ascii: bool) -> Option<BoxStyle> {
        let lines: Vec<Vec<char>> = definition.lines().map(|l| l.chars().collect()).collect();
        if lines.len() != 8 || lines.iter().any(|l| l.len() != 4) {
            return None;
        }
        let c = |line: usize, i: usize| lines[line][i];
        Some(BoxStyle {
            ascii,
            top_left: c(0, 0),
            top: c(0, 1),
            top_divider: c(0, 2),
            top_right: c(0, 3),
            head_left: c(1, 0),
            head_vertical: c(1, 2),
            head_right: c(1, 3),
            head_row_left: c(2, 0),
            head_row_horizontal: c(2, 1),
            head_row_cross: c(2, 2),
            head_row_right: c(2, 3),
            mid_left: c(3, 0),
            mid_vertical: c(3, 2),
            mid_right: c(3, 3),
            row_left: c(4, 0),
            row_horizontal: c(4, 1),
            row_cross: c(4, 2),
            row_right: c(4, 3),
            foot_row_left: c(5, 0),
            foot_row_horizontal: c(5, 1),
            foot_row_cross: c(5, 2),
            foot_row_right: c(5, 3),
            foot_left: c(6, 0),
            foot_vertical: c(6, 2),
            foot_right: c(6, 3),
            bottom_left: c(7, 0),
            bottom: c(7, 1),
            bottom_divider: c(7, 2),
            bottom_right: c(7, 3),
        })
    }

    /// A box by its Rich name (`"SIMPLE"`, `"ROUNDED"`, ...; also rich-click's
    /// `"HORIZONTALS_TOP"` / `"HORIZONTALS_DOUBLE_TOP"`).
    pub fn named(name: &str) -> Option<BoxStyle> {
        BOXES
            .iter()
            .find(|(n, _, _)| *n == name)
            .and_then(|(_, definition, ascii)| BoxStyle::parse(definition, *ascii))
    }

    /// The Rich name of this box, if it is one of the predefined ones.
    pub fn name(&self) -> Option<&'static str> {
        BOXES
            .iter()
            .find(|(_, definition, ascii)| {
                BoxStyle::parse(definition, *ascii).as_ref() == Some(self)
            })
            .map(|(n, _, _)| *n)
    }

    /// Rich `get_plain_headed_box()`.
    pub fn get_plain_headed_box(&self) -> BoxStyle {
        self.name()
            .and_then(|name| {
                PLAIN_HEADED_SUBSTITUTIONS
                    .iter()
                    .find(|(from, _)| *from == name)
            })
            .and_then(|(_, to)| BoxStyle::named(to))
            .unwrap_or(*self)
    }

    fn line(
        left: char,
        fill: char,
        divider: char,
        right: char,
        widths: &[usize],
        edges: bool,
    ) -> String {
        let mut out = String::new();
        if edges {
            out.push(left);
        }
        for (i, &width) in widths.iter().enumerate() {
            out.extend(std::iter::repeat_n(fill, width));
            if i + 1 < widths.len() {
                out.push(divider);
            }
        }
        if edges {
            out.push(right);
        }
        out
    }

    /// Rich `get_top(widths)`.
    pub fn get_top(&self, widths: &[usize]) -> String {
        Self::line(
            self.top_left,
            self.top,
            self.top_divider,
            self.top_right,
            widths,
            true,
        )
    }

    /// Rich `get_row(widths, level, edge)`.
    pub fn get_row(&self, widths: &[usize], level: RowLevel, edge: bool) -> String {
        let (left, horizontal, cross, right) = match level {
            RowLevel::Head => (
                self.head_row_left,
                self.head_row_horizontal,
                self.head_row_cross,
                self.head_row_right,
            ),
            RowLevel::Row => (
                self.row_left,
                self.row_horizontal,
                self.row_cross,
                self.row_right,
            ),
            RowLevel::Mid => (self.mid_left, ' ', self.mid_vertical, self.mid_right),
            RowLevel::Foot => (
                self.foot_row_left,
                self.foot_row_horizontal,
                self.foot_row_cross,
                self.foot_row_right,
            ),
        };
        Self::line(left, horizontal, cross, right, widths, edge)
    }

    /// Rich `get_bottom(widths)`.
    pub fn get_bottom(&self, widths: &[usize]) -> String {
        Self::line(
            self.bottom_left,
            self.bottom,
            self.bottom_divider,
            self.bottom_right,
            widths,
            true,
        )
    }
}

macro_rules! named_boxes {
    ($($name:ident),* $(,)?) => {
        $(
            pub static $name: std::sync::LazyLock<BoxStyle> =
                std::sync::LazyLock::new(|| BoxStyle::named(stringify!($name)).unwrap());
        )*
    };
}

named_boxes!(
    ASCII,
    ASCII2,
    ASCII_DOUBLE_HEAD,
    SQUARE,
    SQUARE_DOUBLE_HEAD,
    MINIMAL,
    MINIMAL_HEAVY_HEAD,
    MINIMAL_DOUBLE_HEAD,
    SIMPLE,
    SIMPLE_HEAD,
    SIMPLE_HEAVY,
    HORIZONTALS,
    ROUNDED,
    HEAVY,
    HEAVY_EDGE,
    HEAVY_HEAD,
    DOUBLE,
    DOUBLE_EDGE,
    MARKDOWN,
    HORIZONTALS_TOP,
    HORIZONTALS_DOUBLE_TOP,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_box_lines() {
        assert_eq!(SIMPLE.get_top(&[2, 3]), " ".repeat(8));
        assert_eq!(SIMPLE.get_row(&[2, 3], RowLevel::Head, false), "──────");
        assert_eq!(SIMPLE.get_row(&[2, 3], RowLevel::Row, false), "      ");
        assert_eq!(HEAVY_HEAD.get_plain_headed_box(), *SQUARE);
        assert_eq!(ROUNDED.get_top(&[1]), "╭─╮");
    }
}
