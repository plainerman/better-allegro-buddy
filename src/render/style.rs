//! Rich `Style` (rich/style.py): parsing style definitions, combining, and SGR rendering.
//!
//! * Attributes are a bitmask with a "set" mask (Rich `_attributes` / `_set_attributes`), so
//!   `not bold` can switch bold off when combining. SGR codes are emitted in bit order
//!   (bold=1, dim=2, italic=3, underline=4, blink=5, blink2=6, reverse=7, conceal=8,
//!   strike=9, underline2=21, frame=51, encircle=52, overline=53), then the foreground,
//!   then the background colour, joined by `;`.
//! * `Style.render(text)` = `ESC[{codes}m{text}ESC[0m`; no codes -> text unchanged.
//! * Links and meta are not supported (Rich renders links with random ids, so they could
//!   never be byte-identical anyway); `link URL` parses but is ignored.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

pub use super::color::{Color, ColorSystem};

/// An immutable, `Copy` style (Rich `Style` without links/meta).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Style {
    /// Attribute bits (see module docs for the bit numbering).
    pub attributes: u16,
    /// Which attribute bits are set (on or explicitly off).
    pub set_attributes: u16,
    pub color: Option<Color>,
    pub bgcolor: Option<Color>,
}

/// Error for an invalid style definition (Rich `StyleSyntaxError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StyleSyntaxError(pub String);

impl std::fmt::Display for StyleSyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

pub const BOLD: u16 = 1;
pub const DIM: u16 = 1 << 1;
pub const ITALIC: u16 = 1 << 2;
pub const UNDERLINE: u16 = 1 << 3;
pub const BLINK: u16 = 1 << 4;
pub const BLINK2: u16 = 1 << 5;
pub const REVERSE: u16 = 1 << 6;
pub const CONCEAL: u16 = 1 << 7;
pub const STRIKE: u16 = 1 << 8;
pub const UNDERLINE2: u16 = 1 << 9;
pub const FRAME: u16 = 1 << 10;
pub const ENCIRCLE: u16 = 1 << 11;
pub const OVERLINE: u16 = 1 << 12;

const SGR: [&str; 13] = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "21", "51", "52", "53",
];

/// Rich `Style.STYLE_ATTRIBUTES`.
fn attribute_bit(word: &str) -> Option<u16> {
    Some(match word {
        "dim" | "d" => DIM,
        "bold" | "b" => BOLD,
        "italic" | "i" => ITALIC,
        "underline" | "u" => UNDERLINE,
        "blink" => BLINK,
        "blink2" => BLINK2,
        "reverse" | "r" => REVERSE,
        "conceal" | "c" => CONCEAL,
        "strike" | "s" => STRIKE,
        "underline2" | "uu" => UNDERLINE2,
        "frame" => FRAME,
        "encircle" => ENCIRCLE,
        "overline" | "o" => OVERLINE,
        _ => return None,
    })
}

static PARSE_CACHE: LazyLock<Mutex<HashMap<String, Result<Style, StyleSyntaxError>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Python `str.split()` whitespace (`str.isspace`).
pub(crate) fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

impl Style {
    /// The null style (renders text unchanged).
    pub const NULL: Style = Style {
        attributes: 0,
        set_attributes: 0,
        color: None,
        bgcolor: None,
    };

    /// A style with the given attribute bits switched on.
    pub const fn attrs(bits: u16) -> Style {
        Style {
            attributes: bits,
            set_attributes: bits,
            color: None,
            bgcolor: None,
        }
    }

    /// `Style(color=...)`.
    pub const fn fg(color: Color) -> Style {
        Style {
            attributes: 0,
            set_attributes: 0,
            color: Some(color),
            bgcolor: None,
        }
    }

    /// `Style(bgcolor=...)` (Rich `background_style`).
    pub const fn bg(color: Option<Color>) -> Style {
        Style {
            attributes: 0,
            set_attributes: 0,
            color: None,
            bgcolor: color,
        }
    }

    /// Parse a Rich style definition such as `"bold red"`, `"dim"`, `"on blue"`,
    /// `"not italic #ff0000"`. Empty string / `"none"` -> [`Style::NULL`]. Cached.
    pub fn parse(definition: &str) -> Result<Style, StyleSyntaxError> {
        if let Some(cached) = PARSE_CACHE.lock().unwrap().get(definition) {
            return cached.clone();
        }
        let result = Self::parse_uncached(definition);
        PARSE_CACHE
            .lock()
            .unwrap()
            .insert(definition.to_owned(), result.clone());
        result
    }

    fn parse_uncached(definition: &str) -> Result<Style, StyleSyntaxError> {
        if definition.trim_matches(py_isspace) == "none" || definition.is_empty() {
            return Ok(Style::NULL);
        }
        let mut style = Style::NULL;
        let mut words = definition.split(py_isspace).filter(|w| !w.is_empty());
        while let Some(original) = words.next() {
            let word = original.to_lowercase();
            match word.as_str() {
                "on" => {
                    let word = words.next().unwrap_or("").to_lowercase();
                    if word.is_empty() {
                        return Err(StyleSyntaxError("color expected after 'on'".into()));
                    }
                    let color = Color::parse(&word).map_err(|e| {
                        StyleSyntaxError(format!(
                            "unable to parse {word:?} as background color; {}",
                            e.0
                        ))
                    })?;
                    style.bgcolor = Some(color);
                }
                "not" => {
                    let word = words.next().unwrap_or("").to_lowercase();
                    let bit = attribute_bit(&word).ok_or_else(|| {
                        StyleSyntaxError(format!(
                            "expected style attribute after 'not', found {word:?}"
                        ))
                    })?;
                    style.set_attributes |= bit;
                    style.attributes &= !bit;
                }
                "link" => {
                    if words.next().is_none() {
                        return Err(StyleSyntaxError("URL expected after 'link'".into()));
                    }
                }
                _ => {
                    if let Some(bit) = attribute_bit(&word) {
                        style.set_attributes |= bit;
                        style.attributes |= bit;
                    } else {
                        let color = Color::parse(&word).map_err(|e| {
                            StyleSyntaxError(format!("unable to parse {word:?} as color; {}", e.0))
                        })?;
                        style.color = Some(color);
                    }
                }
            }
        }
        Ok(style)
    }

    /// Parse a definition that is a compile-time constant in bab; panics on bad syntax.
    pub fn of(definition: &str) -> Style {
        Style::parse(definition).unwrap_or_else(|e| panic!("bad style {definition:?}: {e:?}"))
    }

    /// Rich `not style` (`Style._null`).
    pub fn is_null(&self) -> bool {
        self.set_attributes == 0 && self.color.is_none() && self.bgcolor.is_none()
    }

    /// Rich `self + other`: `other` wins where it sets something.
    pub fn combine(&self, other: &Style) -> Style {
        if other.is_null() {
            return *self;
        }
        if self.is_null() {
            return *other;
        }
        Style {
            attributes: (self.attributes & !other.set_attributes)
                | (other.attributes & other.set_attributes),
            set_attributes: self.set_attributes | other.set_attributes,
            color: other.color.or(self.color),
            bgcolor: other.bgcolor.or(self.bgcolor),
        }
    }

    /// Rich `Style.without_color` (for `NO_COLOR`).
    pub fn without_color(&self) -> Style {
        Style {
            color: None,
            bgcolor: None,
            ..*self
        }
    }

    /// Rich `Style.background_style`.
    pub fn background_style(&self) -> Style {
        Style::bg(self.bgcolor)
    }

    /// `Some(true/false)` if the attribute bit is set, `None` otherwise.
    pub fn get(&self, bit: u16) -> Option<bool> {
        (self.set_attributes & bit != 0).then_some(self.attributes & bit != 0)
    }

    /// SGR parameter string (`"1;31"`) for the given colour system; empty if nothing to emit.
    pub fn sgr_codes(&self, color_system: ColorSystem) -> String {
        let mut out = String::new();
        let attributes = self.attributes & self.set_attributes;
        for (bit, code) in SGR.iter().enumerate() {
            if attributes & (1 << bit) != 0 {
                if !out.is_empty() {
                    out.push(';');
                }
                out.push_str(code);
            }
        }
        if let Some(color) = self.color {
            color
                .downgrade(color_system)
                .push_ansi_codes(true, &mut out);
        }
        if let Some(bgcolor) = self.bgcolor {
            bgcolor
                .downgrade(color_system)
                .push_ansi_codes(false, &mut out);
        }
        out
    }

    /// Rich `Style.render(text, color_system=...)`, appended to `out`.
    pub fn render_into(&self, text: &str, color_system: Option<ColorSystem>, out: &mut String) {
        let Some(system) = color_system else {
            out.push_str(text);
            return;
        };
        if text.is_empty() {
            return;
        }
        let codes = self.sgr_codes(system);
        if codes.is_empty() {
            out.push_str(text);
        } else {
            out.push_str("\x1b[");
            out.push_str(&codes);
            out.push('m');
            out.push_str(text);
            out.push_str("\x1b[0m");
        }
    }

    /// Rich `Style.render(text, color_system=...)`.
    pub fn render(&self, text: &str, color_system: Option<ColorSystem>) -> String {
        let mut out = String::with_capacity(text.len() + 12);
        self.render_into(text, color_system, &mut out);
        out
    }
}

impl std::ops::Add for Style {
    type Output = Style;
    fn add(self, other: Style) -> Style {
        self.combine(&other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sgr(definition: &str) -> String {
        Style::of(definition).sgr_codes(ColorSystem::TrueColor)
    }

    #[test]
    fn parse_and_codes() {
        assert_eq!(sgr("bold red"), "1;31");
        assert_eq!(sgr("dim"), "2");
        assert_eq!(sgr("bold bright_green on blue"), "1;92;44");
        assert_eq!(sgr("not bold"), "");
        assert_eq!(sgr("u i b s"), "1;3;4;9");
        assert_eq!(sgr("#ff0000 on color(200)"), "38;2;255;0;0;48;5;200");
        assert_eq!(sgr("underline2 frame encircle overline"), "21;51;52;53");
        assert_eq!(Style::of("none"), Style::NULL);
        assert_eq!(Style::of(""), Style::NULL);
        assert!(Style::parse("on").is_err());
        assert!(Style::parse("not foo").is_err());
        assert!(Style::parse("bold blurple").is_err());
    }

    #[test]
    fn combine_and_render() {
        let combined = Style::of("bold red") + Style::of("not bold blue");
        assert_eq!(combined.sgr_codes(ColorSystem::Standard), "34");
        assert_eq!(
            Style::of("bold").render("x", Some(ColorSystem::Standard)),
            "\x1b[1mx\x1b[0m"
        );
        assert_eq!(Style::of("bold").render("x", None), "x");
        assert_eq!(
            Style::of("bold").render("", Some(ColorSystem::Standard)),
            ""
        );
        assert_eq!(
            Style::of("bold red")
                .without_color()
                .sgr_codes(ColorSystem::Standard),
            "1"
        );
        assert_eq!(
            Style::of("#ff0000").sgr_codes(ColorSystem::EightBit),
            "38;5;196"
        );
        assert_eq!(Style::of("#ff0000").sgr_codes(ColorSystem::Standard), "31");
    }
}
