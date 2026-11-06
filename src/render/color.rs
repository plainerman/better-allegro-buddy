//! Rich `Color` (rich/color.py): parsing, SGR codes and downgrading between colour systems.

use super::data::{ANSI_COLOR_NAMES, EIGHT_BIT_PALETTE, STANDARD_PALETTE};

/// Rich `ColorSystem` (Console.color_system). `None` at the console level = no colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ColorSystem {
    Standard,
    EightBit,
    TrueColor,
}

/// Rich `Color` (without the display name).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    /// `default` (SGR 39/49).
    Default,
    /// Standard + bright colours, 0..=15 (`red` = 1, `bright_red` = 9, `color(9)`).
    Standard(u8),
    /// `color(n)` / named 256-palette colours, 16..=255.
    EightBit(u8),
    /// `#rrggbb` / `rgb(r,g,b)`.
    TrueColor(u8, u8, u8),
}

/// Rich `ColorParseError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorParseError(pub String);

fn from_number(number: u8) -> Color {
    if number < 16 {
        Color::Standard(number)
    } else {
        Color::EightBit(number)
    }
}

/// Python `int(s)` for the digit/space strings Rich's colour regex admits.
fn py_int(s: &str) -> Option<u32> {
    let s = s.trim();
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse::<u32>().ok().or(Some(u32::MAX))
}

impl Color {
    /// Rich `Color.parse(color)`.
    pub fn parse(color: &str) -> Result<Color, ColorParseError> {
        let original = color;
        let color = color.trim().to_lowercase();
        let color = color.as_str();
        if color == "default" {
            return Ok(Color::Default);
        }
        if let Ok(index) = ANSI_COLOR_NAMES.binary_search_by(|(name, _)| (*name).cmp(color)) {
            return Ok(from_number(ANSI_COLOR_NAMES[index].1));
        }
        let invalid = || ColorParseError(format!("{original:?} is not a valid color"));
        if let Some(hex) = color.strip_prefix('#') {
            if hex.len() == 6 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
                let component = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).unwrap();
                return Ok(Color::TrueColor(component(0), component(2), component(4)));
            }
            return Err(invalid());
        }
        if let Some(inner) = color
            .strip_prefix("color(")
            .and_then(|s| s.strip_suffix(')'))
        {
            if (1..=3).contains(&inner.len()) && inner.bytes().all(|b| b.is_ascii_digit()) {
                let number: u32 = inner.parse().unwrap();
                if number > 255 {
                    return Err(ColorParseError(format!(
                        "color number must be <= 255 in {color:?}"
                    )));
                }
                return Ok(from_number(number as u8));
            }
            return Err(invalid());
        }
        if let Some(inner) = color.strip_prefix("rgb(").and_then(|s| s.strip_suffix(')')) {
            let admissible = !inner.is_empty()
                && inner
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == ',' || c.is_whitespace());
            if !admissible {
                return Err(invalid());
            }
            let components: Vec<&str> = inner.split(',').collect();
            if components.len() != 3 {
                return Err(ColorParseError(format!(
                    "expected three components in {original:?}"
                )));
            }
            let mut values = [0u32; 3];
            for (value, component) in values.iter_mut().zip(&components) {
                *value = py_int(component).ok_or_else(invalid)?;
            }
            if values.iter().any(|&v| v > 255) {
                return Err(ColorParseError(format!(
                    "color components must be <= 255 in {original:?}"
                )));
            }
            return Ok(Color::TrueColor(
                values[0] as u8,
                values[1] as u8,
                values[2] as u8,
            ));
        }
        Err(invalid())
    }

    /// Rich `Color.get_ansi_codes(foreground)`, appended to `out` (`;`-separated).
    pub fn push_ansi_codes(&self, foreground: bool, out: &mut String) {
        use std::fmt::Write;
        if !out.is_empty() {
            out.push(';');
        }
        match *self {
            Color::Default => out.push_str(if foreground { "39" } else { "49" }),
            Color::Standard(n) => {
                let (fore, back) = if n < 8 { (30, 40) } else { (82, 92) };
                let _ = write!(out, "{}", if foreground { fore } else { back } + n as u32);
            }
            Color::EightBit(n) => {
                let _ = write!(out, "{};5;{}", if foreground { 38 } else { 48 }, n);
            }
            Color::TrueColor(r, g, b) => {
                let _ = write!(
                    out,
                    "{};2;{};{};{}",
                    if foreground { 38 } else { 48 },
                    r,
                    g,
                    b
                );
            }
        }
    }

    fn system(&self) -> ColorSystem {
        match self {
            Color::Default | Color::Standard(_) => ColorSystem::Standard,
            Color::EightBit(_) => ColorSystem::EightBit,
            Color::TrueColor(..) => ColorSystem::TrueColor,
        }
    }

    /// Rich `Color.downgrade(system)`.
    pub fn downgrade(&self, system: ColorSystem) -> Color {
        if matches!(self, Color::Default) || self.system() == system {
            return *self;
        }
        match (system, *self) {
            (ColorSystem::EightBit, Color::TrueColor(red, green, blue)) => {
                let (_, l, s) = rgb_to_hls(
                    red as f64 / 255.0,
                    green as f64 / 255.0,
                    blue as f64 / 255.0,
                );
                if s < 0.15 {
                    let gray = (l * 25.0).round_ties_even() as i64;
                    let number = match gray {
                        0 => 16,
                        25 => 231,
                        g => 231 + g,
                    };
                    return Color::EightBit(number as u8);
                }
                let six = |c: u8| {
                    let c = c as f64;
                    if c < 95.0 {
                        c / 95.0
                    } else {
                        1.0 + (c - 95.0) / 40.0
                    }
                };
                let number = 16.0
                    + 36.0 * six(red).round_ties_even()
                    + 6.0 * six(green).round_ties_even()
                    + six(blue).round_ties_even();
                Color::EightBit(number as u8)
            }
            (ColorSystem::Standard, Color::TrueColor(r, g, b)) => {
                Color::Standard(standard_palette_match((r, g, b)))
            }
            (ColorSystem::Standard, Color::EightBit(n)) => {
                Color::Standard(standard_palette_match(EIGHT_BIT_PALETTE[n as usize]))
            }
            _ => *self,
        }
    }
}

fn rgb_to_hls(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let maxc = r.max(g).max(b);
    let minc = r.min(g).min(b);
    let sumc = maxc + minc;
    let rangec = maxc - minc;
    let l = sumc / 2.0;
    if minc == maxc {
        return (0.0, l, 0.0);
    }
    let s = if l <= 0.5 {
        rangec / sumc
    } else {
        rangec / (2.0 - maxc - minc)
    };
    (0.0, l, s)
}

/// Rich `Palette.match` over `STANDARD_PALETTE` (first minimum wins).
fn standard_palette_match((red1, green1, blue1): (u8, u8, u8)) -> u8 {
    let (red1, green1, blue1) = (red1 as i64, green1 as i64, blue1 as i64);
    let distance = |&(red2, green2, blue2): &(u8, u8, u8)| {
        let (red2, green2, blue2) = (red2 as i64, green2 as i64, blue2 as i64);
        let red_mean = (red1 + red2).div_euclid(2);
        let red = red1 - red2;
        let green = green1 - green2;
        let blue = blue1 - blue2;
        let value = (((512 + red_mean) * red * red) >> 8)
            + 4 * green * green
            + (((767 - red_mean) * blue * blue) >> 8);
        (value as f64).sqrt()
    };
    let mut best = 0;
    let mut best_distance = f64::INFINITY;
    for (index, color) in STANDARD_PALETTE.iter().enumerate() {
        let d = distance(color);
        if d < best_distance {
            best = index;
            best_distance = d;
        }
    }
    best as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_colors() {
        assert_eq!(Color::parse("red"), Ok(Color::Standard(1)));
        assert_eq!(Color::parse(" Bright_Red "), Ok(Color::Standard(9)));
        assert_eq!(Color::parse("grey0"), Ok(Color::EightBit(16)));
        assert_eq!(Color::parse("color(7)"), Ok(Color::Standard(7)));
        assert_eq!(Color::parse("color(200)"), Ok(Color::EightBit(200)));
        assert!(Color::parse("color(256)").is_err());
        assert_eq!(Color::parse("#ff0080"), Ok(Color::TrueColor(255, 0, 128)));
        assert_eq!(Color::parse("rgb(1, 2,3)"), Ok(Color::TrueColor(1, 2, 3)));
        assert!(Color::parse("rgb(1,2)").is_err());
        assert!(Color::parse("nope").is_err());
        assert_eq!(Color::parse("default"), Ok(Color::Default));
    }

    #[test]
    fn downgrade() {
        assert_eq!(
            Color::TrueColor(255, 0, 0).downgrade(ColorSystem::EightBit),
            Color::EightBit(196)
        );
        assert_eq!(
            Color::TrueColor(128, 128, 128).downgrade(ColorSystem::EightBit),
            Color::EightBit(244)
        );
        assert_eq!(
            Color::TrueColor(255, 0, 0).downgrade(ColorSystem::Standard),
            Color::Standard(1)
        );
        assert_eq!(
            Color::EightBit(196).downgrade(ColorSystem::Standard),
            Color::Standard(1)
        );
        assert_eq!(
            Color::Standard(1).downgrade(ColorSystem::TrueColor),
            Color::Standard(1)
        );
    }
}
