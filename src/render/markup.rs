//! Rich console markup (`rich.markup.render`).
//!
//! Plain `str` values printed with `Console.print(str)` or used as `Table` cells go through
//! `Console.render_str(text, markup=True)` in Rich: markup tags like `[bold]x[/bold]` become
//! styles, `\[` escapes a bracket, and text that is not a tag (`[0-3]`, `[123]`) stays
//! literal. Tags with unknown style names (`job[abc]`) still disappear (their style
//! resolves to nothing), exactly like Rich. Emoji codes are replaced in the plain parts.
//! `[@handler]` meta tags are accepted and ignored.

use super::emoji::emoji_replace;
use super::style::{Style, py_isspace};
use super::text::{Span, Text};
use super::theme::Theme;
use regex::Regex;
use std::sync::LazyLock;

static RE_TAGS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"((\\*)\[([a-z#/@][^\[]*?)])").unwrap());

/// Rich `MarkupError` message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkupError(pub String);

impl std::fmt::Display for MarkupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone)]
struct Tag {
    name: String,
    parameters: Option<String>,
}

impl Tag {
    fn markup(&self) -> String {
        match &self.parameters {
            None => format!("[{}]", self.name),
            Some(p) => format!("[{}={}]", self.name, p),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Normalized {
    Parsed(Style),
    Raw(String),
}

/// Rich `Style.normalize`: the parsed style, or the stripped lowercase name.
fn normalize(name: &str) -> Normalized {
    match Style::parse(name) {
        Ok(style) => Normalized::Parsed(style),
        Err(_) => Normalized::Raw(name.trim_matches(py_isspace).to_lowercase()),
    }
}

/// The style of `str(open_tag)` as `Console.get_style(.., default=null)` resolves it.
fn tag_style(tag: &Tag, theme: &Theme) -> Style {
    if tag.name.starts_with('@') {
        return Style::NULL;
    }
    match (&tag.parameters, normalize(&tag.name)) {
        (None, Normalized::Parsed(style)) => style,
        (None, Normalized::Raw(name)) => theme.get_style_or_null(&name),
        (Some(parameters), normalized) => {
            let name = match normalized {
                Normalized::Raw(name) => name,
                Normalized::Parsed(_) => tag.name.trim_matches(py_isspace).to_lowercase(),
            };
            theme.get_style_or_null(&format!("{name} {parameters}"))
        }
    }
}

enum Piece<'a> {
    Plain(&'a str),
    Tag(usize, Tag),
}

fn char_offset(s: &str, byte: usize) -> usize {
    if s.is_ascii() {
        byte
    } else {
        s[..byte].chars().count()
    }
}

/// Rich `markup._parse`.
fn parse(markup: &str) -> Vec<Piece<'_>> {
    let mut pieces = Vec::new();
    let mut position = 0;
    for caps in RE_TAGS.captures_iter(markup) {
        let whole = caps.get(0).unwrap();
        let (mut start, end) = (whole.start(), whole.end());
        let escapes = caps.get(2).map_or("", |m| m.as_str());
        let tag_text = caps.get(3).unwrap().as_str();
        if start > position {
            pieces.push(Piece::Plain(&markup[position..start]));
        }
        if !escapes.is_empty() {
            let backslashes = escapes.len() / 2;
            let escaped = escapes.len() % 2 == 1;
            if backslashes > 0 {
                pieces.push(Piece::Plain(&markup[start..start + backslashes]));
                start += backslashes * 2;
            }
            if escaped {
                pieces.push(Piece::Plain(&whole.as_str()[escapes.len()..]));
                position = end;
                continue;
            }
        }
        let (name, parameters) = match tag_text.split_once('=') {
            Some((name, parameters)) => (name, Some(parameters.to_owned())),
            None => (tag_text, None),
        };
        pieces.push(Piece::Tag(
            char_offset(markup, start),
            Tag {
                name: name.to_owned(),
                parameters,
            },
        ));
        position = end;
    }
    if position < markup.len() {
        pieces.push(Piece::Plain(&markup[position..]));
    }
    pieces
}

/// Rich `render(markup, style=base_style, emoji=emoji)` -> `Text`; tag names resolve
/// against `theme`.
pub fn render_with(
    markup: &str,
    base_style: Style,
    theme: &Theme,
    emoji: bool,
) -> Result<Text, MarkupError> {
    let replace = |s: &str| -> String {
        if emoji {
            emoji_replace(s).into_owned()
        } else {
            s.to_owned()
        }
    };
    if !markup.contains('[') {
        return Ok(Text::with_style(replace(markup), base_style));
    }
    let mut text = Text::with_style("", base_style);
    let mut style_stack: Vec<(usize, Tag, Normalized)> = Vec::new();
    let mut spans: Vec<Span> = Vec::new();
    for piece in parse(markup) {
        match piece {
            Piece::Plain(plain) => {
                // Literal backslashes come through as a run of `\` (Rich yields "\\" * n).
                let plain = plain.replace("\\[", "[");
                text.append_styled(&replace(&plain), Style::NULL);
            }
            Piece::Tag(position, tag) => {
                if let Some(rest) = tag.name.strip_prefix('/') {
                    let style_name = rest.trim_matches(py_isspace);
                    let (start, open_tag) = if !style_name.is_empty() {
                        let wanted = normalize(style_name);
                        let found = style_stack.iter().rposition(|(_, _, n)| *n == wanted);
                        match found {
                            Some(index) => {
                                let (start, open, _) = style_stack.remove(index);
                                (start, open)
                            }
                            None => {
                                return Err(MarkupError(format!(
                                    "closing tag '{}' at position {position} doesn't match any open tag",
                                    tag.markup()
                                )));
                            }
                        }
                    } else {
                        match style_stack.pop() {
                            Some((start, open, _)) => (start, open),
                            None => {
                                return Err(MarkupError(format!(
                                    "closing tag '[/]' at position {position} has nothing to close"
                                )));
                            }
                        }
                    };
                    let style = tag_style(&open_tag, theme);
                    spans.push(Span::new(start, text.len(), style));
                } else {
                    let normalized = normalize(&tag.name);
                    style_stack.push((text.len(), tag, normalized));
                }
            }
        }
    }
    let text_length = text.len();
    while let Some((start, tag, _)) = style_stack.pop() {
        spans.push(Span::new(start, text_length, tag_style(&tag, theme)));
    }
    spans.reverse();
    spans.sort_by_key(|s| s.start);
    *text.spans_mut() = spans;
    Ok(text)
}

/// Rich `render(markup, style=base_style)` with the default theme and emoji enabled.
pub fn render(markup: &str, base_style: Style) -> Result<Text, MarkupError> {
    render_with(markup, base_style, &Theme::default(), true)
}

/// Rich `markup.escape(text)`.
pub fn escape(markup: &str) -> String {
    static RE_ESCAPE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(\\*)(\[[a-z#/@][^\[]*?])").unwrap());
    let escaped = RE_ESCAPE.replace_all(markup, |caps: &regex::Captures| {
        let backslashes = &caps[1];
        format!("{backslashes}{backslashes}\\{}", &caps[2])
    });
    if escaped.ends_with('\\') && !escaped.ends_with("\\\\") {
        format!("{escaped}\\")
    } else {
        escaped.into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans(t: &Text) -> Vec<(usize, usize)> {
        t.spans().iter().map(|s| (s.start, s.end)).collect()
    }

    #[test]
    fn tags_and_literals() {
        let t = render("[bold]hi[/bold] [0-3] x", Style::NULL).unwrap();
        assert_eq!(t.plain(), "hi [0-3] x");
        assert_eq!(spans(&t), vec![(0, 2)]);
        let t = render("job[abc]x", Style::NULL).unwrap();
        assert_eq!(t.plain(), "jobx");
        assert_eq!(spans(&t), vec![(3, 4)]);
        let t = render(r"\[bold]x", Style::NULL).unwrap();
        assert_eq!(t.plain(), "[bold]x");
        let t = render("[red]a[bold]b[/]c", Style::NULL).unwrap();
        assert_eq!(t.plain(), "abc");
        assert_eq!(spans(&t), vec![(0, 3), (1, 2)]);
    }

    #[test]
    fn errors() {
        assert_eq!(
            render("a[/x]", Style::NULL).unwrap_err().0,
            "closing tag '[/x]' at position 1 doesn't match any open tag"
        );
        assert_eq!(
            render("[/]", Style::NULL).unwrap_err().0,
            "closing tag '[/]' at position 0 has nothing to close"
        );
    }

    #[test]
    fn escape_roundtrip() {
        assert_eq!(escape("[bold]x"), r"\[bold]x");
        assert_eq!(
            render(&escape("a[bold]b"), Style::NULL).unwrap().plain(),
            "a[bold]b"
        );
    }
}
