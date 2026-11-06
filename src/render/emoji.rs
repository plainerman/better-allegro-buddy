//! Rich `_emoji_replace`: `:name:` codes in console strings become emoji (Rich consoles
//! have `emoji=True` by default, so this applies to every `str` printed or put in a table).

use super::emoji_data::EMOJI;
use regex::{Captures, Regex};
use std::borrow::Cow;
use std::sync::LazyLock;

static EMOJI_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(:(\S*?)(?:(?:\-)(emoji|text))?:)").unwrap());

fn lookup(name: &str) -> Option<&'static str> {
    EMOJI
        .binary_search_by(|(n, _)| (*n).cmp(name))
        .ok()
        .map(|i| EMOJI[i].1)
}

/// Rich `_emoji_replace(text)` (no default variant).
pub fn emoji_replace(text: &str) -> Cow<'_, str> {
    if !text.contains(':') {
        return Cow::Borrowed(text);
    }
    EMOJI_RE.replace_all(text, |caps: &Captures| {
        let name = caps.get(2).map_or("", |m| m.as_str()).to_lowercase();
        match lookup(&name) {
            Some(emoji) => {
                let variant = match caps.get(3).map(|m| m.as_str()) {
                    Some("text") => "\u{fe0e}",
                    Some("emoji") => "\u{fe0f}",
                    _ => "",
                };
                format!("{emoji}{variant}")
            }
            None => caps[1].to_owned(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_known_codes_only() {
        assert_eq!(emoji_replace("a :smiley: b"), "a 😃 b");
        assert_eq!(emoji_replace("12:34:56"), "12:34:56");
        assert_eq!(emoji_replace("x:cat:y"), "x🐱y");
        assert_eq!(emoji_replace(":cat-text:"), "🐱\u{fe0e}");
        assert_eq!(emoji_replace("::"), "::");
    }
}
