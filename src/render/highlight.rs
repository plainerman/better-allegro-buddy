//! Rich `RegexHighlighter` / `ReprHighlighter` (rich/highlighter.py).
//!
//! `Console.print("some str")` highlights by default (Console(highlight=True)): numbers,
//! quoted strings, paths, URLs, UUIDs, ipv4/ipv6/eui48 ... get `repr.*` styles from Rich's
//! default theme. Table cells are not highlighted (Table(highlight=False)). The patterns
//! below are Rich 15's, verbatim except for escaping `[`/`]` inside character classes
//! (Rust regex syntax); `fancy_regex` provides the lookbehinds.

use super::text::Text;
use super::theme::Theme;
use fancy_regex::Regex;
use std::borrow::Cow;
use std::sync::{LazyLock, OnceLock};

/// A regex compiled on first use. Unicode `\w`/`\W`/`\d` classes (Python `re` semantics)
/// take milliseconds to compile, yet on ASCII text they match exactly their ASCII parts:
/// ASCII haystacks (nearly everything bab prints) use an ASCII twin of the pattern
/// ([`ascii_twin`]); the Unicode original is compiled only for non-ASCII text.
#[derive(Debug)]
pub struct LazyRegex {
    source: Cow<'static, str>,
    unicode: OnceLock<Regex>,
    ascii: OnceLock<Regex>,
}

impl LazyRegex {
    pub const fn new(source: &'static str) -> LazyRegex {
        LazyRegex {
            source: Cow::Borrowed(source),
            unicode: OnceLock::new(),
            ascii: OnceLock::new(),
        }
    }

    fn owned(source: &str) -> LazyRegex {
        LazyRegex {
            source: Cow::Owned(source.to_owned()),
            unicode: OnceLock::new(),
            ascii: OnceLock::new(),
        }
    }

    /// The compiled pattern to search `haystack` with.
    pub fn for_haystack(&self, haystack: &str) -> &Regex {
        let compile = |source: &str| Regex::new(source).expect("highlight pattern");
        if haystack.is_ascii() {
            self.ascii
                .get_or_init(|| compile(&ascii_twin(&self.source)))
        } else {
            self.unicode.get_or_init(|| compile(&self.source))
        }
    }
}

/// `pattern` with `\w`, `\W` and `\d` replaced by their ASCII equivalents (inside and
/// outside character classes): the same matches and groups on ASCII haystacks. Assumes
/// what all patterns here follow: classes don't nest and `[`/`]` inside them are escaped.
fn ascii_twin(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len() * 2);
    let mut in_class = false;
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                let Some(escaped) = chars.next() else {
                    out.push(c);
                    break;
                };
                out.push_str(match (escaped, in_class) {
                    ('w', false) => "[0-9A-Za-z_]",
                    ('w', true) => "0-9A-Za-z_",
                    ('W', false) => "[^0-9A-Za-z_]",
                    ('W', true) => r"\x00-/:-@\[-^`{-\x{10FFFF}",
                    ('d', false) => "[0-9]",
                    ('d', true) => "0-9",
                    _ => {
                        out.extend([c, escaped]);
                        continue;
                    }
                });
            }
            '[' if !in_class => {
                in_class = true;
                out.push(c);
            }
            ']' if in_class => {
                in_class = false;
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

/// One highlight pattern (Rich's patterns are expensive to compile and most strings bab
/// prints need only some of them).
#[derive(Debug)]
struct Pattern {
    regex: LazyRegex,
    /// Cheap test that the pattern cannot match when false.
    may_match: fn(&str) -> bool,
    /// (group name, max code points): emulates `[\w_]{1,50}` without its compile cost.
    trim_group: Option<(&'static str, usize)>,
}

fn always(_: &str) -> bool {
    true
}

/// Rich `RegexHighlighter`: each regex adds `base_style + group name` spans, in order.
#[derive(Debug)]
pub struct RegexHighlighter {
    patterns: Vec<Pattern>,
    pub base_style: String,
}

impl RegexHighlighter {
    /// Compile `patterns` (Python `re` syntax that `fancy_regex` understands).
    pub fn new(
        base_style: &str,
        patterns: &[&str],
    ) -> Result<RegexHighlighter, fancy_regex::Error> {
        let mut out = Vec::with_capacity(patterns.len());
        for p in patterns {
            let regex = LazyRegex::owned(p);
            let _ = regex.unicode.set(Regex::new(p)?);
            out.push(Pattern {
                regex,
                may_match: always,
                trim_group: None,
            });
        }
        Ok(RegexHighlighter {
            patterns: out,
            base_style: base_style.to_owned(),
        })
    }

    /// Rich `RegexHighlighter.highlight(text)` (in place).
    pub fn highlight(&self, text: &mut Text, theme: &Theme) {
        for pattern in &self.patterns {
            if !(pattern.may_match)(text.plain()) {
                continue;
            }
            let re = pattern.regex.for_haystack(text.plain());
            let before = text.spans().len();
            text.highlight_regex(re, &self.base_style, theme);
            if let Some((group, max)) = pattern.trim_group {
                let style = theme.get_style_or_null(&format!("{}{group}", self.base_style));
                for span in &mut text.spans_mut()[before..] {
                    if span.style == style && span.end - span.start > max {
                        span.start = span.end - max;
                    }
                }
            }
        }
    }
}

fn has_any(text: &str, chars: &[char]) -> bool {
    text.contains(chars)
}

static REPR: LazyLock<RegexHighlighter> = LazyLock::new(|| {
    let combined = [
        r"(?P<ipv4>[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})",
        r"(?P<ipv6>([A-Fa-f0-9]{1,4}::?){1,7}[A-Fa-f0-9]{1,4})",
        r"(?P<eui64>(?:[0-9A-Fa-f]{1,2}-){7}[0-9A-Fa-f]{1,2}|(?:[0-9A-Fa-f]{1,2}:){7}[0-9A-Fa-f]{1,2}|(?:[0-9A-Fa-f]{4}\.){3}[0-9A-Fa-f]{4})",
        r"(?P<eui48>(?:[0-9A-Fa-f]{1,2}-){5}[0-9A-Fa-f]{1,2}|(?:[0-9A-Fa-f]{1,2}:){5}[0-9A-Fa-f]{1,2}|(?:[0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4})",
        r"(?P<uuid>[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})",
        r"(?P<call>[\w.]*?)\(",
        r"\b(?P<bool_true>True)\b|\b(?P<bool_false>False)\b|\b(?P<none>None)\b",
        r"(?P<ellipsis>\.\.\.)",
        r"(?P<number_complex>(?<!\w)(?:\-?[0-9]+\.?[0-9]*(?:e[-+]?\d+?)?)(?:[-+](?:[0-9]+\.?[0-9]*(?:e[-+]?\d+)?))?j)",
        r"(?P<number>(?<!\w)\-?[0-9]+\.?[0-9]*(e[-+]?\d+?)?\b|0x[0-9a-fA-F]*)",
        r"(?P<path>\B(/[-\w._+]+)*\/)(?P<filename>[-\w._+]*)?",
        r#"(?<![\\\w])(?P<str>b?'''.*?(?<!\\)'''|b?'.*?(?<!\\)'|b?""".*?(?<!\\)"""|b?".*?(?<!\\)")"#,
        r"(?P<url>(file|https|http|ws|wss)://[-0-9a-zA-Z$_+!`(),.?/;:&=%#~@]*)",
    ]
    .join("|");
    let pattern = |source: &str, may_match: fn(&str) -> bool, trim_group| Pattern {
        regex: LazyRegex::owned(source),
        may_match,
        trim_group,
    };
    RegexHighlighter {
        base_style: "repr.".to_owned(),
        patterns: vec![
            pattern(
                r"(?P<tag_start><)(?P<tag_name>[-\w.:|]*)(?P<tag_contents>[\w\W]*)(?P<tag_end>>)",
                |t| t.contains('<') && t.contains('>'),
                None,
            ),
            // Rich: `(?P<attrib_name>[\w_]{1,50})=...`; a longer run of word characters
            // before `=` matches only its last 50 (the span is trimmed accordingly).
            pattern(
                r#"(?P<attrib_name>[\w_]+)=(?P<attrib_value>"?[\w_]+"?)?"#,
                |t| t.contains('='),
                Some(("attrib_name", 50)),
            ),
            pattern(
                r"(?P<brace>[\]\[{}()])",
                |t| has_any(t, &['[', ']', '{', '}', '(', ')']),
                None,
            ),
            pattern(&combined, combined_may_match, None),
        ],
    }
});

/// False only when no alternative of Rich's combined repr regex can match.
fn combined_may_match(t: &str) -> bool {
    t.bytes()
        .any(|b| b.is_ascii_digit() || matches!(b, b'(' | b':' | b'/' | b'\'' | b'"'))
        || t.matches('-').count() >= 4
        || t.matches('.').count() >= 2
        || t.contains("True")
        || t.contains("False")
        || t.contains("None")
}

/// Rich `ReprHighlighter`.
pub fn repr_highlighter() -> &'static RegexHighlighter {
    &REPR
}

/// Apply `ReprHighlighter` to `text` in place (Rich `Highlighter.__call__` -> `highlight`).
pub fn highlight_repr(text: &mut Text) {
    REPR.highlight(text, &Theme::default());
}

/// Test helper: `re`'s ASCII twin finds the same matches and groups as the original.
#[cfg(test)]
pub(crate) fn assert_ascii_twin_agrees(re: &LazyRegex, samples: &[&str]) {
    let groups = |pattern: &str, sample: &str| -> Vec<Vec<Option<(usize, usize)>>> {
        let re = Regex::new(pattern).unwrap();
        re.captures_iter(sample)
            .map(|caps| {
                let caps = caps.unwrap();
                (0..caps.len())
                    .map(|i| caps.get(i).map(|m| (m.start(), m.end())))
                    .collect()
            })
            .collect()
    };
    let twin = ascii_twin(&re.source);
    for sample in samples {
        assert!(sample.is_ascii());
        assert_eq!(
            groups(&re.source, sample),
            groups(&twin, sample),
            "{twin} on {sample:?}"
        );
    }
}

/// ASCII strings that exercise the `\w`/`\W`/`\d` parts of the patterns.
#[cfg(test)]
pub(crate) const TWIN_SAMPLES: &[&str] = &[
    "Canceled 123, 456_7. -1.5e3 2j 0xff 1-2+3.5e+7j x2 a_1(b) _9 9_ abc1",
    "see '/tmp/x.txt' and \"q\\\"x\" b'z' /usr/lib/x-1_2+3/ a=1 b_c=\"d\" <tag a=1>",
    "--opt -x -1 --x-y- --2 -_a a-b --a_b9 x--y [default: 4] <N> (DEPRECATED: use -y)",
    "True False None foo.bar(1) ... 10.0.0.1 ::1 fe80::1:2 de:ad:be:ef:00:01 http://a.b/c?d=1",
    "12345678-1234-1234-1234-123456789abc 01-23-45-67-89-ab-cd-ef [x|y] <a>",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::style::Style;

    #[test]
    fn ascii_twins_agree() {
        for pattern in &REPR.patterns {
            assert_ascii_twin_agrees(&pattern.regex, TWIN_SAMPLES);
        }
        assert_eq!(
            ascii_twin(r"[^\W0-9]\w\d[\\\w]\b"),
            r"[^\x00-/:-@\[-^`{-\x{10FFFF}0-9][0-9A-Za-z_][0-9][\\0-9A-Za-z_]\b"
        );
    }

    fn styled(s: &str) -> Vec<(String, Style)> {
        let mut t = Text::new(s);
        highlight_repr(&mut t);
        t.spans()
            .iter()
            .map(|sp| {
                (
                    s.chars().skip(sp.start).take(sp.end - sp.start).collect(),
                    sp.style,
                )
            })
            .collect()
    }

    #[test]
    fn numbers_and_clock() {
        let number = Style::of("bold cyan not italic");
        let spans = styled("Canceled 123, 456_7.");
        assert_eq!(spans[0], ("123".to_owned(), number));
        let spans = styled("every 10s · 12:34:56 · Ctrl-C to quit");
        assert!(spans.iter().any(|(s, _)| s == "12:34:56"));
        assert!(!spans.iter().any(|(s, _)| s == "10"));
    }

    #[test]
    fn strings_and_paths() {
        let spans = styled("see '/tmp/x.txt' now");
        assert!(spans.iter().any(|(s, _)| s == "'/tmp/x.txt'"));
        assert_eq!(spans.len(), 1);
    }
}
