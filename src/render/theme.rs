//! Rich `Theme` + `Console.get_style`: named styles (`repr.number`, `table.header`, ...)
//! looked up before parsing a definition.

use super::data::DEFAULT_STYLES;
use super::style::{Style, StyleSyntaxError};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

/// A set of named styles; [`Theme::default`] is Rich's `themes.DEFAULT`.
#[derive(Debug, Clone)]
pub struct Theme {
    styles: Arc<HashMap<String, Style>>,
}

static DEFAULT_THEME: LazyLock<Theme> = LazyLock::new(|| Theme {
    styles: Arc::new(
        DEFAULT_STYLES
            .iter()
            .map(|(name, definition)| ((*name).to_owned(), Style::of(definition)))
            .collect(),
    ),
});

impl Default for Theme {
    fn default() -> Theme {
        DEFAULT_THEME.clone()
    }
}

impl Theme {
    /// `Theme(styles, inherit=True)`: the default styles plus/overridden by `styles`.
    pub fn with_styles<'a>(styles: impl IntoIterator<Item = (&'a str, Style)>) -> Theme {
        let mut map = (*DEFAULT_THEME.styles).clone();
        for (name, style) in styles {
            map.insert(name.to_owned(), style);
        }
        Theme {
            styles: Arc::new(map),
        }
    }

    /// The named style, if the theme defines it.
    pub fn lookup(&self, name: &str) -> Option<Style> {
        self.styles.get(name).copied()
    }

    /// Rich `Console.get_style(name)`: theme lookup, else parse the definition.
    pub fn get_style(&self, name: &str) -> Result<Style, StyleSyntaxError> {
        match self.lookup(name) {
            Some(style) => Ok(style),
            None => Style::parse(name),
        }
    }

    /// `Console.get_style(name, default=Style.null())`, as used when rendering spans.
    pub fn get_style_or_null(&self, name: &str) -> Style {
        self.get_style(name).unwrap_or(Style::NULL)
    }
}

/// Resolve a style name with the default theme (invalid -> null, like Rich at render time).
pub fn style(name: &str) -> Style {
    DEFAULT_THEME.get_style_or_null(name)
}
