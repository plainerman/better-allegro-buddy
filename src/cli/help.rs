//! rich-click 1.9.9 help output and error rendering. Owned by P2-B.
//!
//! A port of `RichHelpConfiguration` (theme resolution incl. the `RICH_CLICK_THEME` env
//! override, names or a JSON object), `RichHelpFormatter`/`create_console` (its own
//! recording console: colour system detected on stdout, `force_terminal` from
//! `FORCE_COLOR`/`PY_COLORS`/`GITHUB_ACTIONS`, width `min(max_width=100, size)`,
//! `TERMINAL_WIDTH`), and `rich_help_rendering` (usage, help text, option and command
//! panels, error panels, `Aborted.`) as configured by Python's `CLICK_CONTEXT_SETTINGS`
//! (`theme='modern'`, `commands_table_column_types=['name_with_aliases', 'help']`,
//! `max_width=100`).
//!
//! Not ported: `text_markup='markdown'|'rich'` and emoji (only reachable through a JSON
//! `RICH_CLICK_THEME`), panel definitions (`option_groups`/`command_groups`), headers,
//! epilogs and footers (allegro-buddy defines none), the `random` theme and the
//! `RICH_CLICK_THEME= failed to parse` warning for a broken JSON value. The
//! `UserWarning` for an unknown theme is ported ([`apply_theme_warning`]).

use super::commands::{Command, Const, Param, ParamKind, ParamType, ShowDefault};
use super::parser;
use super::rich::{AlignLeft, Columns, Group, Node, Padded, Panel};
use super::themes::{COLORS, FORMATS, PostCombine, TV, ThemeDef};
use crate::error::UsageError;
use crate::render::boxes::BoxStyle;
use crate::render::console::{ColorChoice, ConsoleConfig, PrintOptions, Stream};
use crate::render::highlight::LazyRegex;
use crate::render::style::Style;
use crate::render::table::{Cell, Column, Table};
use crate::render::text::{Overflow, Text};
use crate::render::{Console, Renderable, Theme};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::LazyLock;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// A configuration value (Python object in `RichHelpConfiguration`).
#[derive(Debug, Clone, PartialEq)]
enum CV {
    None,
    Str(String),
    Bool(bool),
    Int(i64),
    List(Vec<CV>),
}

impl From<TV> for CV {
    fn from(v: TV) -> CV {
        match v {
            TV::None => CV::None,
            TV::Str(s) => CV::Str(s.to_owned()),
            TV::Bool(b) => CV::Bool(b),
            TV::Int(i) => CV::Int(i),
            TV::Tuple(t) => CV::List(t.iter().map(|i| CV::Int(*i)).collect()),
            TV::List(l) => CV::List(l.iter().map(|s| CV::Str((*s).to_owned())).collect()),
        }
    }
}

impl CV {
    fn from_json(value: &serde_json::Value) -> CV {
        match value {
            serde_json::Value::Null => CV::None,
            serde_json::Value::Bool(b) => CV::Bool(*b),
            serde_json::Value::Number(n) => CV::Int(n.as_i64().unwrap_or(0)),
            serde_json::Value::String(s) => CV::Str(s.clone()),
            serde_json::Value::Array(a) => CV::List(a.iter().map(CV::from_json).collect()),
            serde_json::Value::Object(_) => CV::None,
        }
    }
}

type Styles = Vec<(&'static str, CV)>;

fn find(table: &'static [ThemeDef], name: &str) -> Option<&'static ThemeDef> {
    table.iter().find(|t| t.name == name)
}

/// `RichClickTheme.combine`: color + format, then both post-combine callbacks.
fn combine(color: &ThemeDef, format: &ThemeDef) -> Styles {
    let mut styles: Styles = Vec::new();
    let set = |styles: &mut Styles, key: &'static str, value: CV| match styles
        .iter_mut()
        .find(|(k, _)| *k == key)
    {
        Some(entry) => entry.1 = value,
        None => styles.push((key, value)),
    };
    for (k, v) in color.styles.iter().chain(format.styles) {
        set(&mut styles, k, CV::from(*v));
    }
    for callback in [color.callback, format.callback].into_iter().flatten() {
        match callback {
            PostCombine::SpacedDelimiters => {
                set(&mut styles, "delimiter_comma", CV::Str(", ".into()));
                set(&mut styles, "delimiter_slash", CV::Str(" / ".into()));
            }
            PostCombine::NotDimTitle => {
                for key in [
                    "style_options_panel_title_style",
                    "style_commands_panel_title_style",
                ] {
                    let update = match styles.iter().find(|(k, _)| *k == key) {
                        Some((_, CV::Str(v))) if !v.split(' ').any(|w| w == "dim") => {
                            Some(format!("{v} not dim").trim().to_owned())
                        }
                        _ => None,
                    };
                    if let Some(update) = update {
                        set(&mut styles, key, CV::Str(update));
                    }
                }
            }
        }
    }
    styles
}

/// The theme `RICH_CLICK_THEME` names (`apply_theme`): the raw value, or the `"theme"` of a
/// JSON object; looked up with `raise_key_error=False`.
fn env_theme() -> Option<String> {
    let env = std::env::var("RICH_CLICK_THEME").ok()?;
    if !env.trim().starts_with('{') {
        return Some(env);
    }
    match serde_json::from_str(env.trim()) {
        Ok(serde_json::Value::Object(data)) => match data.get("theme") {
            Some(serde_json::Value::String(theme)) => Some(theme.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Whether `get_theme(name)` finds the theme (else it raises `RichClickThemeNotFound`).
fn theme_exists(name: &str) -> bool {
    if name == "random" {
        return true;
    }
    if !name.contains('-') {
        return find(COLORS, name).is_some() || find(FORMATS, name).is_some();
    }
    let parts: Vec<&str> = name.split('-').collect();
    parts.len() == 2 && find(COLORS, parts[0]).is_some() && find(FORMATS, parts[1]).is_some()
}

/// Set when Python's `warnings` filters change, which resets the registry that
/// suppresses repeats of a warning: initially, on every subprocess started by name
/// (`os.get_exec_path` uses `warnings.catch_warnings()`) and on the lazy import of
/// `rich_click._compat_click` (error path of rich-click's `main`).
static WARNINGS_RESET: AtomicBool = AtomicBool::new(true);

/// See [`WARNINGS_RESET`]: core::slurm calls this for every command it starts.
pub fn python_warning_filters_mutated() {
    WARNINGS_RESET.store(true, Ordering::SeqCst);
}

/// rich-click's `apply_theme()` (a `RichHelpConfiguration` or help formatter is created)
/// with an unknown `RICH_CLICK_THEME`: `get_theme` warns
/// `RichClickTheme '<name>' not found.` through `warnings.warn(..., stacklevel=2)`, shown
/// by `warnings.showwarning` unless the registry still holds it. bab prints the same two
/// lines with the module path relative to site-packages (Python's starts with the venv
/// path; the compat harness normalises that prefix).
pub fn apply_theme_warning() {
    static UNKNOWN: OnceLock<Option<String>> = OnceLock::new();
    let unknown = UNKNOWN.get_or_init(|| env_theme().filter(|name| !theme_exists(name)));
    if let Some(name) = unknown
        && WARNINGS_RESET.swap(false, Ordering::SeqCst)
    {
        super::write_stream(
            &mut std::io::stderr(),
            &format!(
                "rich_click/rich_help_configuration.py:437: UserWarning: RichClickTheme \
                     '{name}' not found.\n  theme_styles = get_theme(theme, \
                     raise_key_error=raise_key_error).styles\n"
            ),
        );
    }
}

/// `get_theme(theme, raise_key_error=False)` (an unknown name falls back like Python).
fn get_theme(name: &str) -> Styles {
    let (default, boxed) = (
        find(COLORS, "default").unwrap(),
        find(FORMATS, "box").unwrap(),
    );
    if !name.contains('-') {
        if let Some(color) = find(COLORS, name) {
            return combine(color, boxed);
        }
        if let Some(format) = find(FORMATS, name) {
            return combine(default, format);
        }
        return combine(default, boxed);
    }
    let mut parts = name.split('-');
    let (clr, fmt) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    let extra = parts.next().is_some();
    match (find(COLORS, clr), find(FORMATS, fmt)) {
        (Some(color), Some(format)) if !extra => combine(color, format),
        (Some(color), _) => combine(color, boxed),
        (None, Some(format)) => combine(default, format),
        (None, None) => combine(default, boxed),
    }
}

static DEFAULT_BOX: LazyLock<Styles> = LazyLock::new(|| get_theme("default-box"));

/// Dataclass defaults of the non-theme fields rich-click reads.
fn dataclass_default(key: &str) -> CV {
    let s = |v: &str| CV::Str(v.to_owned());
    let pad = |v: &[i64]| CV::List(v.iter().map(|i| CV::Int(*i)).collect());
    match key {
        "align_options_panel" | "align_commands_panel" | "align_errors_panel" => s("left"),
        "style_options_table_padding" | "style_commands_table_padding" => pad(&[0, 1]),
        "style_options_table_pad_edge" | "style_commands_table_pad_edge" => CV::Bool(false),
        "style_options_table_show_lines" | "style_commands_table_show_lines" => CV::Bool(false),
        "style_options_table_leading" | "style_commands_table_leading" => CV::Int(0),
        "style_aborted" => s("red"),
        "helptext_aliases_string" => s("Aliases: {}"),
        "arguments_panel_title" => s("Arguments"),
        "options_panel_title" => s("Options"),
        "commands_panel_title" => s("Commands"),
        "errors_panel_title" => s("Error"),
        "aborted_text" => s("Aborted."),
        "padding_helptext_deprecated" | "padding_helptext_first_line" => CV::Int(0),
        "padding_errors_panel" => pad(&[0, 0, 1, 0]),
        "padding_errors_suggestion" => pad(&[0, 1, 0, 1]),
        "padding_errors_epilogue" => pad(&[0, 1, 1, 1]),
        "helptext_show_aliases" => CV::Bool(true),
        "use_click_short_help" => CV::Bool(false),
        "group_arguments_options" => CV::Bool(false),
        "commands_before_options" => CV::Bool(false),
        "enable_theme_env_var" => CV::Bool(true),
        "color_system" => s("auto"),
        _ => CV::None,
    }
}

/// rich-click's truthy() for the force-terminal env vars.
fn truthy(value: &str) -> Option<bool> {
    match value.to_lowercase().as_str() {
        "y" | "yes" | "t" | "true" | "1" => Some(true),
        "n" | "no" | "f" | "false" | "0" => Some(false),
        _ => None,
    }
}

/// The resolved `RichHelpConfiguration`.
pub struct Config {
    values: HashMap<String, CV>,
}

impl Config {
    /// `RichHelpConfiguration(theme='modern', commands_table_column_types=[...],
    /// max_width=100)` after `apply_theme(force_default=True)`.
    pub fn load() -> Config {
        let mut values: HashMap<String, CV> = HashMap::new();
        values.insert(
            "commands_table_column_types".into(),
            CV::List(vec![
                CV::Str("name_with_aliases".into()),
                CV::Str("help".into()),
            ]),
        );
        values.insert("max_width".into(), CV::Int(100));
        let width = std::env::var("TERMINAL_WIDTH")
            .ok()
            .filter(|w| !w.is_empty())
            .and_then(|w| w.trim().parse::<i64>().ok());
        values.insert("width".into(), width.map_or(CV::None, CV::Int));
        let force = ["FORCE_COLOR", "PY_COLORS", "GITHUB_ACTIONS"]
            .iter()
            .find_map(|var| std::env::var(var).ok())
            .and_then(|v| truthy(&v));
        values.insert("force_terminal".into(), force.map_or(CV::None, CV::Bool));

        if let Ok(env) = std::env::var("RICH_CLICK_THEME")
            && env.trim().starts_with('{')
            && let Ok(serde_json::Value::Object(data)) = serde_json::from_str(env.trim())
        {
            for (key, value) in data.iter().filter(|(k, _)| *k != "theme") {
                if !Self::known(key) {
                    break;
                }
                values.insert(key.clone(), CV::from_json(value));
            }
        }
        let theme = env_theme().unwrap_or_else(|| "modern".to_owned());
        for (key, value) in get_theme(&theme) {
            values.entry(key.to_owned()).or_insert(value);
        }
        Config { values }
    }

    fn known(key: &str) -> bool {
        DEFAULT_BOX.iter().any(|(k, _)| *k == key)
            || dataclass_default(key) != CV::None
            || matches!(
                key,
                "width"
                    | "max_width"
                    | "force_terminal"
                    | "style_helptext_aliases"
                    | "style_errors_suggestion"
                    | "style_errors_suggestion_command"
                    | "style_options_table_row_styles"
                    | "style_commands_table_row_styles"
                    | "style_commands_table_column_width_ratio"
                    | "header_text"
                    | "footer_text"
                    | "errors_suggestion"
                    | "errors_epilogue"
                    | "show_arguments"
                    | "show_metavars_column"
                    | "append_metavars_help"
                    | "option_envvar_first"
                    | "text_markup"
                    | "text_kwargs"
                    | "text_paragraph_linebreaks"
                    | "text_emojis"
                    | "use_markdown"
                    | "use_markdown_emoji"
                    | "use_rich_markup"
                    | "command_groups"
                    | "option_groups"
                    | "highlighter_patterns"
                    | "legacy_windows"
                    | "theme"
            )
    }

    fn get(&self, key: &str) -> CV {
        if let Some(v) = self.values.get(key) {
            return v.clone();
        }
        if let Some((_, v)) = DEFAULT_BOX.iter().find(|(k, _)| *k == key) {
            return v.clone();
        }
        dataclass_default(key)
    }

    fn str(&self, key: &str) -> Option<String> {
        match self.get(key) {
            CV::Str(s) => Some(s),
            _ => None,
        }
    }

    fn text(&self, key: &str) -> String {
        self.str(key).unwrap_or_default()
    }

    fn flag(&self, key: &str) -> bool {
        match self.get(key) {
            CV::Bool(b) => b,
            CV::Int(i) => i != 0,
            CV::Str(s) => !s.is_empty(),
            CV::List(l) => !l.is_empty(),
            CV::None => false,
        }
    }

    fn int(&self, key: &str) -> Option<i64> {
        match self.get(key) {
            CV::Int(i) => Some(i),
            _ => None,
        }
    }

    fn list(&self, key: &str) -> Vec<String> {
        match self.get(key) {
            CV::List(l) => l
                .into_iter()
                .filter_map(|v| match v {
                    CV::Str(s) => Some(s),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// `Padding.unpack(value)`.
    fn padding(&self, key: &str) -> (usize, usize, usize, usize) {
        let ints: Vec<usize> = match self.get(key) {
            CV::Int(i) => vec![i.max(0) as usize],
            CV::List(l) => l
                .iter()
                .map(|v| match v {
                    CV::Int(i) => (*i).max(0) as usize,
                    _ => 0,
                })
                .collect(),
            _ => vec![0],
        };
        match ints[..] {
            [p] => (p, p, p, p),
            [v, h] => (v, h, v, h),
            [t, r, b, l] => (t, r, b, l),
            _ => (0, 0, 0, 0),
        }
    }
}

/// The rich-click formatter: its config, console and highlighters.
pub struct Formatter {
    config: Config,
    console: Console,
    theme: Theme,
}

static OPTION_HIGHLIGHTS: [LazyRegex; 4] = [
    LazyRegex::new(r"(^|[^\w\-])(?P<switch>-([^\W0-9][\w\-]*\w|[^\W0-9]))"),
    LazyRegex::new(r"(^|[^\w\-])(?P<option>--([^\W0-9][\w\-]*\w|[^\W0-9]))"),
    LazyRegex::new(r"(?P<metavar><[^>]+>)"),
    LazyRegex::new(r"(?P<deprecated>\(DEPRECATED(?:\: .*?)?\))$"),
];

static USAGE_HIGHLIGHT: LazyRegex = LazyRegex::new(r"(?P<argument>\w+)");

static METAVAR_HIGHLIGHTS: [LazyRegex; 3] = [
    LazyRegex::new(r"(^|\s)(?P<metavar_sep>(\[|<))"),
    LazyRegex::new(r"(?P<metavar_sep>\|)"),
    LazyRegex::new(r"(?P<metavar_sep>(\]|>)$)"),
];

/// `inspect.cleandoc`.
fn cleandoc(doc: &str) -> String {
    let expanded = expand_tabs(doc);
    let mut lines: Vec<String> = expanded.split('\n').map(str::to_owned).collect();
    let margin = lines
        .iter()
        .skip(1)
        .filter_map(|line| {
            let content = line.trim_start().chars().count();
            (content > 0).then(|| line.chars().count() - content)
        })
        .min();
    if let Some(first) = lines.first_mut() {
        *first = first.trim_start().to_owned();
    }
    if let Some(margin) = margin {
        for line in lines.iter_mut().skip(1) {
            *line = line.chars().skip(margin).collect();
        }
    }
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    while lines.first().is_some_and(|l| l.is_empty()) {
        lines.remove(0);
    }
    lines.join("\n")
}

/// `str.expandtabs()` (tab size 8).
fn expand_tabs(s: &str) -> String {
    let mut out = String::new();
    let mut column = 0;
    for ch in s.chars() {
        match ch {
            '\t' => {
                let n = 8 - column % 8;
                out.push_str(&" ".repeat(n));
                column += n;
            }
            '\n' | '\r' => {
                out.push(ch);
                column = 0;
            }
            _ => {
                out.push(ch);
                column += 1;
            }
        }
    }
    out
}

/// Python `str.splitlines()`.
fn splitlines(s: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((i, ch)) = chars.next() {
        let is_break = matches!(
            ch,
            '\n' | '\r'
                | '\x0b'
                | '\x0c'
                | '\x1c'
                | '\x1d'
                | '\x1e'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if is_break {
            lines.push(&s[start..i]);
            let mut next = i + ch.len_utf8();
            if ch == '\r'
                && let Some(&(j, '\n')) = chars.peek()
            {
                chars.next();
                next = j + 1;
            }
            start = next;
        }
    }
    if start < s.len() {
        lines.push(&s[start..]);
    }
    lines
}

impl Formatter {
    pub fn new() -> Formatter {
        apply_theme_warning();
        let config = Config::load();
        let style = |key: &str| {
            let definition = config.text(key);
            Style::parse(&definition).unwrap_or(Style::NULL)
        };
        let theme = Theme::with_styles([
            ("option", style("style_option")),
            ("command", style("style_command")),
            ("argument", style("style_argument")),
            ("switch", style("style_switch")),
            ("metavar", style("style_metavar")),
            ("metavar_sep", style("style_metavar_separator")),
            ("usage", style("style_usage")),
            ("deprecated", style("style_deprecated")),
        ]);
        let force_terminal = match config.get("force_terminal") {
            CV::Bool(b) => Some(b),
            _ => None,
        };
        let console_config = |width: Option<usize>| ConsoleConfig {
            stream: None,
            detect_stream: Some(Stream::Stdout),
            force_terminal,
            color_system: ColorChoice::Auto,
            width,
            theme: theme.clone(),
            ..ConsoleConfig::default()
        };
        let width = config.int("width").map(|w| w.max(0) as usize);
        let mut console = Console::new(console_config(width));
        if let Some(max_width) = config.int("max_width") {
            let width = (max_width.max(0) as usize).min(console.size().0);
            console = Console::new(console_config(Some(width)));
        }
        Formatter {
            config,
            console,
            theme,
        }
    }

    fn style(&self, key: &str) -> Style {
        match self.config.str(key) {
            Some(definition) => self.theme.get_style_or_null(&definition),
            None => Style::NULL,
        }
    }

    fn opt_style(&self, key: &str) -> Option<Style> {
        self.config
            .str(key)
            .map(|d| self.theme.get_style_or_null(&d))
    }

    fn write(&self, renderable: &dyn Renderable, style: Option<Style>) {
        self.console.print_with(
            renderable,
            PrintOptions {
                style,
                ..PrintOptions::default()
            },
        );
    }

    fn getvalue(&self) -> String {
        self.console.take_output()
    }

    fn highlight(&self, mut text: Text, patterns: &[LazyRegex]) -> Text {
        for re in patterns {
            let re = re.for_haystack(text.plain());
            text.highlight_regex(re, "", &self.theme);
        }
        text
    }

    /// `Text.from_markup(markup, style=style)`.
    fn markup(&self, markup: &str, style: Style) -> Text {
        crate::render::markup::render_with(markup, style, &self.theme, true)
            .unwrap_or_else(|_| Text::with_style(markup, style))
    }

    /// `formatter.rich_text(text, style)` for `text_markup='ansi'`.
    fn rich_text(&self, text: &str, style: Style) -> Text {
        let text = cleandoc(text);
        let lines: Vec<Text> = splitlines(&text)
            .into_iter()
            .map(|line| Text::new(line.rsplit('\r').next().unwrap_or(line)))
            .collect();
        let joiner = Text::with_style("\n", style);
        let joined = joiner.join(lines);
        self.highlight(joined, &OPTION_HIGHLIGHTS)
    }

    fn panel_box(&self, key: &str) -> BoxStyle {
        let name = self.config.str(key).unwrap_or_else(|| "SIMPLE".to_owned());
        rich_click_box(&name)
    }

    /// `get_rich_usage(formatter, prog, args)`.
    fn usage(&self, prog: &str, args: &str) -> Padded {
        let mut args_text = Text::with_style(args, self.style("style_usage_separator"));
        let re = USAGE_HIGHLIGHT.for_haystack(args_text.plain());
        args_text.highlight_regex(re, "", &self.theme);
        let columns = Columns(vec![
            Rc::new(Text::with_style("Usage:", self.style("style_usage"))),
            Rc::new(Text::with_style(prog, self.style("style_usage_command"))),
            Rc::new(args_text),
        ]);
        Padded::new(
            Rc::new(columns),
            self.config.padding("padding_usage"),
            self.style("style_padding_usage"),
        )
    }
}

impl Default for Formatter {
    fn default() -> Self {
        Formatter::new()
    }
}

/// `rich_box.get_box(name)`: rich-click's own boxes, else `rich.box.<name>`.
fn rich_click_box(name: &str) -> BoxStyle {
    let definition = match name {
        "HORIZONTALS_TOP" => Some(" ── \n    \n    \n    \n    \n    \n    \n    \n"),
        "HORIZONTALS_DOUBLE_TOP" => Some(" ══ \n    \n    \n    \n    \n    \n    \n    \n"),
        "BLANK" => Some("    \n    \n    \n    \n    \n    \n    \n    \n"),
        _ => None,
    };
    match definition {
        Some(definition) => BoxStyle::parse(definition, false).expect("rich-click box"),
        None => BoxStyle::named(name).unwrap_or_else(|| BoxStyle::named("ROUNDED").unwrap()),
    }
}

/// `collect_usage_pieces`: `[OPTIONS]`, argument metavars, `COMMAND [ARGS]...`.
fn usage_pieces(command: &'static Command) -> String {
    let mut pieces = vec!["[OPTIONS]".to_owned()];
    pieces.extend(
        command
            .params
            .iter()
            .filter(|p| p.kind == ParamKind::Argument)
            .map(parser::make_metavar),
    );
    if command.is_group {
        pieces.push("COMMAND [ARGS]...".to_owned());
    }
    pieces.join(" ")
}

/// Python `str(default)` for a parameter default.
fn default_display(
    param: &Param,
    default_map: Option<&[(&'static str, String)]>,
) -> Option<String> {
    if let Some((_, raw)) = default_map.and_then(|m| m.iter().find(|(d, _)| *d == param.dest)) {
        return Some(raw.clone());
    }
    match param.default {
        Const::None => None,
        Const::Bool(true) => Some("True".to_owned()),
        Const::Bool(false) => Some("False".to_owned()),
        Const::Int(i) => Some(i.to_string()),
        Const::Float(f) => Some(parser::py_float_repr(f)),
        Const::Str(s) => Some(s.to_owned()),
    }
}

impl Formatter {
    /// The `help` column of an option row (`get_help_parameter`).
    fn parameter_help(
        &self,
        param: &Param,
        default_map: Option<&[(&'static str, String)]>,
    ) -> Node {
        let mut sections: Vec<Text> = Vec::new();
        for section in self.config.list("options_table_help_sections") {
            match section.as_str() {
                "help" => {
                    if let Some(help) = param.help.filter(|h| !h.is_empty()) {
                        let paragraphs: Vec<String> = help
                            .split("\n\n")
                            .map(|p| {
                                if p.starts_with('\u{8}') {
                                    format!("{}\n", p.trim_matches(['\u{8}', '\n']))
                                } else {
                                    p.replace('\n', " ").trim().to_owned()
                                }
                            })
                            .collect();
                        let text = paragraphs.join("\n");
                        sections.push(self.rich_text(text.trim(), self.style("style_option_help")));
                    }
                }
                "required" if param.required => sections.push(self.markup(
                    &self.config.text("required_long_string"),
                    self.style("style_required_long"),
                )),
                "default" if param.kind == ParamKind::Option => {
                    let shown = match param.show_default {
                        ShowDefault::Text(text) => Some(format!("({text})")),
                        ShowDefault::Yes => match (param.ty, default_display(param, default_map)) {
                            (ParamType::Flag, _) => None,
                            (_, Some(value)) if value.is_empty() => Some("\"\"".to_owned()),
                            (_, value) => value,
                        },
                        ShowDefault::No => None,
                    };
                    if let Some(shown) = shown.filter(|s| !s.is_empty()) {
                        let markup = self
                            .config
                            .text("default_string")
                            .replace("{}", &shown.replace('[', "\\["));
                        sections.push(self.markup(&markup, self.style("style_option_default")));
                    }
                }
                "range" => {
                    if let ParamType::FloatRange { min } = param.ty {
                        let markup = self
                            .config
                            .text("append_range_help_string")
                            .replace("{}", &format!("x>={min}"));
                        let style = self
                            .opt_style("style_range_append")
                            .unwrap_or_else(|| self.style("style_metavar_append"));
                        sections.push(self.markup(&markup, style));
                    }
                }
                "metavar" | "metavar_short" => {
                    if let Some(metavar) = self.metavar_append(param, section == "metavar", true) {
                        sections.push(metavar);
                    }
                }
                _ => {}
            }
        }
        let mut sep = Text::with_style(" ", self.style("style_option_help"));
        sep.overflow = Some(Overflow::Fold);
        Rc::new(Columns(vec![Rc::new(sep.join(sections))]))
    }

    /// `_get_parameter_metavar`.
    fn metavar_append(&self, param: &Param, show_range: bool, append: bool) -> Option<Text> {
        let mut metavar = parser::make_metavar(param);
        if param.kind == ParamKind::Argument {
            metavar = parser::type_name(param.ty).to_uppercase();
        }
        if param.kind == ParamKind::Option && (metavar == "BOOLEAN" || parser::is_flag(param)) {
            return None;
        }
        let mut metavar = metavar.replace(['[', ']'], "");
        if show_range && let ParamType::FloatRange { min } = param.ty {
            metavar.push_str(&format!(" x>={min}"));
        }
        let markup = self
            .config
            .text("append_metavars_help_string")
            .replace("{}", &metavar);
        let style = if append {
            self.style("style_metavar_append")
        } else {
            self.style("style_metavar")
        };
        let mut text = self.markup(&markup, style);
        text.overflow = Some(Overflow::Fold);
        Some(text)
    }

    /// `_get_parameter_help_metavar_col`.
    fn metavar_column(&self, param: &Param, show_range: bool) -> Option<Text> {
        let mut plain = String::new();
        let mut metavar = parser::make_metavar(param);
        if param.kind == ParamKind::Argument {
            metavar = parser::type_name(param.ty).to_uppercase();
        }
        if param.kind == ParamKind::Argument || (metavar != "BOOLEAN" && !parser::is_flag(param)) {
            plain.push_str(&metavar);
        }
        if let ParamType::FloatRange { min } = param.ty
            && show_range
        {
            plain.push(' ');
            plain.push_str(
                &self
                    .config
                    .text("range_string")
                    .replace("{}", &format!("x>={min}")),
            );
        }
        if plain.is_empty() {
            return None;
        }
        let mut text = Text::with_style(plain, self.style("style_metavar"));
        text.overflow = Some(Overflow::Fold);
        Some(self.highlight(text, &METAVAR_HIGHLIGHTS))
    }

    /// `_get_parameter_help_opt`: (primary, secondary, long, short, all).
    fn opt_names(&self, param: &Param) -> [Option<Text>; 5] {
        let comma = Text::with_style(
            self.config.text("delimiter_comma"),
            self.style("style_option_help"),
        );
        let (mut short, mut long): (Vec<&str>, Vec<&str>) = (Vec::new(), Vec::new());
        for decl in param.decls {
            if param.kind == ParamKind::Argument || decl.contains("--") {
                long.push(decl);
            } else {
                short.push(decl);
            }
        }
        let mut primary: Vec<Text> = Vec::new();
        let mut short_cols: Vec<Text> = Vec::new();
        let mut long_cols: Vec<Text> = Vec::new();
        for o in &short {
            let t = Text::with_style(o.trim(), self.style("style_switch"));
            primary.extend([t.clone(), comma.clone()]);
            short_cols.extend([t, comma.clone()]);
        }
        for o in &long {
            let name = if param.kind == ParamKind::Argument {
                o.to_uppercase()
            } else {
                o.trim().to_owned()
            };
            let t = Text::with_style(name, self.style("style_option"));
            primary.extend([t.clone(), comma.clone()]);
            long_cols.extend([t, comma.clone()]);
        }
        let renderable = |mut cols: Vec<Text>| -> Option<Text> {
            cols.pop();
            match cols.len() {
                0 => None,
                1 => cols.pop(),
                _ => {
                    let mut sep = Text::new("");
                    sep.overflow = Some(Overflow::Fold);
                    Some(sep.join(cols))
                }
            }
        };
        [
            renderable(primary.clone()),
            None,
            renderable(long_cols),
            renderable(short_cols),
            renderable(primary),
        ]
    }

    /// `get_parameter_rich_table_row`.
    fn option_row(
        &self,
        param: &Param,
        default_map: Option<&[(&'static str, String)]>,
    ) -> Vec<Option<Cell>> {
        let [primary, secondary, long, short, all] = self.opt_names(param);
        let column_types = self.config.list("options_table_column_types");
        let metavar_padded = if column_types
            .iter()
            .any(|c| c == "opt_all_metavar" || c == "opt_long_metavar")
        {
            self.metavar_append(param, false, false)
        } else {
            None
        };
        let join_metavar = |opt: &Option<Text>| -> Option<Text> {
            match (opt, &metavar_padded) {
                (_, None) => opt.clone(),
                (None, Some(m)) => Some(m.clone()),
                (Some(o), Some(m)) => Some(
                    Text::with_style(" ", self.style("style_option_help"))
                        .join([o.clone(), m.clone()]),
                ),
            }
        };
        column_types
            .iter()
            .map(|column| match column.as_str() {
                "required" => param
                    .required
                    .then(|| {
                        Text::with_style(
                            self.config.text("required_short_string"),
                            self.style("style_required_short"),
                        )
                    })
                    .map(Cell::Text),
                "opt_long" => long.clone().map(Cell::Text),
                "opt_short" => short.clone().map(Cell::Text),
                "opt_primary" => primary.clone().map(Cell::Text),
                "opt_secondary" => secondary.clone().map(Cell::Text),
                "opt_all" => all.clone().map(Cell::Text),
                "metavar" => self.metavar_column(param, true).map(Cell::Text),
                "metavar_short" => self.metavar_column(param, false).map(Cell::Text),
                "opt_all_metavar" => join_metavar(&all).map(Cell::Text),
                "opt_long_metavar" => join_metavar(&long).map(Cell::Text),
                "help" => Some(Cell::Renderable(self.parameter_help(param, default_map))),
                _ => None,
            })
            .collect()
    }

    /// `_get_command_help`.
    fn command_help(&self, command: &Command) -> Text {
        let help = command.short_help.or(command.help).unwrap_or("");
        let cleaned = cleandoc(help);
        let first = cleaned.split("\n\n").next().unwrap_or("");
        let first = if first.starts_with('\u{8}') {
            first.replace("\u{8}\n", "")
        } else {
            first.replace('\n', " ")
        };
        self.rich_text(first.trim(), self.style("style_command_help"))
    }

    fn command_row(&self, command: &'static Command) -> Vec<Option<Cell>> {
        let name = Text::with_style(command.name, self.style("style_command"));
        let comma = Text::with_style(
            self.config.text("delimiter_comma"),
            self.style("style_command_help"),
        );
        let mut aliases: Vec<Text> = Vec::new();
        for (i, alias) in command.aliases.iter().enumerate() {
            if i > 0 {
                aliases.push(comma.clone());
            }
            aliases.push(Text::with_style(
                *alias,
                self.style("style_command_aliases"),
            ));
        }
        self.config
            .list("commands_table_column_types")
            .iter()
            .map(|column| match column.as_str() {
                "name" => Some(Cell::Text(name.clone())),
                "aliases" => {
                    (!aliases.is_empty()).then(|| Cell::Text(Text::new("").join(aliases.clone())))
                }
                "name_with_aliases" => Some(Cell::Text(if aliases.is_empty() {
                    name.clone()
                } else {
                    let mut sep = Text::new("");
                    sep.overflow = Some(Overflow::Ellipsis);
                    let mut parts = vec![name.clone(), comma.clone()];
                    parts.extend(aliases.clone());
                    sep.join(parts)
                })),
                "help" => Some(Cell::Text(self.command_help(command))),
                _ => None,
            })
            .collect()
    }

    /// The table part shared by option and command panels.
    fn panel_table(&self, kind: &str, columns: Vec<Column>, rows: Vec<Vec<Option<Cell>>>) -> Table {
        let key = |suffix: &str| format!("style_{kind}_table_{suffix}");
        let mut table = Table::new();
        table.highlight = kind == "options";
        table.show_header = false;
        table.show_lines = self.config.flag(&key("show_lines"));
        table.leading = self.config.int(&key("leading")).unwrap_or(0).max(0) as usize;
        table.box_style = self.config.str(&key("box")).map(|b| rich_click_box(&b));
        table.border_style = self.style(&key("border_style"));
        table.row_styles = match self.config.get(&key("row_styles")) {
            CV::List(l) => l
                .iter()
                .filter_map(|v| match v {
                    CV::Str(s) => Some(self.theme.get_style_or_null(s)),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        table.pad_edge = self.config.flag(&key("pad_edge"));
        table.padding = self.config.padding(&key("padding"));
        table.expand = self.config.flag(&key("expand"));
        for column in columns {
            table.add_column_spec(column);
        }
        let width = rows.first().map_or(0, Vec::len);
        let keep: Vec<bool> = (0..width)
            .map(|i| {
                rows.iter().any(|row| match &row[i] {
                    Some(Cell::Text(t)) => !t.is_empty(),
                    Some(Cell::Str(s)) => !s.is_empty(),
                    Some(Cell::Renderable(_)) => true,
                    None => false,
                })
            })
            .collect();
        for row in rows {
            let cells: Vec<Cell> = row
                .into_iter()
                .zip(&keep)
                .filter(|(_, keep)| **keep)
                .map(|(cell, _)| cell.unwrap_or_else(|| Cell::Str(String::new())))
                .collect();
            table.add_row(cells, false);
        }
        table
    }

    fn panel(&self, kind: &str, name_key: &str, table: Table) -> Panel {
        let key = |suffix: &str| format!("style_{kind}_panel_{suffix}");
        let title = self
            .config
            .text("panel_title_string")
            .replace("{}", &self.config.text(name_key));
        let title = Text::with_style(title, self.style(&key("title_style")));
        let box_key = key("box");
        let (box_style, blank_top) = match self.config.get(&box_key) {
            CV::None => (rich_click_box("SIMPLE"), false),
            CV::Str(name) if name == "BLANK" => (rich_click_box("BLANK"), true),
            _ => (self.panel_box(&box_key), false),
        };
        let title_padding = self.config.int("panel_title_padding").unwrap_or(0).max(0) as usize;
        Panel {
            renderable: Rc::new(table),
            box_style,
            title: Some(Panel::rich_click_title(title, title_padding)),
            title_align: self
                .config
                .str(&format!("align_{kind}_panel"))
                .unwrap_or_else(|| "left".to_owned()),
            expand: true,
            style: self.style(&key("style")),
            border_style: self.style(&key("border")),
            padding: self.config.padding(&key("padding")),
            blank_top,
        }
    }

    /// `RichCommand.format_help` into this formatter.
    fn format_help(
        &self,
        path: &[String],
        command: &'static Command,
        default_map: Option<&[(&'static str, String)]>,
    ) {
        self.write(&self.usage(&path.join(" "), &usage_pieces(command)), None);
        self.write_help_text(command);
        let options: Vec<&Param> = parser::params_of(command)
            .filter(|p| p.kind == ParamKind::Option && !p.hidden)
            .collect();
        let option_rows: Vec<Vec<Option<Cell>>> = options
            .iter()
            .map(|p| self.option_row(p, default_map))
            .collect();
        let options_table = self.panel_table("options", Vec::new(), option_rows);
        if !options_table.rows.is_empty() {
            self.write(
                &self.panel("options", "options_panel_title", options_table),
                None,
            );
        }
        if command.is_group {
            let columns = vec![
                Column {
                    style: self.style("style_command"),
                    no_wrap: true,
                    ..Column::default()
                },
                Column {
                    no_wrap: false,
                    ..Column::default()
                },
            ];
            let rows: Vec<Vec<Option<Cell>>> = parser::list_commands(command)
                .into_iter()
                .filter(|c| !c.hidden)
                .map(|c| self.command_row(c))
                .collect();
            let table = self.panel_table("commands", columns, rows);
            if !table.rows.is_empty() {
                self.write(&self.panel("commands", "commands_panel_title", table), None);
            }
        }
    }

    /// `get_rich_help_text`.
    fn write_help_text(&self, command: &'static Command) {
        if command.help.is_none() && command.aliases.is_empty() {
            return;
        }
        let help = cleandoc(command.help.unwrap_or(""));
        let help = help.split('\u{c}').next().unwrap_or("").to_owned();
        let mut paragraphs = help.split("\n\n");
        let mut first = paragraphs.next().unwrap_or("").to_owned();
        if !first.starts_with('\u{8}') {
            first = first.replace('\n', " ");
        }
        let mut items: Vec<Node> = vec![Rc::new(Padded::new(
            Rc::new(self.rich_text(first.trim(), self.style("style_helptext_first_line"))),
            self.config.padding("padding_helptext_first_line"),
            self.style("style_padding_helptext"),
        ))];
        let remaining: Vec<&str> = paragraphs.collect();
        if !remaining.is_empty() {
            let mut buf = String::new();
            for para in remaining {
                if para.starts_with('\u{8}') {
                    buf.push_str(para.trim_matches(['\u{8}', '\n']));
                    buf.push('\n');
                    buf.push('\n');
                    continue;
                }
                let (mut continuation, mut list, mut first_line) = (false, false, true);
                for p in para.split('\n') {
                    if p.starts_with("* ") || p.starts_with("- ") {
                        continuation = false;
                        if !first_line {
                            buf.push('\n');
                        }
                        list = true;
                        buf.push_str(p);
                    } else if list && p.starts_with("  ") {
                        buf.push(' ');
                        buf.push_str(p.trim_start());
                    } else if p.starts_with("    ") || p.starts_with("> ") {
                        continuation = false;
                        list = false;
                        if !first_line {
                            buf.push('\n');
                        }
                        buf.push_str(p);
                    } else if continuation {
                        buf.push(' ');
                        buf.push_str(p);
                    } else {
                        list = false;
                        buf.push_str(p);
                        continuation = true;
                    }
                    first_line = false;
                }
                buf.push('\n');
            }
            items.push(Rc::new(self.rich_text(&buf, self.style("style_helptext"))));
        }
        if !command.aliases.is_empty() && self.config.flag("helptext_show_aliases") {
            let style = self
                .opt_style("style_helptext_aliases")
                .unwrap_or_else(|| self.style("style_helptext"));
            let markup = self
                .config
                .text("helptext_aliases_string")
                .replace("{}", &command.aliases.join(", "));
            items.push(Rc::new(self.markup(&markup, style)));
        }
        let block = Padded::new(
            Rc::new(AlignLeft(Rc::new(Group(items)))),
            self.config.padding("padding_helptext"),
            self.style("style_padding_helptext"),
        );
        self.write(&block, None);
    }

    /// `rich_format_error` (usage line and hint only when there is a context).
    fn format_error(&self, message: &str, usage: Option<&UsageError>) {
        if let Some(usage) = usage {
            let command = parser::command_at(&usage.command_path);
            self.write(
                &self.usage(&usage.command_path.join(" "), &usage_pieces(command)),
                None,
            );
            if usage.show_help_hint {
                let command_style = self
                    .opt_style("style_errors_suggestion_command")
                    .unwrap_or_else(|| self.style("style_option"));
                let hint = Text::new(" ").join([
                    Text::new("Try"),
                    Text::with_style(
                        format!("'{} -h'", usage.command_path.join(" ")),
                        command_style,
                    ),
                    Text::new("for help"),
                ]);
                let style = self
                    .opt_style("style_errors_suggestion")
                    .unwrap_or_else(|| self.style("style_helptext"));
                let padded = Padded::new(
                    Rc::new(hint),
                    self.config.padding("padding_errors_suggestion"),
                    self.style("style_padding_errors"),
                );
                self.write(&padded, Some(style));
            }
        }
        let body = self.highlight(Text::new(message), &OPTION_HIGHLIGHTS);
        let box_name = self
            .config
            .str("style_errors_panel_box")
            .unwrap_or_else(|| "ROUNDED".to_owned());
        let panel = Panel {
            renderable: Rc::new(body),
            box_style: rich_click_box(if box_name.is_empty() {
                "ROUNDED"
            } else {
                &box_name
            }),
            title: Some(Panel::rich_title(
                self.markup(&self.config.text("errors_panel_title"), Style::NULL),
            )),
            title_align: self.config.text("align_errors_panel"),
            expand: true,
            style: Style::NULL,
            border_style: self.style("style_errors_panel_border"),
            padding: (0, 1, 0, 1),
            blank_top: box_name == "BLANK",
        };
        let padded = Padded::new(
            Rc::new(panel),
            self.config.padding("padding_errors_panel"),
            self.style("style_padding_errors"),
        );
        self.write(&padded, None);
    }
}

/// The help page of `path` as rich-click renders it (`ctx.get_help()` + newline).
pub fn render_help(path: &[String], default_map: Option<&[(&'static str, String)]>) -> String {
    let formatter = Formatter::new();
    formatter.format_help(path, parser::command_at(path), default_map);
    let mut out = formatter.getvalue().trim_end_matches('\n').to_owned();
    out.push('\n');
    out
}

/// Print the help page of the command at `path` (`["bab", "dev"]`) to stdout;
/// `default_map` is the `[dev]` config map of `bab dev start` (shown as defaults).
pub fn print_help(path: &[String], default_map: Option<&[(&'static str, String)]>) {
    super::write_stream(&mut std::io::stdout(), &render_help(path, default_map));
}

/// What rich-click prints for a usage error (`formatter.write_error(e)`).
pub fn render_usage_error(error: &UsageError) -> String {
    let formatter = Formatter::new();
    let context = (!error.command_path.is_empty()).then_some(error);
    formatter.format_error(&error.message, context);
    formatter.getvalue()
}

/// Print a usage error the rich-click way to stderr.
pub fn print_usage_error(error: &UsageError) {
    super::write_stream(&mut std::io::stderr(), &render_usage_error(error));
}

/// Print a `ClickException` message the rich-click way to stderr.
pub fn print_click_exception(message: &str) {
    let formatter = Formatter::new();
    formatter.format_error(message, None);
    super::write_stream(&mut std::io::stderr(), &formatter.getvalue());
}

/// Print rich-click's `Aborted.` (style `red`) to stderr.
pub fn print_aborted() {
    let formatter = Formatter::new();
    let text = formatter.config.text("aborted_text");
    let style = formatter.style("style_aborted");
    formatter.console.print_str_with(
        &text,
        PrintOptions {
            style: Some(style),
            ..PrintOptions::default()
        },
    );
    super::write_stream(&mut std::io::stderr(), &formatter.getvalue());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_twins_agree() {
        use crate::render::highlight::{TWIN_SAMPLES, assert_ascii_twin_agrees};
        let all = OPTION_HIGHLIGHTS.iter().chain(&METAVAR_HIGHLIGHTS);
        for re in all.chain([&USAGE_HIGHLIGHT]) {
            assert_ascii_twin_agrees(re, TWIN_SAMPLES);
        }
    }

    #[test]
    fn unknown_themes_like_rich_click_get_theme() {
        // recorded with rich-click 1.9.9 `get_theme(name)` (KeyError = unknown)
        for (name, exists) in [
            ("bogus", false),
            ("nord", true),
            ("nord-box", true),
            ("nord-bogus", false),
            ("random", true),
            ("x-y-z", false),
            ("-", false),
            ("default", true),
            ("box", true),
            ("modern", true),
            ("nord-modern-x", false),
            ("star-slim", true),
            ("Nord", false),
        ] {
            assert_eq!(theme_exists(name), exists, "{name}");
        }
    }
}
