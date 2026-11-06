//! click-compatible argument parser (click 8.3.2 semantics). Owned by P2-B.
//!
//! A port of click's `_OptionParser` (option tokenizing: `-p x`, `-px`, `--partition x`,
//! `--partition=x`, clustered short flags, `--`, optional-value options like `--watch`),
//! `Command.parse_args`/`Parameter.handle_parse_result` (sources: command line >
//! `default_map` > default; type conversion, required arguments, callbacks, eager
//! `--help`/`--version` in invocation order) and `Group.invoke`/`resolve_command`
//! (aliases, unknown commands, `invoke_without_command`, `no_args_is_help`).
//!
//! Order of events (click): the main group parses its own args; eager `--version`/`--help`
//! end parsing; then the subcommand is resolved, then the main group callback runs
//! (`load_default_map`, which may fail with `AbError::Click`), THEN the subcommand's args
//! are parsed (so `bab info --help` with a broken config shows the config error).
//!
//! The same engine runs in click's `resilient_parsing` mode for shell completion
//! ([`resolve_context`]): errors are swallowed, callbacks of groups and eager options are
//! skipped.

use super::commands::{Callback, Command, Const, MAIN, Param, ParamKind, ParamType, ShowDefault};
use crate::error::{AbError, Result, UsageError};
use crate::pyfmt::repr_str;
use std::collections::VecDeque;

/// A parsed parameter value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Str(String),
    Int(i64),
    Float(f64),
}

/// Values for one invoked command, by `dest`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Args {
    pub values: Vec<(&'static str, Value)>,
}

impl Args {
    pub fn get(&self, dest: &str) -> &Value {
        self.values
            .iter()
            .find(|(d, _)| *d == dest)
            .map(|(_, v)| v)
            .unwrap_or(&Value::None)
    }

    /// Flag value (`false` when absent).
    pub fn flag(&self, dest: &str) -> bool {
        matches!(self.get(dest), Value::Bool(true))
    }

    pub fn opt_str(&self, dest: &str) -> Option<&str> {
        match self.get(dest) {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn str(&self, dest: &str) -> &str {
        self.opt_str(dest)
            .unwrap_or_else(|| panic!("{dest}: no string value"))
    }

    pub fn opt_int(&self, dest: &str) -> Option<i64> {
        match self.get(dest) {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn int(&self, dest: &str) -> i64 {
        self.opt_int(dest)
            .unwrap_or_else(|| panic!("{dest}: no int value"))
    }

    pub fn opt_float(&self, dest: &str) -> Option<f64> {
        match self.get(dest) {
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn float(&self, dest: &str) -> f64 {
        self.opt_float(dest)
            .unwrap_or_else(|| panic!("{dest}: no float value"))
    }
}

/// The outcome of parsing argv.
#[derive(Debug, Clone, PartialEq)]
pub enum Parsed {
    /// Run `command` (a leaf, or `info` for the bare main group) with `args`; `path` is
    /// the command path for messages (`["bab", "dev", "start"]`).
    Run {
        path: Vec<String>,
        command: &'static Command,
        args: Args,
    },
    /// Print the help page of `path` to stdout and exit with `code` (0 for `-h`, 2 for a
    /// group invoked without subcommand: click's `NoArgsIsHelpError`). `default_map` is
    /// the `[dev]` map when the page is `bab dev start` (shown as defaults).
    Help {
        path: Vec<String>,
        default_map: Option<DefaultMap>,
        code: i32,
    },
    /// `--version`: print `bab, version 2.7`, exit 0.
    Version,
}

/// `[dev]` default map entries: click param dest -> raw string from config.
pub type DefaultMap = Vec<(&'static str, String)>;

/// click's `ParameterSource` (the values bab needs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    CommandLine,
    DefaultMap,
    Default,
}

/// The `-h/--help` option click adds to every command (`help_option_names`).
pub static HELP: Param = Param {
    kind: ParamKind::Option,
    decls: &["-h", "--help"],
    dest: "help",
    ty: ParamType::Flag,
    flag_value: None,
    default: Const::Bool(false),
    show_default: ShowDefault::No,
    metavar: None,
    help: Some("Show this message and exit."),
    required: false,
    completer: None,
    callback: None,
    hidden: false,
};

/// `command.get_params(ctx)`: declared params plus the help option.
pub fn params_of(command: &'static Command) -> impl Iterator<Item = &'static Param> {
    command.params.iter().chain(std::iter::once(&HELP))
}

pub fn is_eager(param: &Param) -> bool {
    param.dest == "help" || param.ty == ParamType::Version
}

/// `is_flag` (store_const; never consumes a value).
pub fn is_flag(param: &Param) -> bool {
    matches!(param.ty, ParamType::Flag | ParamType::Version)
}

fn exposed(param: &Param) -> bool {
    !is_eager(param)
}

/// click's `_split_opt`: (prefix, rest).
pub fn split_opt(opt: &str) -> (&str, &str) {
    let mut chars = opt.chars();
    let Some(first) = chars.next() else {
        return ("", "");
    };
    if first.is_alphanumeric() {
        return ("", opt);
    }
    let n = first.len_utf8();
    if chars.next() == Some(first) {
        let m = n + first.len_utf8();
        return (&opt[..m], &opt[m..]);
    }
    (&opt[..n], &opt[n..])
}

/// Short (`-p`) vs long (`--partition`, `-foo`) declaration (click `_Option`).
fn is_short_decl(decl: &str) -> bool {
    let (prefix, rest) = split_opt(decl);
    prefix.chars().count() == 1 && rest.chars().count() == 1
}

/// Find a subcommand by name or alias (rich-click `_alias_mapping` + `commands.get`).
pub fn get_command(group: &'static Command, name: &str) -> Option<&'static Command> {
    let canonical = group
        .subcommands
        .iter()
        .find(|c| c.aliases.contains(&name))
        .map_or(name, |c| c.name);
    group.subcommands.iter().find(|c| c.name == canonical)
}

/// `list_commands`: sorted by name, `l` moved right after `dev` (`_Main.list_commands`).
pub fn list_commands(group: &'static Command) -> Vec<&'static Command> {
    let mut names: Vec<&'static Command> = group.subcommands.iter().collect();
    names.sort_by(|a, b| a.name.cmp(b.name));
    if let (Some(l), Some(_)) = (
        names.iter().position(|c| c.name == "l"),
        names.iter().position(|c| c.name == "dev"),
    ) {
        let cmd = names.remove(l);
        let at = names.iter().position(|c| c.name == "dev").unwrap() + 1;
        names.insert(at, cmd);
    }
    names
}

/// The command at `path` (`["bab", "dev", "start"]`, aliases allowed).
pub fn command_at(path: &[String]) -> &'static Command {
    let mut command = &MAIN;
    for name in path.iter().skip(1) {
        command = get_command(command, name).unwrap_or(command);
    }
    command
}

/// A parsed context (click `Context`).
#[derive(Debug, Clone)]
pub struct Ctx {
    pub command: &'static Command,
    /// Info names from the root (`["bab", "q"]`).
    pub path: Vec<String>,
    /// `ctx.params` by dest; `None` = UNSET.
    pub values: Vec<(&'static str, Option<Value>)>,
    pub sources: Vec<(&'static str, Source)>,
    pub protected_args: Vec<String>,
    pub args: Vec<String>,
    /// `ctx.default_map` of this context (only `dev start` has one).
    pub default_map: Option<DefaultMap>,
    /// The full `[dev]` map handed down from the main group callback.
    carried_map: Option<DefaultMap>,
    pub resilient: bool,
}

impl Ctx {
    fn new(
        command: &'static Command,
        path: Vec<String>,
        default_map: Option<DefaultMap>,
        carried_map: Option<DefaultMap>,
        resilient: bool,
    ) -> Ctx {
        Ctx {
            command,
            path,
            values: Vec::new(),
            sources: Vec::new(),
            protected_args: Vec::new(),
            args: Vec::new(),
            default_map,
            carried_map,
            resilient,
        }
    }

    pub fn source(&self, dest: &str) -> Option<Source> {
        self.sources
            .iter()
            .find(|(d, _)| *d == dest)
            .map(|(_, s)| *s)
    }

    pub fn value(&self, dest: &str) -> Option<&Value> {
        self.values
            .iter()
            .find(|(d, _)| *d == dest)
            .and_then(|(_, v)| v.as_ref())
    }

    fn set_source(&mut self, dest: &'static str, source: Source) {
        match self.sources.iter_mut().find(|(d, _)| *d == dest) {
            Some(entry) => entry.1 = source,
            None => self.sources.push((dest, source)),
        }
    }

    fn lookup_default(&self, dest: &str) -> Option<&str> {
        self.default_map
            .as_ref()?
            .iter()
            .find(|(d, _)| *d == dest)
            .map(|(_, v)| v.as_str())
    }

    fn usage_error(&self, message: impl Into<String>) -> AbError {
        AbError::Usage(UsageError {
            message: message.into(),
            command_path: self.path.clone(),
            show_help_hint: true,
        })
    }

    fn into_args(self) -> Args {
        Args {
            values: self
                .values
                .into_iter()
                .map(|(d, v)| (d, v.unwrap_or(Value::None)))
                .collect(),
        }
    }
}

/// Why parsing stopped early.
#[derive(Debug)]
enum Stop {
    Help(Box<Ctx>, i32),
    Version,
    Error(AbError),
}

impl From<AbError> for Stop {
    fn from(err: AbError) -> Stop {
        Stop::Error(err)
    }
}

/// Raw option values produced by the tokenizer.
#[derive(Debug, Clone, PartialEq)]
enum Raw {
    Str(String),
    /// click `FLAG_NEEDS_VALUE`: an optional-value option given without a value.
    NeedsValue,
    /// store_const (flags): `True`.
    Const,
}

enum TokError {
    NoSuchOption {
        name: String,
        possibilities: Vec<String>,
    },
    Other(String),
}

impl TokError {
    fn message(self) -> String {
        match self {
            TokError::Other(message) => message,
            TokError::NoSuchOption {
                name,
                mut possibilities,
            } => {
                let message = format!("No such option: {name}");
                if possibilities.is_empty() {
                    return message;
                }
                possibilities.sort();
                let joined = possibilities.join(", ");
                if possibilities.len() == 1 {
                    format!("{message} Did you mean {joined}?")
                } else {
                    format!("{message} (Possible options: {joined})")
                }
            }
        }
    }
}

struct TokState {
    opts: Vec<(&'static str, Raw)>,
    largs: Vec<String>,
    rargs: VecDeque<String>,
    order: Vec<&'static str>,
}

impl TokState {
    fn store(&mut self, dest: &'static str, value: Raw) {
        match self.opts.iter_mut().find(|(d, _)| *d == dest) {
            Some(entry) => entry.1 = value,
            None => self.opts.push((dest, value)),
        }
        self.order.push(dest);
    }
}

/// click's `_OptionParser` for one command.
struct OptionParser {
    short: Vec<(String, &'static Param)>,
    long: Vec<(String, &'static Param)>,
    arguments: Vec<&'static Param>,
    interspersed: bool,
}

impl OptionParser {
    fn new(command: &'static Command) -> OptionParser {
        let mut parser = OptionParser {
            short: Vec::new(),
            long: Vec::new(),
            arguments: Vec::new(),
            interspersed: !command.is_group,
        };
        for param in params_of(command) {
            match param.kind {
                ParamKind::Argument => parser.arguments.push(param),
                ParamKind::Option => {
                    for decl in param.decls {
                        let table = if is_short_decl(decl) {
                            &mut parser.short
                        } else {
                            &mut parser.long
                        };
                        match table.iter_mut().find(|(d, _)| d == decl) {
                            Some(entry) => entry.1 = param,
                            None => table.push(((*decl).to_owned(), param)),
                        }
                    }
                }
            }
        }
        parser
    }

    /// `parse_args`: (opts, leftover args, invocation order). In resilient mode a usage
    /// error stops tokenizing and returns what was collected so far.
    fn parse_args(
        &self,
        args: &[String],
        resilient: bool,
    ) -> std::result::Result<TokState, TokError> {
        let mut state = TokState {
            opts: Vec::new(),
            largs: Vec::new(),
            rargs: args.iter().cloned().collect(),
            order: Vec::new(),
        };
        if let Err(err) = self.process_args_for_options(&mut state) {
            if resilient {
                return Ok(state);
            }
            return Err(err);
        }
        self.process_args_for_args(&mut state);
        Ok(state)
    }

    fn process_args_for_args(&self, state: &mut TokState) {
        let mut rest: VecDeque<String> =
            state.largs.drain(..).chain(state.rargs.drain(..)).collect();
        for param in &self.arguments {
            match rest.pop_front() {
                Some(value) => state.store(param.dest, Raw::Str(value)),
                None => state.order.push(param.dest),
            }
        }
        state.largs = rest.into_iter().collect();
    }

    fn process_args_for_options(&self, state: &mut TokState) -> std::result::Result<(), TokError> {
        while let Some(arg) = state.rargs.pop_front() {
            if arg == "--" {
                return Ok(());
            } else if arg.starts_with('-') && arg.chars().count() > 1 {
                self.process_opts(&arg, state)?;
            } else if self.interspersed {
                state.largs.push(arg);
            } else {
                state.rargs.push_front(arg);
                return Ok(());
            }
        }
        Ok(())
    }

    fn process_opts(&self, arg: &str, state: &mut TokState) -> std::result::Result<(), TokError> {
        let (long_opt, explicit) = match arg.split_once('=') {
            Some((opt, value)) => (opt, Some(value)),
            None => (arg, None),
        };
        match self.match_long_opt(long_opt, explicit, state) {
            Err(TokError::NoSuchOption { .. }) if !arg.starts_with("--") => {
                self.match_short_opt(arg, state)
            }
            other => other,
        }
    }

    fn match_long_opt(
        &self,
        opt: &str,
        explicit: Option<&str>,
        state: &mut TokState,
    ) -> std::result::Result<(), TokError> {
        let Some((_, param)) = self.long.iter().find(|(d, _)| d == opt) else {
            let possibilities = get_close_matches(opt, self.long.iter().map(|(d, _)| d.as_str()));
            return Err(TokError::NoSuchOption {
                name: opt.to_owned(),
                possibilities,
            });
        };
        let value = if !is_flag(param) {
            if let Some(value) = explicit {
                state.rargs.push_front(value.to_owned());
            }
            self.value_from_state(opt, param, state)?
        } else if explicit.is_some() {
            return Err(TokError::Other(format!(
                "Option {} does not take a value.",
                repr_str(opt)
            )));
        } else {
            Raw::Const
        };
        state.store(param.dest, value);
        Ok(())
    }

    fn match_short_opt(
        &self,
        arg: &str,
        state: &mut TokState,
    ) -> std::result::Result<(), TokError> {
        let chars: Vec<char> = arg.chars().collect();
        let prefix = chars[0];
        let mut i = 1;
        for &ch in &chars[1..] {
            let opt = format!("{prefix}{ch}");
            i += 1;
            let Some((_, param)) = self.short.iter().find(|(d, _)| *d == opt) else {
                return Err(TokError::NoSuchOption {
                    name: opt,
                    possibilities: Vec::new(),
                });
            };
            let mut stop = false;
            let value = if !is_flag(param) {
                if i < chars.len() {
                    state.rargs.push_front(chars[i..].iter().collect());
                    stop = true;
                }
                self.value_from_state(&opt, param, state)?
            } else {
                Raw::Const
            };
            state.store(param.dest, value);
            if stop {
                break;
            }
        }
        Ok(())
    }

    fn value_from_state(
        &self,
        name: &str,
        param: &Param,
        state: &mut TokState,
    ) -> std::result::Result<Raw, TokError> {
        let needs_value = param.flag_value.is_some();
        match state.rargs.front() {
            None if needs_value => Ok(Raw::NeedsValue),
            None => Err(TokError::Other(format!(
                "Option {} requires an argument.",
                repr_str(name)
            ))),
            Some(next) if needs_value && next.starts_with('-') && next.chars().count() > 1 => {
                Ok(Raw::NeedsValue)
            }
            Some(_) => Ok(Raw::Str(state.rargs.pop_front().unwrap())),
        }
    }
}

/// A value on its way through `process_value`.
#[derive(Debug, Clone, PartialEq)]
enum Pending {
    Unset,
    Raw(String),
    Const(Const),
}

/// `param.get_error_hint(ctx)`.
pub fn error_hint(param: &Param) -> String {
    match param.kind {
        ParamKind::Option => param
            .decls
            .iter()
            .map(|d| format!("'{d}'"))
            .collect::<Vec<_>>()
            .join(" / "),
        ParamKind::Argument => format!("'{}'", make_metavar(param)),
    }
}

/// The type's own metavar (`ParamType.get_metavar`): only Choice has one.
fn type_metavar(param: &Param) -> Option<String> {
    match param.ty {
        ParamType::Choice(choices) => {
            let joined = choices.join("|");
            if param.required && param.kind == ParamKind::Argument {
                Some(format!("{{{joined}}}"))
            } else {
                Some(format!("[{joined}]"))
            }
        }
        _ => None,
    }
}

/// click's type name (`ParamType.name`).
pub fn type_name(ty: ParamType) -> &'static str {
    match ty {
        ParamType::Flag | ParamType::Version => "boolean",
        ParamType::Str => "text",
        ParamType::Int => "integer",
        ParamType::FloatRange { .. } => "float range",
        ParamType::Choice(_) => "choice",
    }
}

/// `param.make_metavar(ctx)`.
pub fn make_metavar(param: &Param) -> String {
    if let Some(metavar) = param.metavar {
        return metavar.to_owned();
    }
    match param.kind {
        ParamKind::Option => {
            type_metavar(param).unwrap_or_else(|| type_name(param.ty).to_uppercase())
        }
        ParamKind::Argument => {
            let var = type_metavar(param)
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| param.dest.to_uppercase());
            if param.required {
                var
            } else {
                format!("[{var}]")
            }
        }
    }
}

/// `{x!r}` of a Python float (`str(float)`).
pub fn py_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .to_owned();
    }
    let sci = format!("{value:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        let point = exp + 1;
        let body = if point <= 0 {
            format!("0.{}{digits}", "0".repeat((-point) as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{body}")
    } else {
        let mantissa = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let exp_sign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exp_sign}{:02}", exp.abs())
    }
}

/// `[0-9](_?[0-9])*`
fn valid_digits(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('_')
        && !s.ends_with('_')
        && !s.contains("__")
        && s.bytes().all(|b| b.is_ascii_digit() || b == b'_')
}

/// Python `int(s)` for a str (base 10).
fn py_int(s: &str) -> Option<i64> {
    let t = s.trim();
    let (negative, digits) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if !valid_digits(digits) {
        return None;
    }
    let clean: String = digits.chars().filter(|c| *c != '_').collect();
    let magnitude = clean.parse::<i128>().unwrap_or(i128::MAX);
    let value = if negative { -magnitude } else { magnitude };
    Some(value.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
}

/// Python `float(s)` for a str.
fn py_float(s: &str) -> Option<f64> {
    let t = s.trim();
    let (sign, body) = match t.as_bytes().first() {
        Some(b'-') => (-1.0, &t[1..]),
        Some(b'+') => (1.0, &t[1..]),
        _ => (1.0, t),
    };
    let lower = body.to_ascii_lowercase();
    if lower == "inf" || lower == "infinity" {
        return Some(sign * f64::INFINITY);
    }
    if lower == "nan" {
        return Some(f64::NAN);
    }
    let (number, exponent) = match body.find(['e', 'E']) {
        Some(at) => (&body[..at], Some(&body[at + 1..])),
        None => (body, None),
    };
    let (int_part, frac_part) = match number.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (number, None),
    };
    let ok_number = match frac_part {
        None => valid_digits(int_part),
        Some(frac) => {
            (int_part.is_empty() || valid_digits(int_part))
                && (frac.is_empty() || valid_digits(frac))
                && !(int_part.is_empty() && frac.is_empty())
        }
    };
    let ok_exp = exponent.is_none_or(|e| {
        let e = e.strip_prefix(['+', '-']).unwrap_or(e);
        valid_digits(e)
    });
    if !ok_number || !ok_exp {
        return None;
    }
    let clean: String = body.chars().filter(|c| *c != '_').collect();
    clean.parse::<f64>().ok().map(|v| sign * v)
}

const BOOL_STATES: [(&str, bool); 13] = [
    ("1", true),
    ("0", false),
    ("yes", true),
    ("no", false),
    ("true", true),
    ("false", false),
    ("on", true),
    ("off", false),
    ("t", true),
    ("f", false),
    ("y", true),
    ("n", false),
    ("", false),
];

/// Convert/validate one raw value for `param` (click `ParamType.convert`), giving the
/// click `BadParameter` message on failure.
pub fn convert(param: &Param, raw: &str) -> std::result::Result<Value, String> {
    match param.ty {
        ParamType::Str => Ok(Value::Str(raw.to_owned())),
        ParamType::Flag | ParamType::Version => {
            let key = raw.trim().to_lowercase();
            BOOL_STATES
                .iter()
                .find(|(state, _)| *state == key)
                .map(|(_, v)| Value::Bool(*v))
                .ok_or_else(|| {
                    let mut states: Vec<&str> = BOOL_STATES.iter().map(|(s, _)| *s).collect();
                    states.sort_unstable();
                    format!(
                        "{} is not a valid boolean. Recognized values: {}",
                        repr_str(raw),
                        states.join(", ")
                    )
                })
        }
        ParamType::Int => py_int(raw)
            .map(Value::Int)
            .ok_or_else(|| format!("{} is not a valid integer.", repr_str(raw))),
        ParamType::FloatRange { min } => {
            let value = py_float(raw)
                .ok_or_else(|| format!("{} is not a valid float range.", repr_str(raw)))?;
            check_range(value, min)
        }
        ParamType::Choice(choices) => {
            if choices.contains(&raw) {
                Ok(Value::Str(raw.to_owned()))
            } else {
                Err(invalid_choice(raw, choices))
            }
        }
    }
}

fn check_range(value: f64, min: f64) -> std::result::Result<Value, String> {
    if value < min {
        return Err(format!(
            "{} is not in the range x>={min}.",
            py_float_repr(value)
        ));
    }
    Ok(Value::Float(value))
}

/// `time.sleep` converts seconds to int64 nanoseconds: from 2**63 ns (about 292 years)
/// on it raises `OverflowError`, and `nan` raises `ValueError`.
const PY_SLEEP_LIMIT_NS: f64 = 9_223_372_036_854_775_808.0;

/// bab-only guard for a FloatRange value that becomes a sleep (`--watch`,
/// `--poll-interval`). click accepts `nan`, `inf` and huge values, and Python then crashes
/// in `time.sleep` with a traceback (possibly after `dev start` already submitted a job);
/// bab rejects them with a click-style `BadParameter` before using the value.
pub fn sleep_duration(param: &Param, value: f64) -> Result<std::time::Duration> {
    if value * 1e9 < PY_SLEEP_LIMIT_NS
        && let Ok(duration) = std::time::Duration::try_from_secs_f64(value)
    {
        return Ok(duration);
    }
    let min = match param.ty {
        ParamType::FloatRange { min } => min,
        _ => 0.0,
    };
    Err(AbError::Usage(UsageError {
        message: format!(
            "Invalid value for {}: {} is not in the range {min}<=x<{}.",
            error_hint(param),
            py_float_repr(value),
            py_float_repr(PY_SLEEP_LIMIT_NS / 1e9)
        ),
        command_path: Vec::new(),
        show_help_hint: true,
    }))
}

fn invalid_choice(raw: &str, choices: &[&str]) -> String {
    let listed = choices
        .iter()
        .map(|c| repr_str(c))
        .collect::<Vec<_>>()
        .join(", ");
    if choices.len() == 1 {
        format!("{} is not {listed}.", repr_str(raw))
    } else {
        format!("{} is not one of {listed}.", repr_str(raw))
    }
}

/// `type_cast_value` for a constant (defaults, flag values).
fn convert_const(param: &Param, value: Const) -> std::result::Result<Value, String> {
    match (param.ty, value) {
        (_, Const::None) => Ok(Value::None),
        (ParamType::FloatRange { min }, Const::Float(f)) => check_range(f, min),
        (ParamType::FloatRange { min }, Const::Int(i)) => check_range(i as f64, min),
        (_, Const::Bool(b)) => Ok(Value::Bool(b)),
        (_, Const::Int(i)) => Ok(Value::Int(i)),
        (_, Const::Float(f)) => Ok(Value::Float(f)),
        (_, Const::Str(s)) => convert(param, s),
    }
}

/// `Parameter.consume_value` (+ `Option.consume_value`).
fn consume_value(
    ctx: &Ctx,
    param: &'static Param,
    opts: &[(&'static str, Raw)],
) -> (Pending, Source) {
    if let Some((_, raw)) = opts.iter().find(|(d, _)| *d == param.dest) {
        let value = match raw {
            Raw::Str(s) => Pending::Raw(s.clone()),
            Raw::NeedsValue => Pending::Const(param.flag_value.unwrap_or(Const::None)),
            Raw::Const => Pending::Const(Const::Bool(true)),
        };
        return (value, Source::CommandLine);
    }
    if let Some(value) = ctx.lookup_default(param.dest) {
        return (Pending::Raw(value.to_owned()), Source::DefaultMap);
    }
    let value = match param.default {
        Const::None => Pending::Unset,
        other => Pending::Const(other),
    };
    (value, Source::Default)
}

/// What a processed eager option asks for.
enum Eager {
    No,
    Help,
    Version,
}

/// `process_value`: type cast, required check, callback. `Ok(None)` = UNSET.
fn process_value(
    ctx: &Ctx,
    param: &'static Param,
    value: Pending,
) -> std::result::Result<(Option<Value>, Eager), AbError> {
    let bad = |message: String| {
        ctx.usage_error(format!(
            "Invalid value for {}: {message}",
            error_hint(param)
        ))
    };
    if param.kind == ParamKind::Option && param.ty == ParamType::Flag && value == Pending::Unset {
        return Ok((Some(Value::Bool(false)), Eager::No));
    }
    let value = match value {
        Pending::Unset => None,
        Pending::Raw(raw) => Some(convert(param, &raw).map_err(bad)?),
        Pending::Const(c) => Some(convert_const(param, c).map_err(bad)?),
    };
    if param.required && value.is_none() {
        let kind = match param.kind {
            ParamKind::Argument => "argument",
            ParamKind::Option => "option",
        };
        let mut message = format!("Missing {kind} {}.", error_hint(param));
        if let ParamType::Choice(choices) = param.ty {
            message.push_str(&format!(" Choose from:\n\t{}", choices.join(",\n\t")));
        }
        return Err(ctx.usage_error(message));
    }
    let value = match (param.callback, value) {
        (Some(Callback::NormalizeGpuType), Some(Value::Str(s))) => {
            match crate::core::gpu::try_canonical_gpu_type(Some(&s)) {
                Ok(canonical) => Some(Value::Str(canonical)),
                // click: a callback raising during resilient parsing (completion) -> None
                Err(_) if ctx.resilient => Some(Value::None),
                Err(err) => return Err(AbError::Click(err.0)),
            }
        }
        (Some(Callback::NormalizeGpuType), _) => Some(Value::None),
        (None, value) => value,
    };
    let eager = match (param.dest, param.ty, &value) {
        (_, ParamType::Version, Some(Value::Bool(true))) if !ctx.resilient => Eager::Version,
        ("help", _, Some(Value::Bool(true))) if !ctx.resilient => Eager::Help,
        _ => Eager::No,
    };
    Ok((value, eager))
}

/// `Command.parse_args(ctx, args)`.
fn parse_command_args(ctx: &mut Ctx, args: &[String]) -> std::result::Result<(), Stop> {
    let command = ctx.command;
    if args.is_empty() && command.is_group && !command.invoke_without_command && !ctx.resilient {
        return Err(Stop::Help(Box::new(ctx.clone()), 2));
    }
    let parser = OptionParser::new(command);
    let state = parser
        .parse_args(args, ctx.resilient)
        .map_err(|err| match err {
            // click raises BadOptionUsage without a context: no usage line, no hint.
            TokError::Other(message) => AbError::Usage(UsageError {
                message,
                command_path: Vec::new(),
                show_help_hint: false,
            }),
            no_such => ctx.usage_error(no_such.message()),
        })?;
    let mut params: Vec<&'static Param> = params_of(command).collect();
    // iter_params_for_processing: eager first, then by first invocation, then declaration.
    params.sort_by_key(|p| {
        let idx = state
            .order
            .iter()
            .position(|d| *d == p.dest)
            .unwrap_or(usize::MAX);
        (!is_eager(p), idx)
    });
    for param in params {
        let (value, source) = consume_value(ctx, param, &state.opts);
        ctx.set_source(param.dest, source);
        let value = match process_value(ctx, param, value) {
            Ok((_, Eager::Help)) => return Err(Stop::Help(Box::new(ctx.clone()), 0)),
            Ok((_, Eager::Version)) => return Err(Stop::Version),
            Ok((value, Eager::No)) => value,
            Err(_) if ctx.resilient => None,
            Err(err) => return Err(err.into()),
        };
        if exposed(param) {
            match ctx.values.iter_mut().find(|(d, _)| *d == param.dest) {
                Some(entry) if entry.1.is_none() => entry.1 = value,
                Some(_) => {}
                None => ctx.values.push((param.dest, value)),
            }
        }
    }
    let rest = state.largs;
    if !rest.is_empty() && !command.is_group && !ctx.resilient {
        let noun = if rest.len() == 1 {
            "argument"
        } else {
            "arguments"
        };
        return Err(ctx
            .usage_error(format!("Got unexpected extra {noun} ({})", rest.join(" ")))
            .into());
    }
    if command.is_group {
        let mut rest = rest.into_iter();
        ctx.protected_args = rest.next().into_iter().collect();
        ctx.args = rest.collect();
    } else {
        ctx.args = rest;
    }
    Ok(())
}

/// `Command.make_context(info_name, args, parent=...)`.
fn make_context(
    command: &'static Command,
    path: Vec<String>,
    args: &[String],
    parent: Option<&Ctx>,
    resilient: bool,
) -> std::result::Result<Ctx, Stop> {
    let (default_map, carried) = match parent {
        Some(parent) => {
            let name = path.last().map(String::as_str);
            match (parent.path.len(), name) {
                (1, Some("dev")) => (None, parent.carried_map.clone()),
                (2, Some("start")) if parent.path[1] == "dev" => (parent.carried_map.clone(), None),
                _ => (None, None),
            }
        }
        None => (None, None),
    };
    let mut ctx = Ctx::new(command, path, default_map, carried, resilient);
    parse_command_args(&mut ctx, args)?;
    Ok(ctx)
}

/// `Group.resolve_command`: the subcommand for `args[0]` (None only when resilient).
fn resolve_command(
    ctx: &mut Ctx,
    args: &[String],
) -> std::result::Result<Option<&'static Command>, Stop> {
    let name = &args[0];
    let command = get_command(ctx.command, name);
    if command.is_none() && !ctx.resilient {
        if !split_opt(name).0.is_empty() {
            let mut again = ctx.clone();
            again.values.clear();
            again.sources.clear();
            parse_command_args(&mut again, args)?;
        }
        return Err(ctx
            .usage_error(format!("No such command {}.", repr_str(name)))
            .into());
    }
    Ok(command)
}

/// Default arguments of `command` when invoked via `ctx.invoke(cmd)` (main -> info).
fn default_args(command: &'static Command) -> Args {
    Args {
        values: command
            .params
            .iter()
            .map(|p| {
                let value = convert_const(p, p.default).unwrap_or(Value::None);
                (p.dest, value)
            })
            .collect(),
    }
}

/// Parse `argv` (without argv[0]) against [`super::commands::MAIN`]. `load_default_map`
/// is called at the point where click runs the main group callback (see module docs).
pub fn parse(argv: &[String], load_default_map: &dyn Fn() -> Result<DefaultMap>) -> Result<Parsed> {
    match parse_inner(argv, load_default_map) {
        Ok(parsed) => Ok(parsed),
        Err(Stop::Error(err)) => Err(err),
        Err(Stop::Version) => Ok(Parsed::Version),
        Err(Stop::Help(ctx, code)) => Ok(Parsed::Help {
            path: ctx.path,
            default_map: ctx.default_map,
            code,
        }),
    }
}

fn parse_inner(
    argv: &[String],
    load_default_map: &dyn Fn() -> Result<DefaultMap>,
) -> std::result::Result<Parsed, Stop> {
    let mut ctx = make_context(&MAIN, vec![MAIN.name.to_owned()], argv, None, false)?;
    loop {
        if !ctx.command.is_group {
            return Ok(Parsed::Run {
                path: ctx.path.clone(),
                command: ctx.command,
                args: ctx.into_args(),
            });
        }
        if ctx.protected_args.is_empty() {
            if ctx.command.invoke_without_command {
                ctx.carried_map = Some(load_default_map()?);
                let info = get_command(ctx.command, "info").expect("info command");
                let mut path = ctx.path.clone();
                path.push(info.name.to_owned());
                return Ok(Parsed::Run {
                    path,
                    command: info,
                    args: default_args(info),
                });
            }
            return Err(ctx.usage_error("Missing command.").into());
        }
        let mut args = std::mem::take(&mut ctx.protected_args);
        args.append(&mut ctx.args);
        let command = resolve_command(&mut ctx, &args)?.expect("resolved");
        if ctx.path.len() == 1 {
            ctx.carried_map = Some(load_default_map()?);
        }
        let mut path = ctx.path.clone();
        path.push(args[0].clone());
        ctx = make_context(command, path, &args[1..], Some(&ctx), false)?;
    }
}

/// click's `_resolve_context` (completion): parse the complete args resiliently and
/// follow subcommands. Never fails.
pub fn resolve_context(args: &[String]) -> Ctx {
    let resilient = |r: std::result::Result<Ctx, Stop>| match r {
        Ok(ctx) => Some(ctx),
        Err(Stop::Help(ctx, _)) => Some(*ctx),
        Err(_) => None,
    };
    let root_path = vec![MAIN.name.to_owned()];
    let Some(mut ctx) = resilient(make_context(&MAIN, root_path.clone(), args, None, true)) else {
        return Ctx::new(&MAIN, root_path, None, None, true);
    };
    let mut args: Vec<String> = ctx
        .protected_args
        .iter()
        .chain(&ctx.args)
        .cloned()
        .collect();
    while !args.is_empty() {
        if !ctx.command.is_group {
            break;
        }
        let Some(command) = get_command(ctx.command, &args[0]) else {
            return ctx;
        };
        let mut path = ctx.path.clone();
        path.push(args[0].clone());
        match resilient(make_context(command, path, &args[1..], Some(&ctx), true)) {
            Some(sub) => ctx = sub,
            None => return ctx,
        }
        args = ctx
            .protected_args
            .iter()
            .chain(&ctx.args)
            .cloned()
            .collect();
    }
    ctx
}

/// `difflib.get_close_matches(word, possibilities)` (n=3, cutoff=0.6).
pub fn get_close_matches<'a>(
    word: &str,
    possibilities: impl Iterator<Item = &'a str>,
) -> Vec<String> {
    let b: Vec<char> = word.chars().collect();
    let mut scored: Vec<(f64, &str)> = possibilities
        .filter_map(|x| {
            let a: Vec<char> = x.chars().collect();
            let ratio = sequence_ratio(&a, &b);
            (ratio >= 0.6).then_some((ratio, x))
        })
        .collect();
    // heapq.nlargest(3, (score, x)): by score, then x, descending.
    scored.sort_by(|l, r| r.0.total_cmp(&l.0).then_with(|| r.1.cmp(l.1)));
    scored
        .into_iter()
        .take(3)
        .map(|(_, x)| x.to_owned())
        .collect()
}

/// `SequenceMatcher(None, a, b).ratio()`.
fn sequence_ratio(a: &[char], b: &[char]) -> f64 {
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    let mut b2j: std::collections::HashMap<char, Vec<usize>> = std::collections::HashMap::new();
    for (j, ch) in b.iter().enumerate() {
        b2j.entry(*ch).or_default().push(j);
    }
    if b.len() >= 200 {
        let ntest = b.len() / 100 + 1;
        b2j.retain(|_, idxs| idxs.len() <= ntest);
    }
    let mut matches = 0;
    let mut queue = vec![(0, a.len(), 0, b.len())];
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = find_longest_match(a, b, &b2j, alo, ahi, blo, bhi);
        if k > 0 {
            matches += k;
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    2.0 * matches as f64 / total as f64
}

fn find_longest_match(
    a: &[char],
    b: &[char],
    b2j: &std::collections::HashMap<char, Vec<usize>>,
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> (usize, usize, usize) {
    let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0);
    let mut j2len: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (i, ch) in a.iter().enumerate().take(ahi).skip(alo) {
        let mut new: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        if let Some(js) = b2j.get(ch) {
            for &j in js {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let k = j
                    .checked_sub(1)
                    .and_then(|p| j2len.get(&p))
                    .copied()
                    .unwrap_or(0)
                    + 1;
                new.insert(j, k);
                if k > bestsize {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
        }
        j2len = new;
    }
    while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
        besti -= 1;
        bestj -= 1;
        bestsize += 1;
    }
    while besti + bestsize < ahi
        && bestj + bestsize < bhi
        && a[besti + bestsize] == b[bestj + bestsize]
    {
        bestsize += 1;
    }
    (besti, bestj, bestsize)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_map() -> Result<DefaultMap> {
        Ok(Vec::new())
    }

    fn parse_ok(argv: &[&str]) -> (Vec<String>, Args) {
        parse_with(argv, &no_map)
    }

    fn parse_with(argv: &[&str], map: &dyn Fn() -> Result<DefaultMap>) -> (Vec<String>, Args) {
        let argv: Vec<String> = argv.iter().map(|s| (*s).to_owned()).collect();
        match parse(&argv, map).expect("parses") {
            Parsed::Run { path, args, .. } => (path, args),
            other => panic!("not a run: {other:?}"),
        }
    }

    fn usage_error(argv: &[&str]) -> UsageError {
        let argv: Vec<String> = argv.iter().map(|s| (*s).to_owned()).collect();
        match parse(&argv, &no_map) {
            Err(AbError::Usage(err)) => err,
            other => panic!("expected a usage error, got {other:?}"),
        }
    }

    #[test]
    fn bare_invocation_runs_info() {
        let (path, args) = parse_ok(&[]);
        assert_eq!(path, ["bab", "info"]);
        assert!(!args.flag("all_partitions"));
        assert!(
            parse_ok(&["info", "--all-partitions"])
                .1
                .flag("all_partitions")
        );
    }

    #[test]
    fn queue_options_and_alias() {
        let (path, args) = parse_ok(&["q", "--user", "alice", "-pgpusmall", "--all"]);
        assert_eq!(path, ["bab", "q"]);
        assert_eq!(args.opt_str("user"), Some("alice"));
        assert_eq!(args.opt_str("partition"), Some("gpusmall"));
        assert!(args.flag("all_users"));
        assert_eq!(args.opt_float("watch"), None);
        assert_eq!(
            parse_ok(&["q", "-p", "a", "-p", "b"])
                .1
                .opt_str("partition"),
            Some("b")
        );
        assert_eq!(
            parse_ok(&["q", "--partition=x"]).1.opt_str("partition"),
            Some("x")
        );
    }

    #[test]
    fn watch_takes_an_optional_value() {
        assert_eq!(parse_ok(&["q", "-w"]).1.opt_float("watch"), Some(10.0));
        assert_eq!(
            parse_ok(&["q", "-w", "15"]).1.opt_float("watch"),
            Some(15.0)
        );
        assert_eq!(parse_ok(&["q", "-w15"]).1.opt_float("watch"), Some(15.0));
        assert_eq!(
            parse_ok(&["q", "--watch=30"]).1.opt_float("watch"),
            Some(30.0)
        );
        let (_, args) = parse_ok(&["q", "-w", "-p", "gpusmall"]);
        assert_eq!(args.opt_float("watch"), Some(10.0));
        assert_eq!(args.opt_str("partition"), Some("gpusmall"));
        assert_eq!(
            usage_error(&["q", "--watch", "5"]).message,
            "Invalid value for '-w' / '--watch': 5.0 is not in the range x>=10."
        );
        assert_eq!(
            usage_error(&["q", "--watch", "abc"]).message,
            "Invalid value for '-w' / '--watch': 'abc' is not a valid float range."
        );
    }

    #[test]
    fn sleeps_reject_what_time_sleep_cannot_take() {
        use super::super::commands::{POLL_INTERVAL, WATCH};
        use std::time::Duration;
        // click's FloatRange lets these through, like Python's
        for raw in ["nan", "inf", "1e10"] {
            assert!(parse_ok(&["q", "-w", raw]).1.opt_float("watch").is_some());
        }
        let message = |param: &Param, value: f64| match sleep_duration(param, value) {
            Err(AbError::Usage(err)) => {
                assert!(err.command_path.is_empty() && err.show_help_hint);
                err.message
            }
            other => panic!("expected a usage error, got {other:?}"),
        };
        assert_eq!(
            message(&POLL_INTERVAL, f64::NAN),
            "Invalid value for '--poll-interval': nan is not in the range \
             1<=x<9223372036.854776."
        );
        assert_eq!(
            message(&WATCH, f64::INFINITY),
            "Invalid value for '-w' / '--watch': inf is not in the range \
             10<=x<9223372036.854776."
        );
        assert!(message(&WATCH, 1e10).contains(": 10000000000.0 is not in the range"));
        assert_eq!(
            sleep_duration(&POLL_INTERVAL, 2.5),
            Ok(Duration::from_millis(2500))
        );
        // time.sleep(9.2e9) is accepted by Python (it just sleeps very long)
        assert!(sleep_duration(&WATCH, 9.2e9).is_ok());
    }

    #[test]
    fn why_fix_flag_value_and_choices() {
        let (_, args) = parse_ok(&["why", "65177", "--fix"]);
        assert_eq!(args.opt_str("fix"), Some("auto"));
        assert_eq!(args.opt_str("jobid"), Some("65177"));
        let (_, args) = parse_ok(&["why", "65177", "--nodes", "--fix", "cpus", "-y"]);
        assert_eq!(args.opt_str("fix"), Some("cpus"));
        assert!(args.flag("show_nodes") && args.flag("assume_yes"));
        assert_eq!(
            usage_error(&["why", "65177", "--fix", "disk"]).message,
            "Invalid value for '--fix': 'disk' is not one of 'auto', 'mem', 'cpus', 'gpus'."
        );
        assert_eq!(parse_ok(&["why"]).1.opt_str("jobid"), None);
    }

    #[test]
    fn jobs_and_dev_defaults() {
        assert_eq!(parse_ok(&["job", "history"]).1.int("limit"), 50);
        assert_eq!(
            parse_ok(&["job", "history", "--limit", "100"])
                .1
                .int("limit"),
            100
        );
        let (_, args) = parse_ok(&["job", "output", "--stderr", "12345"]);
        assert_eq!((args.str("jobid"), args.flag("stderr")), ("12345", true));
        let err = usage_error(&["job", "output"]);
        assert_eq!(err.message, "Missing argument 'JOBID'.");
        assert_eq!(err.command_path, ["bab", "job", "output"]);
        let (_, args) = parse_ok(&["dev", "start", "-g", "2", "-c4"]);
        assert_eq!(args.str("partition"), "auto");
        assert_eq!(args.str("time_limit"), "08:00:00");
        assert_eq!((args.opt_int("gpu"), args.int("cpus")), (Some(2), 4));
        assert_eq!(args.float("poll_interval"), 5.0);
        let (path, args) = parse_ok(&["l", "--command", "nvidia-smi"]);
        assert_eq!(path, ["bab", "l"]);
        assert_eq!(args.opt_str("command"), Some("nvidia-smi"));
        assert!(!args.flag("srun"));
        assert_eq!(
            usage_error(&["l", "--poll-interval", "0"]).command_path,
            ["bab", "l"]
        );
    }

    #[test]
    fn default_map_applies_to_dev_start_only() {
        let map = || Ok(vec![("gpu", "3".to_owned()), ("mem", "32G".to_owned())]);
        let (_, args) = parse_with(&["dev", "start", "-g", "1"], &map);
        assert_eq!((args.opt_int("gpu"), args.str("mem")), (Some(1), "32G"));
        let bad = || Ok(vec![("cpus", "abc".to_owned())]);
        let argv = vec!["dev".to_owned(), "start".to_owned()];
        match parse(&argv, &bad) {
            Err(AbError::Usage(err)) => assert_eq!(
                err.message,
                "Invalid value for '-c' / '--cpus': 'abc' is not a valid integer."
            ),
            other => panic!("{other:?}"),
        }
        let failing = || Err(AbError::Click("broken".to_owned()));
        let argv = vec!["info".to_owned(), "-h".to_owned()];
        assert_eq!(
            parse(&argv, &failing),
            Err(AbError::Click("broken".to_owned()))
        );
        let argv = vec!["-h".to_owned()];
        assert!(matches!(
            parse(&argv, &failing),
            Ok(Parsed::Help { code: 0, .. })
        ));
    }

    #[test]
    fn unknown_options_and_commands() {
        assert_eq!(
            usage_error(&["q", "--al"]).message,
            "No such option: --al (Possible options: --all, --help)"
        );
        assert_eq!(
            usage_error(&["info", "--all-partition"]).message,
            "No such option: --all-partition Did you mean --all-partitions?"
        );
        assert_eq!(usage_error(&["q", "-x"]).message, "No such option: -x");
        assert_eq!(usage_error(&["foo"]).message, "No such command 'foo'.");
        assert_eq!(
            usage_error(&["q", "a", "b"]).message,
            "Got unexpected extra arguments (a b)"
        );
        let err = usage_error(&["q", "--user"]);
        assert_eq!(err.message, "Option '--user' requires an argument.");
        assert!(err.command_path.is_empty() && !err.show_help_hint);
    }

    #[test]
    fn help_version_and_groups() {
        let parse_raw = |argv: &[&str]| {
            let argv: Vec<String> = argv.iter().map(|s| (*s).to_owned()).collect();
            parse(&argv, &no_map).unwrap()
        };
        assert_eq!(parse_raw(&["--version", "-h"]), Parsed::Version);
        assert!(matches!(parse_raw(&["dev"]), Parsed::Help { code: 2, .. }));
        match parse_raw(&["q", "--watch", "5", "-h"]) {
            Parsed::Help { path, code, .. } => {
                assert_eq!((path, code), (vec!["bab".into(), "q".into()], 0))
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(usage_error(&["dev", "--"]).message, "Missing command.");
    }

    #[test]
    fn gpualloc_is_hidden_and_l_follows_dev() {
        let names: Vec<&str> = list_commands(&MAIN).iter().map(|c| c.name).collect();
        assert_eq!(
            names,
            [
                "completion",
                "config",
                "dev",
                "l",
                "free",
                "gpualloc",
                "info",
                "job",
                "node",
                "queue",
                "why"
            ]
        );
        assert!(get_command(&MAIN, "gpualloc").unwrap().hidden);
        assert_eq!(
            parse_ok(&["gpualloc", "gpu042"]).1.opt_str("node"),
            Some("gpu042")
        );
    }

    #[test]
    fn python_number_formats() {
        assert_eq!(py_float_repr(5.0), "5.0");
        assert_eq!(py_float_repr(1e-5), "1e-05");
        assert_eq!(py_float_repr(1e16), "1e+16");
        assert_eq!(py_float_repr(123456.789), "123456.789");
        assert_eq!(py_float_repr(0.0001), "0.0001");
        assert_eq!(py_int(" -5 "), Some(-5));
        assert_eq!(py_int("1_000"), Some(1000));
        assert_eq!(py_int("1__0"), None);
        assert_eq!(py_float("1_0.5"), Some(10.5));
        assert_eq!(py_float("5."), Some(5.0));
        assert_eq!(py_float(".5e1"), Some(5.0));
        assert!(py_float("Infinity").unwrap().is_infinite());
        assert_eq!(py_float("e5"), None);
    }

    #[test]
    fn close_matches_like_difflib() {
        let options = ["--all", "--user", "--partition", "--watch", "--help"];
        assert_eq!(
            get_close_matches("--al", options.iter().copied()),
            ["--all", "--help"]
        );
        assert_eq!(
            get_close_matches("--wat", options.iter().copied()),
            ["--watch", "--all"]
        );
        assert!(get_close_matches("--zzzzzz", options.iter().copied()).is_empty());
    }
}
