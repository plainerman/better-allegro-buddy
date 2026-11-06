//! Shell completion: click 8.3.2's completion protocol and scripts, plus `bab completion`.
//! Owned by P2-B (protocol, scripts) and P3-F (command wiring/`--install`).
//!
//! * Protocol (click `_main_shell_completion` / `shell_complete`): when
//!   `_{PROG}_COMPLETE` (prog = basename of argv[0], `-`/`.` -> `_`, upper-cased) is set
//!   to `{bash,zsh,fish}_{source,complete}`, print the script or answer the request from
//!   `COMP_WORDS`/`COMP_CWORD` (`BashComplete`/`ZshComplete`/`FishComplete` formats) and
//!   exit 0; an unknown shell or instruction exits 1 silently.
//! * `bab completion [SHELL] [--prog P] [--install]`: Python `completion_command`,
//!   `_completion_var`, `_completion_destinations`, `_install_completion`.

use super::commands::{Command, Completer, ParamKind, ParamType};
use super::parser::{self, Ctx, Source};
use crate::core::RecordExt;
use crate::error::{AbError, Result, UsageError};
use std::path::Path;

const SOURCE_BASH: &str = r#"%(complete_func)s() {
    local IFS=$'\n'
    local response

    response=$(env COMP_WORDS="${COMP_WORDS[*]}" COMP_CWORD=$COMP_CWORD %(complete_var)s=bash_complete $1)

    for completion in $response; do
        IFS=',' read type value <<< "$completion"

        if [[ $type == 'dir' ]]; then
            COMPREPLY=()
            compopt -o dirnames
        elif [[ $type == 'file' ]]; then
            COMPREPLY=()
            compopt -o default
        elif [[ $type == 'plain' ]]; then
            COMPREPLY+=($value)
        fi
    done

    return 0
}

%(complete_func)s_setup() {
    complete -o nosort -F %(complete_func)s %(prog_name)s
}

%(complete_func)s_setup;
"#;

const SOURCE_ZSH: &str = r#"#compdef %(prog_name)s

%(complete_func)s() {
    local -a completions
    local -a completions_with_descriptions
    local -a response
    (( ! $+commands[%(prog_name)s] )) && return 1

    response=("${(@f)$(env COMP_WORDS="${words[*]}" COMP_CWORD=$((CURRENT-1)) %(complete_var)s=zsh_complete %(prog_name)s)}")

    for type key descr in ${response}; do
        if [[ "$type" == "plain" ]]; then
            if [[ "$descr" == "_" ]]; then
                completions+=("$key")
            else
                completions_with_descriptions+=("$key":"$descr")
            fi
        elif [[ "$type" == "dir" ]]; then
            _path_files -/
        elif [[ "$type" == "file" ]]; then
            _path_files -f
        fi
    done

    if [ -n "$completions_with_descriptions" ]; then
        _describe -V unsorted completions_with_descriptions -U
    fi

    if [ -n "$completions" ]; then
        compadd -U -V unsorted -a completions
    fi
}

if [[ $zsh_eval_context[-1] == loadautofunc ]]; then
    # autoload from fpath, call function directly
    %(complete_func)s "$@"
else
    # eval/source/. command, register function for later
    compdef %(complete_func)s %(prog_name)s
fi
"#;

const SOURCE_FISH: &str = r#"function %(complete_func)s;
    set -l response (env %(complete_var)s=fish_complete COMP_WORDS=(commandline -cp) COMP_CWORD=(commandline -t) %(prog_name)s);

    for completion in $response;
        set -l metadata (string split "," $completion);

        if test $metadata[1] = "dir";
            __fish_complete_directories $metadata[2];
        else if test $metadata[1] = "file";
            __fish_complete_path $metadata[2];
        else if test $metadata[1] = "plain";
            echo $metadata[2];
        end;
    end;
end;

complete --no-files --command %(prog_name)s --arguments "(%(complete_func)s)";
"#;

const SHELLS: [&str; 3] = ["bash", "zsh", "fish"];

/// Handle a completion request if the completion env var for `prog_name` is set.
/// Returns `Some(exit_code)` when it was a completion invocation.
pub fn handle_env_request(prog_name: &str) -> Option<i32> {
    let complete_var = format!("_{}_COMPLETE", prog_name.replace(['-', '.'], "_")).to_uppercase();
    let instruction = std::env::var(&complete_var)
        .ok()
        .filter(|v| !v.is_empty())?;
    let (shell, instruction) = instruction.split_once('_').unwrap_or((&instruction, ""));
    if !SHELLS.contains(&shell) {
        return Some(1);
    }
    match instruction {
        "source" => {
            super::echo(&source(shell, prog_name, &complete_var));
            Some(0)
        }
        "complete" => match complete(shell) {
            Some(out) => {
                super::echo(&out);
                Some(0)
            }
            None => Some(1),
        },
        _ => Some(1),
    }
}

/// `_completion_var(prog)`.
pub fn completion_var(prog: &str) -> String {
    let normalized: String = prog
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { '_' })
        .collect();
    let normalized = normalized.trim_matches('_');
    if normalized.is_empty() {
        "_AB_COMPLETE".to_owned()
    } else {
        format!("_{}_COMPLETE", normalized.to_uppercase())
    }
}

/// `ShellComplete.func_name`.
fn func_name(prog: &str) -> String {
    let safe: String = prog
        .replace('-', "_")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    format!("_{safe}_completion")
}

/// Click's completion script for `shell` (`bash`, `zsh`, `fish`), `prog`, `complete_var`
/// (`ShellComplete.source()`; bash also runs click's Bash version check).
pub fn source(shell: &str, prog: &str, complete_var: &str) -> String {
    let template = match shell {
        "bash" => {
            check_bash_version();
            SOURCE_BASH
        }
        "zsh" => SOURCE_ZSH,
        _ => SOURCE_FISH,
    };
    template
        .replace("%(complete_func)s", &func_name(prog))
        .replace("%(complete_var)s", complete_var)
        .replace("%(prog_name)s", prog)
}

/// `BashComplete._check_version`: warn on stderr when bash is missing or older than 4.4.
fn check_bash_version() {
    let version = which("bash").and_then(|bash| {
        std::process::Command::new(bash)
            .args(["--norc", "-c", "echo \"${BASH_VERSION}\""])
            .stdin(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .output()
            .ok()
    });
    let parsed = version.and_then(|out| {
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        let mut parts = text.splitn(3, '.');
        let major: String = parts.next()?.to_owned();
        let minor: String = parts.next()?.to_owned();
        let patch = parts.next()?;
        let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
        let patch_ok = patch.bytes().next().is_some_and(|b| b.is_ascii_digit());
        (digits(&major) && digits(&minor) && patch_ok).then_some((major, minor))
    });
    let mut err = std::io::stderr();
    match parsed {
        Some((major, minor)) => {
            if major.as_str() < "4" || (major == "4" && minor.as_str() < "4") {
                super::write_stream(
                    &mut err,
                    "Shell completion is not supported for Bash versions older than 4.4.\n",
                );
            }
        }
        None => super::write_stream(
            &mut err,
            "Couldn't detect Bash version, shell completion is not supported.\n",
        ),
    }
}

/// `shutil.which(name)`.
fn which(name: &str) -> Option<String> {
    let path = std::env::var("PATH").unwrap_or_else(|_| "/bin:/usr/bin".to_owned());
    if path.is_empty() {
        return None;
    }
    let mut seen = Vec::new();
    for dir in path.split(':') {
        if seen.contains(&dir) {
            continue;
        }
        seen.push(dir);
        let candidate = if dir.is_empty() || dir.ends_with('/') {
            format!("{dir}{name}")
        } else {
            format!("{dir}/{name}")
        };
        let c = std::ffi::CString::new(candidate.clone()).ok()?;
        let executable = unsafe { libc::access(c.as_ptr(), libc::X_OK) } == 0;
        if executable && Path::new(&candidate).exists() && !Path::new(&candidate).is_dir() {
            return Some(candidate);
        }
    }
    None
}

/// A completion suggestion (click `CompletionItem`, always type `plain` here).
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub value: String,
    pub help: Option<String>,
}

/// Python list slicing/indexing helpers for `cwords[1:cword]` / `cwords[cword]`.
fn py_index(len: usize, index: i64) -> Option<usize> {
    let resolved = if index < 0 { len as i64 + index } else { index };
    (0..len as i64)
        .contains(&resolved)
        .then_some(resolved as usize)
}

fn py_slice_end(len: usize, end: i64) -> usize {
    let end = if end < 0 { len as i64 + end } else { end };
    end.clamp(0, len as i64) as usize
}

/// `get_completion_args` + `get_completions` + `format_completion` for `shell`.
fn complete(shell: &str) -> Option<String> {
    let words = split_arg_string(&std::env::var("COMP_WORDS").ok()?);
    let cword = std::env::var("COMP_CWORD").ok()?;
    let (args, incomplete) = if shell == "fish" {
        let incomplete = if cword.is_empty() {
            String::new()
        } else {
            split_arg_string(&cword)
                .into_iter()
                .next()
                .unwrap_or_default()
        };
        let mut args: Vec<String> = words.iter().skip(1).cloned().collect();
        if !incomplete.is_empty() && args.last() == Some(&incomplete) {
            args.pop();
        }
        (args, incomplete)
    } else {
        let cword: i64 = cword.trim().parse().ok()?;
        let end = py_slice_end(words.len(), cword);
        let args = words.get(1..end.max(1)).unwrap_or_default().to_vec();
        let incomplete = py_index(words.len(), cword)
            .map(|i| words[i].clone())
            .unwrap_or_default();
        (args, incomplete)
    };
    let items = get_completions(args, &incomplete);
    let lines: Vec<String> = items
        .into_iter()
        .map(|item| match shell {
            "bash" => format!("plain,{}", item.value),
            "zsh" => {
                let help = item.help.as_deref().unwrap_or("_");
                let help = if help.is_empty() { "_" } else { help };
                let value = if help != "_" {
                    item.value.replace(':', "\\:")
                } else {
                    item.value
                };
                format!("plain\n{value}\n{help}")
            }
            _ => match item.help.as_deref().filter(|h| !h.is_empty()) {
                Some(help) => format!("plain,{}\t{help}", item.value),
                None => format!("plain,{}", item.value),
            },
        })
        .collect();
    Some(lines.join("\n"))
}

/// `ShellComplete.get_completions(args, incomplete)`.
pub fn get_completions(mut args: Vec<String>, incomplete: &str) -> Vec<Item> {
    let ctx = parser::resolve_context(&args);
    let mut incomplete = incomplete.to_owned();
    // _resolve_incomplete
    if incomplete == "=" {
        incomplete.clear();
    } else if incomplete.contains('=') && incomplete.starts_with('-') {
        let (name, rest) = incomplete.split_once('=').unwrap();
        args.push(name.to_owned());
        incomplete = rest.to_owned();
    }
    if !args.iter().any(|a| a == "--") && incomplete.starts_with('-') {
        return command_completions(&ctx, &incomplete);
    }
    let params: Vec<_> = parser::params_of(ctx.command).collect();
    for param in &params {
        if param.kind == ParamKind::Option && !parser::is_flag(param) {
            let last_option = args.iter().rev().take(1).find(|a| a.starts_with('-'));
            if last_option.is_some_and(|opt| param.decls.contains(&opt.as_str())) {
                return param_completions(param, &incomplete);
            }
        }
    }
    for param in &params {
        if param.kind == ParamKind::Argument && ctx.source(param.dest) != Some(Source::CommandLine)
        {
            return param_completions(param, &incomplete);
        }
    }
    command_completions(&ctx, &incomplete)
}

/// `Command.shell_complete` / `Group.shell_complete`.
fn command_completions(ctx: &Ctx, incomplete: &str) -> Vec<Item> {
    let mut items = Vec::new();
    if ctx.command.is_group {
        for command in parser::list_commands(ctx.command) {
            if command.name.starts_with(incomplete) && !command.hidden {
                items.push(Item {
                    value: command.name.to_owned(),
                    help: Some(short_help_str(command)),
                });
            }
        }
    }
    if incomplete
        .chars()
        .next()
        .is_some_and(|c| !c.is_alphanumeric())
    {
        for param in parser::params_of(ctx.command) {
            if param.kind != ParamKind::Option
                || param.hidden
                || ctx.source(param.dest) == Some(Source::CommandLine)
            {
                continue;
            }
            // list(set(['-h', '--help'])) is ['--help', '-h'] for most hash seeds.
            let decls: &[&str] = if param.dest == "help" {
                &["--help", "-h"]
            } else {
                param.decls
            };
            for name in decls {
                if name.starts_with(incomplete) {
                    items.push(Item {
                        value: (*name).to_owned(),
                        help: param.help.map(str::to_owned),
                    });
                }
            }
        }
    }
    items
}

/// `Command.get_short_help_str()`.
pub fn short_help_str(command: &Command) -> String {
    command
        .short_help
        .or(command.help)
        .unwrap_or("")
        .trim()
        .to_owned()
}

/// `Parameter.shell_complete`: the custom completer, else the type's (Choice).
fn param_completions(param: &super::commands::Param, incomplete: &str) -> Vec<Item> {
    let plain = |values: Vec<String>| {
        values
            .into_iter()
            .map(|value| Item { value, help: None })
            .collect()
    };
    if let Some(completer) = param.completer {
        let ids: Vec<String> = match completer {
            Completer::JobHistory => {
                crate::core::jobs::job_history_completion_ids(incomplete, None)
            }
            Completer::DevJob => crate::core::jobs::find_dev_jobs(None, None)
                .unwrap_or_default()
                .iter()
                .map(|job| job.get_str("JobId").to_owned())
                .filter(|id| id.starts_with(incomplete))
                .collect(),
            Completer::PendingJob => crate::core::jobs::load_user_pending_jobs(None)
                .unwrap_or_default()
                .iter()
                .map(|job| job.get_str("JobID").to_owned())
                .filter(|id| id.starts_with(incomplete))
                .collect(),
        };
        return plain(ids);
    }
    match param.ty {
        ParamType::Choice(choices) => plain(
            choices
                .iter()
                .filter(|c| c.starts_with(incomplete))
                .map(|c| (*c).to_owned())
                .collect(),
        ),
        _ => Vec::new(),
    }
}

/// click's `split_arg_string`: `shlex.split` (posix, whitespace_split, no comments) that
/// keeps an unterminated last token instead of failing.
pub fn split_arg_string(input: &str) -> Vec<String> {
    #[derive(Clone, Copy, PartialEq)]
    enum State {
        Space,
        Word,
        Quote(char),
        Escape(char),
    }
    let is_space = |c: char| matches!(c, ' ' | '\t' | '\r' | '\n');
    let mut out = Vec::new();
    let mut token = String::new();
    let mut quoted = false;
    let mut state = State::Space;
    for ch in input.chars() {
        match state {
            State::Space => {
                if is_space(ch) {
                    if !token.is_empty() || quoted {
                        out.push(std::mem::take(&mut token));
                        quoted = false;
                    }
                } else if ch == '\\' {
                    state = State::Escape('a');
                } else if ch == '\'' || ch == '"' {
                    state = State::Quote(ch);
                } else {
                    token.push(ch);
                    state = State::Word;
                }
            }
            State::Quote(q) => {
                quoted = true;
                if ch == q {
                    state = State::Word;
                } else if ch == '\\' && q == '"' {
                    state = State::Escape(q);
                } else {
                    token.push(ch);
                }
            }
            State::Escape(prev) => {
                if prev != 'a' && ch != '\\' && ch != prev {
                    token.push('\\');
                }
                token.push(ch);
                state = if prev == 'a' {
                    State::Word
                } else {
                    State::Quote(prev)
                };
            }
            State::Word => {
                if is_space(ch) {
                    state = State::Space;
                    if !token.is_empty() || quoted {
                        out.push(std::mem::take(&mut token));
                        quoted = false;
                    }
                } else if ch == '\'' || ch == '"' {
                    state = State::Quote(ch);
                } else if ch == '\\' {
                    state = State::Escape('a');
                } else {
                    token.push(ch);
                }
            }
        }
    }
    match state {
        State::Quote(_) | State::Escape(_) => out.push(token),
        _ if !token.is_empty() || quoted => out.push(token),
        _ => {}
    }
    out
}

/// Python `Path(s)` -> `str` normalisation (collapse `//`, drop `.` parts and the
/// trailing slash).
fn py_path(s: &str) -> String {
    let root = if s.starts_with("//") && !s.starts_with("///") {
        "//"
    } else if s.starts_with('/') {
        "/"
    } else {
        ""
    };
    let parts: Vec<&str> = s
        .split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .collect();
    let joined = format!("{root}{}", parts.join("/"));
    if joined.is_empty() {
        ".".to_owned()
    } else {
        joined
    }
}

fn py_join(base: &str, rest: &str) -> String {
    py_path(&format!("{base}/{rest}"))
}

/// `Path.home()`.
fn home() -> String {
    let home = match std::env::var("HOME") {
        Ok(home) => home,
        Err(_) => passwd_home().unwrap_or_default(),
    };
    let trimmed = home.trim_end_matches('/');
    py_path(if trimmed.is_empty() { "/" } else { trimmed })
}

fn passwd_home() -> Option<String> {
    unsafe {
        let pw = libc::getpwuid(libc::getuid());
        if pw.is_null() {
            return None;
        }
        Some(
            std::ffi::CStr::from_ptr((*pw).pw_dir)
                .to_string_lossy()
                .into_owned(),
        )
    }
}

fn env_or(name: &str, fallback: impl FnOnce() -> String) -> String {
    match std::env::var(name) {
        Ok(value) if !value.is_empty() => py_path(&value),
        _ => fallback(),
    }
}

/// `_completion_destinations(shell, prog)`: (script, startup file sourcing it).
fn completion_destinations(shell: &str, prog: &str) -> (String, Option<String>) {
    if shell == "fish" {
        let config = env_or("XDG_CONFIG_HOME", || py_join(&home(), ".config"));
        return (
            py_join(&config, &format!("fish/completions/{prog}.fish")),
            None,
        );
    }
    let data = env_or("XDG_DATA_HOME", || py_join(&home(), ".local/share"));
    let script = py_join(&data, &format!("allegro-buddy/completion-{prog}.{shell}"));
    if shell == "bash" {
        return (script, Some(py_join(&home(), ".bashrc")));
    }
    let zdotdir = env_or("ZDOTDIR", home);
    (script, Some(py_join(&zdotdir, ".zshrc")))
}

fn io_error(path: &str, err: &std::io::Error) -> AbError {
    crate::error::die(format!("{path}: {err}"))
}

/// `_install_completion(shell, prog, source)`.
fn install_completion(shell: &str, prog: &str, source: &str) -> Result<()> {
    let (script, startup) = completion_destinations(shell, prog);
    let source = if shell == "zsh" {
        format!("(( $+functions[compdef] )) || {{ autoload -Uz compinit && compinit; }}\n{source}")
    } else {
        source.to_owned()
    };
    if let Some(parent) = Path::new(&script).parent() {
        std::fs::create_dir_all(parent).map_err(|e| io_error(&script, &e))?;
    }
    std::fs::write(&script, format!("{source}\n")).map_err(|e| io_error(&script, &e))?;
    let mut note = String::new();
    if let Some(startup) = startup {
        let line = format!("[ -f '{script}' ] && . '{script}'");
        let existing = if Path::new(&startup).is_file() {
            std::fs::read_to_string(&startup).map_err(|e| io_error(&startup, &e))?
        } else {
            String::new()
        };
        if existing.contains(&line) {
            note = format!("; {startup} already sources it");
        } else {
            let separator = if existing.is_empty() || existing.ends_with('\n') {
                ""
            } else {
                "\n"
            };
            use std::io::Write;
            let mut rc = std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&startup)
                .map_err(|e| io_error(&startup, &e))?;
            rc.write_all(format!("{separator}{line}\n").as_bytes())
                .map_err(|e| io_error(&startup, &e))?;
            note = format!(", sourced from {startup}");
        }
    }
    super::echo(&format!(
        "Installed {shell} completion{note}. Open a new shell to use it."
    ));
    Ok(())
}

/// `completion_command(shell, prog, install)`.
pub fn cmd_completion(shell: Option<&str>, prog: &str, install: bool) -> Result<()> {
    let shell = match shell {
        Some(shell) => shell.to_owned(),
        None => {
            let detected = std::env::var("SHELL").unwrap_or_default();
            let name = py_path(&detected)
                .rsplit('/')
                .next()
                .filter(|n| *n != ".")
                .unwrap_or("")
                .to_owned();
            if !SHELLS.contains(&name.as_str()) {
                let shown = if name.is_empty() {
                    "unset"
                } else {
                    name.as_str()
                };
                return Err(AbError::Usage(UsageError {
                    message: format!(
                        "cannot detect a supported shell from $SHELL ({}); pass one of bash, zsh, fish",
                        crate::pyfmt::repr_str(shown)
                    ),
                    command_path: Vec::new(),
                    show_help_hint: true,
                }));
            }
            name
        }
    };
    let script = source(&shell, prog, &completion_var(prog));
    if install {
        install_completion(&shell, prog, &script)
    } else {
        super::echo(&script);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_var_like_python() {
        assert_eq!(completion_var("bab"), "_BAB_COMPLETE");
        assert_eq!(completion_var("allegro-buddy"), "_ALLEGRO_BUDDY_COMPLETE");
        assert_eq!(completion_var("--"), "_AB_COMPLETE");
        assert_eq!(completion_var("a.b"), "_A_B_COMPLETE");
    }

    #[test]
    fn split_arg_string_like_shlex() {
        assert_eq!(split_arg_string("bab dev  start"), ["bab", "dev", "start"]);
        assert_eq!(split_arg_string("example 'my file"), ["example", "my file"]);
        assert_eq!(split_arg_string("example my\\"), ["example", "my"]);
        assert_eq!(split_arg_string("a '' b"), ["a", "", "b"]);
        assert_eq!(split_arg_string(r#"a "x\"y\z" c"#), ["a", r#"x"y\z"#, "c"]);
        assert_eq!(split_arg_string("a\\ b"), ["a b"]);
        assert_eq!(split_arg_string("bab \""), ["bab", ""]);
    }

    #[test]
    fn py_path_normalizes() {
        assert_eq!(py_path("/home/x/"), "/home/x");
        assert_eq!(py_path("//a//b/./c"), "//a/b/c");
        assert_eq!(py_path("///a"), "/a");
        assert_eq!(py_path(""), ".");
    }
}
