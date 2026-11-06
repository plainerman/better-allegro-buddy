//! Layered configuration: shipped cluster preset < site file < user file. Port of
//! `settings.py`. Owned by P2-C.
//!
//! # Contract
//!
//! * [`load`] == `functools.cache`d `settings.load()`: computed once per process on
//!   success (a `OnceLock`); errors are NOT cached (Python re-raises next call).
//!   [`get`] is the infallible accessor for code that runs after the cli layer has loaded
//!   settings successfully (it builds the `[dev]` default map first, turning a
//!   [`ConfigError`] into `AbError::Click(message)`); `get` panics if `load` would fail,
//!   so completion callbacks must use [`load`] and degrade gracefully.
//! * Values keep Python dict insertion order across layers (a key first set by the preset
//!   keeps its position when the user file overrides it): `gpu_labels` order is the
//!   display order of GPU columns. `config show` sorts by (section, key) itself.
//! * INI parsing must match `configparser.ConfigParser(interpolation=None,
//!   inline_comment_prefixes=('#', ';'))` incl. its error messages in `cannot parse
//!   {source}: {exc}` (e.g. `MissingSectionHeaderError`, `DuplicateOptionError`,
//!   `DuplicateSectionError`, `ParsingError` texts), keys lower-cased (optionxform),
//!   multi-line continuation values, `[DEFAULT]` semantics (DEFAULT is not in
//!   `sections()` but its keys appear in every section's `items()`!), `=`/`:` delimiters,
//!   full-line comments `#`/`;`, empty lines.
//! * Error messages use Python `repr` (`{raw_key!r}`, `sorted(...)` lists): see
//!   [`crate::pyfmt`].
//! * Presets are embedded with `include_str!` ([`PRESETS`]).
//! * The cluster name cache lives at `$XDG_CACHE_HOME|~/.cache/allegro-buddy/
//!   cluster-name.<hostname>` (hostname via [`crate::hooks::hostname`]); probed with
//!   `scontrol show config` (unchecked [`crate::core::slurm::try_run`]: never prints, a
//!   missing scontrol is just None), regex `^ClusterName\s*=\s*(\S+)` multiline, rc must
//!   be 0. Write errors ignored. Like Python, the file is read on every call (a failed
//!   probe is repeated by the next caller).
//! * Unit tests get their settings from `crate::hooks::testing::Ctx` (default: the
//!   allegro preset alone, like `tests/conftest.py`); [`load_with_env`] is the testable
//!   core of [`load`].

use crate::pyfmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// `ab dev start` options settable in `[dev]`: config key -> click parameter name.
pub const DEV_START_KEYS: [(&str, &str); 9] = [
    ("partition", "partition"),
    ("time", "time_limit"),
    ("gpu", "gpu"),
    ("gpu_type", "gpu_type"),
    ("cpus", "cpus"),
    ("mem", "mem"),
    ("account", "account"),
    ("constraint", "constraint"),
    ("exclude", "exclude"),
];

/// Allowed sections and keys; `None` = any key (lookup table). Python `SECTIONS`
/// (`dev` = DEV_START_KEYS + `log_dir`, `max_allocations`).
pub const SECTIONS: [(&str, Option<&[&str]>); 4] = [
    (
        "dev",
        Some(&[
            "partition",
            "time",
            "gpu",
            "gpu_type",
            "cpus",
            "mem",
            "account",
            "constraint",
            "exclude",
            "log_dir",
            "max_allocations",
        ]),
    ),
    ("cluster", Some(&["preset", "gpu_partitions", "node_login"])),
    ("gpu-aliases", None),
    ("gpu-labels", None),
];

pub const NODE_LOGIN_METHODS: [&str; 2] = ["ssh", "srun"];

/// Shipped presets (name, INI text), sorted by name (`preset_names()`).
pub const PRESETS: [(&str, &str); 2] = [
    ("allegro", include_str!("../presets/allegro.ini")),
    ("bhlrn4", include_str!("../presets/bhlrn4.ini")),
];

/// `settings.ConfigError`; the message is shown verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

/// One merged value: `(section, key) -> (value, source)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub section: String,
    pub key: String,
    pub value: String,
    /// `preset allegro`, `site file`, `user file`.
    pub source: String,
}

/// `Settings` (frozen dataclass + cached properties).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Settings {
    pub preset: Option<String>,
    pub preset_reason: String,
    /// Insertion-ordered like the Python dict (see module docs).
    pub values: Vec<Entry>,
    /// `gpu_aliases`: normalized alias -> normalized target (ordered).
    pub gpu_aliases: Vec<(String, String)>,
    /// `gpu_labels`: canonical type -> label, configured (display) order.
    pub gpu_labels: Vec<(String, String)>,
}

impl Settings {
    /// `get(section, key)`.
    pub fn get(&self, section: &str, key: &str) -> Option<&str> {
        self.values
            .iter()
            .find(|e| e.section == section && e.key == key)
            .map(|e| e.value.as_str())
    }

    /// `get_list(section, key)`: comma-split, stripped, empties dropped.
    pub fn get_list(&self, section: &str, key: &str) -> Vec<String> {
        self.get(section, key)
            .unwrap_or("")
            .split(',')
            .map(pyfmt::strip)
            .filter(|item| !item.is_empty())
            .map(str::to_owned)
            .collect()
    }

    /// `section(section)`: key -> value pairs in order.
    pub fn section(&self, section: &str) -> Vec<(String, String)> {
        self.values
            .iter()
            .filter(|e| e.section == section)
            .map(|e| (e.key.clone(), e.value.clone()))
            .collect()
    }

    /// `gpu_aliases.get(normalized)`.
    pub fn gpu_alias(&self, normalized: &str) -> Option<&str> {
        lookup(&self.gpu_aliases, normalized)
    }

    /// `gpu_labels.get(gpu_type)`.
    pub fn gpu_label(&self, gpu_type: &str) -> Option<&str> {
        lookup(&self.gpu_labels, gpu_type)
    }

    /// The `Settings(...)` constructor plus its cached properties.
    fn new(preset: Option<String>, preset_reason: String, values: Vec<Entry>) -> Settings {
        let mut settings = Settings {
            preset,
            preset_reason,
            values,
            gpu_aliases: Vec::new(),
            gpu_labels: Vec::new(),
        };
        for (alias, target) in settings.section("gpu-aliases") {
            insert_ordered(
                &mut settings.gpu_aliases,
                normalize_gpu_type(&alias),
                normalize_gpu_type(&target),
            );
        }
        for (gpu_type, label) in settings.section("gpu-labels") {
            insert_ordered(
                &mut settings.gpu_labels,
                normalize_gpu_type(&gpu_type),
                label,
            );
        }
        settings
    }
}

fn lookup<'a>(pairs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    pairs
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// `dict[key] = value`: a new key goes last, an existing one keeps its place.
fn insert_ordered<K: PartialEq, V>(pairs: &mut Vec<(K, V)>, key: K, value: V) {
    match pairs.iter_mut().find(|(k, _)| *k == key) {
        Some(slot) => slot.1 = value,
        None => pairs.push((key, value)),
    }
}

static SETTINGS: OnceLock<Settings> = OnceLock::new();

/// Environment lookup (`os.environ.get`), injectable for tests.
pub type Env<'a> = &'a dyn Fn(&str) -> Option<String>;

fn process_env(name: &str) -> Option<String> {
    std::env::var_os(name).map(|v| v.to_string_lossy().into_owned())
}

/// `settings.load()`, cached on success.
pub fn load() -> Result<&'static Settings, ConfigError> {
    #[cfg(test)]
    {
        Ok(crate::hooks::testing::ctx().settings)
    }
    #[cfg(not(test))]
    {
        if let Some(settings) = SETTINGS.get() {
            return Ok(settings);
        }
        let settings = load_with_env(&process_env)?;
        Ok(SETTINGS.get_or_init(|| settings))
    }
}

/// Settings after a successful [`load`] (see module docs); loads if needed and panics
/// on a config error.
pub fn get() -> &'static Settings {
    load().unwrap_or_else(|e| panic!("settings::get() before a successful load: {e}"))
}

/// `Path.home()`: `$HOME` if set (`/` when it is only slashes or empty), else the
/// passwd entry of the current user.
pub fn home_dir_with_env(env: Env<'_>) -> PathBuf {
    match env("HOME") {
        Some(home) => {
            let trimmed = home.trim_end_matches('/');
            PathBuf::from(if trimmed.is_empty() { "/" } else { trimmed })
        }
        None => PathBuf::from(crate::hooks::home_of(None).unwrap_or_else(|| "~".to_owned())),
    }
}

fn non_empty(env: Env<'_>, name: &str) -> Option<String> {
    env(name).filter(|v| !v.is_empty())
}

fn user_config_path_with_env(env: Env<'_>) -> PathBuf {
    let base = non_empty(env, "XDG_CONFIG_HOME")
        .map_or_else(|| home_dir_with_env(env).join(".config"), PathBuf::from);
    base.join("allegro-buddy").join("config.ini")
}

fn site_config_path_with_env(env: Env<'_>) -> PathBuf {
    PathBuf::from(
        non_empty(env, "AB_SITE_CONFIG")
            .unwrap_or_else(|| "/etc/allegro-buddy/config.ini".to_owned()),
    )
}

/// `user_config_path()`: `$XDG_CONFIG_HOME|~/.config` + `/allegro-buddy/config.ini`.
pub fn user_config_path() -> PathBuf {
    user_config_path_with_env(&process_env)
}

/// `site_config_path()`: `$AB_SITE_CONFIG` or `/etc/allegro-buddy/config.ini`.
pub fn site_config_path() -> PathBuf {
    site_config_path_with_env(&process_env)
}

/// `preset_names()`.
pub fn preset_names() -> Vec<&'static str> {
    PRESETS.iter().map(|(name, _)| *name).collect()
}

/// `preset_text(name)`.
pub fn preset_text(name: &str) -> Option<&'static str> {
    PRESETS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, text)| *text)
}

fn cache_dir_with_env(env: Env<'_>) -> PathBuf {
    let base = non_empty(env, "XDG_CACHE_HOME")
        .map_or_else(|| home_dir_with_env(env).join(".cache"), PathBuf::from);
    base.join("allegro-buddy")
}

fn cluster_name_cache_path_with_env(env: Env<'_>) -> PathBuf {
    cache_dir_with_env(env).join(cluster_name_cache_file(crate::hooks::hostname()))
}

/// `<cache dir>/<name>.<host>`: the directory `ab` caches the cluster name in, for a file
/// only `bab` uses.
pub fn bab_cache_path(name: &str) -> PathBuf {
    cache_dir_with_env(&process_env).join(format!(
        "{name}.{}",
        crate::hooks::hostname().replace('/', "_")
    ))
}

/// `cluster-name.<host>`, always one path component. Python joins the raw hostname, so a
/// (pathological) `/` in it would point outside the cache directory; bab writes `_`
/// instead. Any real hostname is unchanged.
fn cluster_name_cache_file(hostname: &str) -> String {
    format!("cluster-name.{}", hostname.replace('/', "_"))
}

/// `cluster_name_cache_path()`.
pub fn cluster_name_cache_path() -> PathBuf {
    cluster_name_cache_path_with_env(&process_env)
}

fn slurm_cluster_name_with_env(env: Env<'_>) -> Option<String> {
    let cache = cluster_name_cache_path_with_env(env);
    if let Ok(bytes) = std::fs::read(&cache) {
        let text = pyfmt::decode_text(bytes);
        let name = pyfmt::strip(&text);
        return (!name.is_empty()).then(|| name.to_owned());
    }
    let name = probe_cluster_name();
    if let Some(name) = &name {
        if let Some(parent) = cache.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&cache, format!("{name}\n"));
    }
    name
}

/// `slurm_cluster_name()`: Slurm's `ClusterName`, cached per host in a file. Unit tests
/// get `Ctx::cluster_name` instead (never touching the real cache).
pub fn slurm_cluster_name() -> Option<String> {
    #[cfg(test)]
    {
        crate::hooks::testing::ctx().cluster_name.map(str::to_owned)
    }
    #[cfg(not(test))]
    {
        slurm_cluster_name_with_env(&process_env)
    }
}

static CLUSTER_NAME_LINE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"(?m)^ClusterName\s*=\s*(\S+)").expect("ClusterName regex")
});

/// `probe_cluster_name()`.
pub fn probe_cluster_name() -> Option<String> {
    let output = crate::core::slurm::try_run(&["scontrol", "show", "config"], false).ok()?;
    if output.status != 0 {
        return None;
    }
    CLUSTER_NAME_LINE
        .captures(&output.stdout)
        .map(|c| c[1].to_owned())
}

/// `normalize_gpu_type("RTX-A5000")` -> `rtx_a5000`.
pub fn normalize_gpu_type(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut in_gap = false;
    for c in name.to_lowercase().chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            out.push(c);
            in_gap = false;
        } else if !in_gap {
            out.push('_');
            in_gap = true;
        }
    }
    out.trim_matches('_').to_owned()
}

fn allowed_keys(section: &str) -> Option<Option<&'static [&'static str]>> {
    SECTIONS
        .iter()
        .find(|(name, _)| *name == section)
        .map(|(_, keys)| *keys)
}

/// `normalize_key(section, key)`.
pub fn normalize_key(section: &str, key: &str) -> String {
    let key = pyfmt::strip(key).to_lowercase();
    match allowed_keys(section) {
        Some(Some(_)) => key.replace('-', "_"),
        _ => key,
    }
}

/// One config layer: ordered `((section, key), value)` pairs, keys unique (a Python dict).
pub type Layer = Vec<((String, String), String)>;

/// What `configparser` read: `[DEFAULT]` options and the sections, in file order; values
/// are the joined (multi-line) strings.
#[derive(Debug, Default)]
struct Ini {
    defaults: Vec<(String, String)>,
    sections: Vec<(String, Vec<(String, String)>)>,
}

impl Ini {
    /// `parser.items(section)`: DEFAULT's options first, overridden by the section's.
    fn items(&self, section: usize) -> Vec<(String, String)> {
        let mut items = self.defaults.clone();
        for (key, value) in &self.sections[section].1 {
            insert_ordered(&mut items, key.clone(), value.clone());
        }
        items
    }
}

/// Where the current option lives while reading.
#[derive(Clone, Copy, PartialEq)]
enum Target {
    Defaults,
    Section(usize),
}

/// Byte index of the first inline comment (`#`/`;` at the start or after whitespace),
/// found the way configparser does: both prefixes advance one occurrence per round and
/// the first round with a hit wins.
fn inline_comment_start(line: &str) -> Option<usize> {
    let mut cursors: Vec<(char, Option<usize>)> = vec![('#', None), (';', None)];
    loop {
        let mut best: Option<usize> = None;
        let mut next = Vec::new();
        for (prefix, previous) in &cursors {
            let from = previous.map_or(0, |i| i + 1);
            let Some(offset) = line.get(from..).and_then(|rest| rest.find(*prefix)) else {
                continue;
            };
            let index = from + offset;
            next.push((*prefix, Some(index)));
            let after_space = line[..index]
                .chars()
                .next_back()
                .is_some_and(pyfmt::is_space);
            if index == 0 || after_space {
                best = Some(best.map_or(index, |b| b.min(index)));
            }
        }
        if best.is_some() || next.is_empty() {
            return best;
        }
        cursors = next;
    }
}

/// `configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#', ';'))
/// .read_string(text, source=source)`; Err is `str(exc)`.
fn read_ini(text: &str, source: &str) -> Result<Ini, String> {
    let mut ini = Ini::default();
    let mut values: Vec<(Target, String, Vec<String>)> = Vec::new();
    let mut added_sections: Vec<String> = Vec::new();
    let mut added_options: Vec<(String, String)> = Vec::new();
    let mut cursect: Option<Target> = None;
    let mut sectname: Option<String> = None;
    let mut optname: Option<String> = None;
    let mut indent_level = 0usize;
    let mut parse_error: Option<String> = None;
    let source_repr = pyfmt::repr_str(source);
    let mut handle_error = |lineno: usize, line: &str| {
        let message = parse_error
            .get_or_insert_with(|| format!("Source contains parsing errors: {source_repr}"));
        message.push_str(&format!("\n\t[line {lineno:2}]: {}", pyfmt::repr_str(line)));
    };

    for (index, line) in text.split_inclusive('\n').enumerate() {
        let lineno = index + 1;
        let mut comment_start = inline_comment_start(line);
        let stripped_line = pyfmt::strip(line);
        if stripped_line.starts_with('#') || stripped_line.starts_with(';') {
            comment_start = Some(0);
        }
        let value = pyfmt::strip(&line[..comment_start.unwrap_or(line.len())]);
        let slot = |values: &[(Target, String, Vec<String>)],
                    target: Option<Target>,
                    name: Option<&str>| {
            let (target, name) = (target?, name.filter(|n| !n.is_empty())?);
            values
                .iter()
                .position(|(t, n, _)| *t == target && n == name)
        };
        if value.is_empty() {
            if comment_start.is_none()
                && let Some(i) = slot(&values, cursect, optname.as_deref())
            {
                values[i].2.push(String::new());
            }
            continue;
        }
        let cur_indent_level = line.chars().take_while(|c| pyfmt::is_space(*c)).count();
        let continuation = cursect.is_some()
            && optname.as_deref().is_some_and(|n| !n.is_empty())
            && cur_indent_level > indent_level;
        if continuation {
            if let Some(i) = slot(&values, cursect, optname.as_deref()) {
                values[i].2.push(value.to_owned());
            }
            continue;
        }
        indent_level = cur_indent_level;
        let header = value.strip_prefix('[').and_then(|rest| {
            rest.rfind(']')
                .filter(|&end| end > 0)
                .map(|end| &rest[..end])
        });
        if let Some(header) = header {
            if let Some(position) = ini.sections.iter().position(|(name, _)| name == header) {
                if added_sections.iter().any(|s| s == header) {
                    return Err(format!(
                        "While reading from {source_repr} [line {lineno:2}]: section {} already exists",
                        pyfmt::repr_str(header)
                    ));
                }
                cursect = Some(Target::Section(position));
                added_sections.push(header.to_owned());
            } else if header == "DEFAULT" {
                cursect = Some(Target::Defaults);
            } else {
                ini.sections.push((header.to_owned(), Vec::new()));
                cursect = Some(Target::Section(ini.sections.len() - 1));
                added_sections.push(header.to_owned());
            }
            sectname = Some(header.to_owned());
            optname = None;
            continue;
        }
        let Some(target) = cursect else {
            return Err(format!(
                "File contains no section headers.\nfile: {source_repr}, line: {lineno}\n{}",
                pyfmt::repr_str(line)
            ));
        };
        let Some(delimiter) = value.find(['=', ':']) else {
            handle_error(lineno, line);
            continue;
        };
        let option = value[..delimiter].trim_end_matches(pyfmt::is_space);
        if option.is_empty() {
            handle_error(lineno, line);
        }
        let name = option.to_lowercase();
        let section = sectname.clone().unwrap_or_default();
        if added_options
            .iter()
            .any(|(s, o)| *s == section && *o == name)
        {
            return Err(format!(
                "While reading from {source_repr} [line {lineno:2}]: option {} in section {} already exists",
                pyfmt::repr_str(&name),
                pyfmt::repr_str(&section)
            ));
        }
        added_options.push((section, name.clone()));
        let optval = pyfmt::strip(&value[delimiter + 1..]).to_owned();
        match values
            .iter_mut()
            .find(|(t, n, _)| *t == target && *n == name)
        {
            Some(slot) => slot.2 = vec![optval],
            None => values.push((target, name.clone(), vec![optval])),
        }
        optname = Some(name);
    }

    for (target, name, lines) in values {
        let joined = lines.join("\n");
        let joined = joined.trim_end_matches(pyfmt::is_space).to_owned();
        match target {
            Target::Defaults => insert_ordered(&mut ini.defaults, name, joined),
            Target::Section(i) => insert_ordered(&mut ini.sections[i].1, name, joined),
        }
    }
    match parse_error {
        Some(message) => Err(message),
        None => Ok(ini),
    }
}

/// `parse_layer(text, source)`.
pub fn parse_layer(text: &str, source: &str) -> Result<Layer, ConfigError> {
    let ini =
        read_ini(text, source).map_err(|e| ConfigError(format!("cannot parse {source}: {e}")))?;
    let mut sorted_sections: Vec<&str> = SECTIONS.iter().map(|(name, _)| *name).collect();
    sorted_sections.sort_unstable();
    let mut values = Layer::new();
    for (index, (section, _)) in ini.sections.iter().enumerate() {
        let Some(allowed) = allowed_keys(section) else {
            return Err(ConfigError(format!(
                "unknown section [{section}] in {source}; choose from {}",
                pyfmt::repr_list(&sorted_sections)
            )));
        };
        for (raw_key, value) in ini.items(index) {
            let key = normalize_key(section, &raw_key);
            if let Some(allowed) = allowed
                && !allowed.contains(&key.as_str())
            {
                let mut sorted: Vec<&str> = allowed.to_vec();
                sorted.sort_unstable();
                return Err(ConfigError(format!(
                    "unknown key {} in [{section}] of {source}; choose from {}",
                    pyfmt::repr_str(&raw_key),
                    pyfmt::repr_list(&sorted)
                )));
            }
            insert_ordered(
                &mut values,
                (section.clone(), key),
                pyfmt::strip(&value).to_owned(),
            );
        }
    }
    Ok(values)
}

/// `read_file_layer(path)`: missing file -> empty; other OSError -> `cannot read {path}:
/// {exc}` (Python `str(OSError)`, e.g. `[Errno 21] Is a directory: '/x'`).
pub fn read_file_layer(path: &Path) -> Result<Layer, ConfigError> {
    match std::fs::read(path) {
        Ok(bytes) => parse_layer(&pyfmt::decode_text(bytes), &path.to_string_lossy()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Layer::new()),
        Err(err) => Err(ConfigError(format!(
            "cannot read {}: {}",
            path.display(),
            pyfmt::os_error_str(&err, path)
        ))),
    }
}

fn layer_get<'a>(layer: &'a Layer, section: &str, key: &str) -> Option<&'a str> {
    layer
        .iter()
        .find(|((s, k), _)| s == section && k == key)
        .map(|(_, v)| v.as_str())
}

fn pick_preset_with_env(
    env: Env<'_>,
    file_choice: Option<&str>,
) -> Result<(Option<String>, String), ConfigError> {
    let env_choice = env("AB_CLUSTER");
    for (choice, why) in [
        (env_choice.as_deref(), "$AB_CLUSTER"),
        (file_choice, "[cluster] preset"),
    ] {
        let Some(choice) = choice.filter(|c| !c.is_empty()) else {
            continue;
        };
        if choice == "none" {
            return Ok((None, format!("disabled by {why}")));
        }
        if !preset_names().contains(&choice) {
            return Err(ConfigError(format!(
                "unknown cluster preset {} from {why}; choose from {} or none",
                pyfmt::repr_str(choice),
                pyfmt::repr_list(&preset_names())
            )));
        }
        return Ok((Some(choice.to_owned()), format!("set by {why}")));
    }
    Ok(match slurm_cluster_name_with_env(env) {
        None => (None, "Slurm did not report a ClusterName".to_owned()),
        Some(cluster) if preset_names().contains(&cluster.as_str()) => {
            let reason = format!("Slurm ClusterName is {cluster}");
            (Some(cluster), reason)
        }
        Some(cluster) => (
            None,
            format!("no preset ships for Slurm ClusterName {cluster}"),
        ),
    })
}

/// `pick_preset(file_choice)` -> (preset, reason).
pub fn pick_preset(file_choice: Option<&str>) -> Result<(Option<String>, String), ConfigError> {
    pick_preset_with_env(&process_env, file_choice)
}

/// Merge the layers (preset < site < user) like `load()`'s dict comprehension.
fn merge(
    preset: Option<String>,
    reason: String,
    site: Layer,
    user: Layer,
) -> Result<Settings, ConfigError> {
    let mut layers: Vec<(String, Layer)> = Vec::new();
    if let Some(name) = &preset {
        let source = format!("preset {name}");
        let text = preset_text(name).unwrap_or("");
        layers.push((source.clone(), parse_layer(text, &source)?));
    }
    layers.push(("site file".to_owned(), site));
    layers.push(("user file".to_owned(), user));
    let mut values: Vec<Entry> = Vec::new();
    for (source, layer) in layers {
        for ((section, key), value) in layer {
            match values
                .iter_mut()
                .find(|e| e.section == section && e.key == key)
            {
                Some(entry) => {
                    entry.value = value;
                    entry.source = source.clone();
                }
                None => values.push(Entry {
                    section,
                    key,
                    value,
                    source: source.clone(),
                }),
            }
        }
    }
    let settings = Settings::new(preset, reason, values);
    validate(&settings)?;
    Ok(settings)
}

/// `settings.load()` without the cache, reading the environment through `env`.
pub fn load_with_env(env: Env<'_>) -> Result<Settings, ConfigError> {
    let site = read_file_layer(&site_config_path_with_env(env))?;
    let user = read_file_layer(&user_config_path_with_env(env))?;
    let file_choice = layer_get(&user, "cluster", "preset")
        .filter(|v| !v.is_empty())
        .or_else(|| layer_get(&site, "cluster", "preset"))
        .map(str::to_owned);
    let (preset, reason) = pick_preset_with_env(env, file_choice.as_deref())?;
    merge(preset, reason, site, user)
}

/// `validate(settings)`.
pub fn validate(settings: &Settings) -> Result<(), ConfigError> {
    if let Some(login) = settings.get("cluster", "node_login")
        && !NODE_LOGIN_METHODS.contains(&login)
    {
        return Err(ConfigError(format!(
            "[cluster] node-login must be one of {}, got {}",
            pyfmt::repr_list(&NODE_LOGIN_METHODS),
            pyfmt::repr_str(login)
        )));
    }
    if let Some(limit) = settings.get("dev", "max_allocations") {
        let ok = !limit.is_empty()
            && limit.bytes().all(|b| b.is_ascii_digit())
            && !limit.trim_start_matches('0').is_empty();
        if !ok {
            return Err(ConfigError(format!(
                "[dev] max-allocations must be a whole number >= 1, got {}",
                pyfmt::repr_str(limit)
            )));
        }
    }
    Ok(())
}

/// Settings builders for unit tests.
#[cfg(test)]
pub mod testing {
    use super::*;

    /// A preset plus a user file text (the conftest setup: `AB_CLUSTER=<preset>`).
    pub fn with_preset(preset: &str, user_lines: &[&str]) -> Settings {
        let user = parse_layer(&user_lines.join("\n"), "user file").expect("user layer");
        merge(
            Some(preset.to_owned()),
            "set by $AB_CLUSTER".to_owned(),
            Layer::new(),
            user,
        )
        .expect("settings")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::slurm::fake::{FakeSlurm, fail};
    use crate::hooks::testing::Ctx;
    use std::collections::HashMap;

    /// A scratch directory under the target dir (never /tmp).
    fn scratch(name: &str) -> PathBuf {
        let exe = std::env::current_exe().expect("test exe");
        let dir = exe
            .parent()
            .and_then(Path::parent)
            .expect("target dir")
            .join("bab-test-scratch")
            .join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// The conftest isolation: config, cache and site file under `dir`, AB_CLUSTER=allegro.
    fn isolated(dir: &Path, extra: &[(&str, Option<&str>)]) -> HashMap<String, String> {
        let mut env = HashMap::from([
            (
                "XDG_CONFIG_HOME".to_owned(),
                dir.join("config").display().to_string(),
            ),
            (
                "XDG_CACHE_HOME".to_owned(),
                dir.join("cache").display().to_string(),
            ),
            (
                "AB_SITE_CONFIG".to_owned(),
                dir.join("site/config.ini").display().to_string(),
            ),
            ("AB_CLUSTER".to_owned(), "allegro".to_owned()),
            ("HOME".to_owned(), dir.join("home").display().to_string()),
        ]);
        for (name, value) in extra {
            match value {
                Some(v) => env.insert(name.to_string(), v.to_string()),
                None => env.remove(*name),
            };
        }
        env
    }

    fn load_in(env: &HashMap<String, String>) -> Result<Settings, ConfigError> {
        load_with_env(&|name| env.get(name).cloned())
    }

    fn write_user_config(dir: &Path, text: &str) {
        let path = dir.join("config/allegro-buddy/config.ini");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    fn layer(pairs: &[(&str, &str, &str)]) -> Layer {
        pairs
            .iter()
            .map(|(s, k, v)| ((s.to_string(), k.to_string()), v.to_string()))
            .collect()
    }

    // test_cli.py TestDevConfigDefaults
    #[test]
    fn config_presets_dev_start() {
        let dir = scratch("dev-start");
        write_user_config(
            &dir,
            "[dev]\ngpu = 2\ngpu-type = RTX_A5000\nmem = 32G\ntime = 04:00:00\n",
        );
        let settings = load_in(&isolated(&dir, &[])).unwrap();
        assert_eq!(settings.get("dev", "gpu"), Some("2"));
        assert_eq!(settings.get("dev", "gpu_type"), Some("RTX_A5000"));
        assert_eq!(settings.get("dev", "mem"), Some("32G"));
        assert_eq!(settings.get("dev", "time"), Some("04:00:00"));
        assert_eq!(settings.get("dev", "partition"), Some("gpu"));
        let sources: Vec<(&str, &str)> = settings
            .values
            .iter()
            .map(|e| (e.key.as_str(), e.source.as_str()))
            .collect();
        assert_eq!(
            sources,
            [
                ("partition", "preset allegro"),
                ("gpu", "user file"),
                ("a5000", "preset allegro"),
                ("gtx1080", "preset allegro"),
                ("gtx1080ti", "preset allegro"),
                ("rtx_a5000", "preset allegro"),
                ("gtx_1080ti", "preset allegro"),
                ("gpu_type", "user file"),
                ("mem", "user file"),
                ("time", "user file"),
            ]
        );
        assert_eq!(settings.preset.as_deref(), Some("allegro"));
        assert_eq!(settings.preset_reason, "set by $AB_CLUSTER");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn invalid_configs_are_rejected() {
        let cases = [
            (
                "[dev]\ngpus = 2\n",
                "unknown key 'gpus' in [dev] of {user}; choose from ['account', 'constraint', 'cpus', 'exclude', 'gpu', 'gpu_type', 'log_dir', 'max_allocations', 'mem', 'partition', 'time']",
            ),
            (
                "[cluster]\nnode-login = mosh\n",
                "[cluster] node-login must be one of ['ssh', 'srun'], got 'mosh'",
            ),
            (
                "[dev]\nmax-allocations = 0\n",
                "[dev] max-allocations must be a whole number >= 1, got '0'",
            ),
            (
                "[cluster]\npreset = nowhere\n",
                "unknown cluster preset 'nowhere' from [cluster] preset; choose from ['allegro', 'bhlrn4'] or none",
            ),
            (
                "[Dev]\ngpu = 1\n",
                "unknown section [Dev] in {user}; choose from ['cluster', 'dev', 'gpu-aliases', 'gpu-labels']",
            ),
        ];
        let dir = scratch("invalid");
        let user = dir
            .join("config/allegro-buddy/config.ini")
            .display()
            .to_string();
        for (text, message) in cases {
            write_user_config(&dir, text);
            let env = isolated(&dir, &[("AB_CLUSTER", None)]);
            let ctx = Ctx::default()
                .hostname("login1")
                .slurm(FakeSlurm::always(fail(1, "")))
                .install();
            let err = load_in(&env).unwrap_err();
            drop(ctx);
            assert_eq!(err.0, message.replace("{user}", &user), "{text}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preset_choice_order() {
        let env = |value: Option<&'static str>| {
            move |name: &str| {
                (name == "AB_CLUSTER")
                    .then_some(value)
                    .flatten()
                    .map(str::to_owned)
            }
        };
        assert_eq!(
            pick_preset_with_env(&env(Some("none")), Some("allegro")).unwrap(),
            (None, "disabled by $AB_CLUSTER".to_owned())
        );
        assert_eq!(
            pick_preset_with_env(&env(Some("")), Some("bhlrn4")).unwrap(),
            (
                Some("bhlrn4".to_owned()),
                "set by [cluster] preset".to_owned()
            )
        );
        assert_eq!(
            pick_preset_with_env(&env(Some("x'y")), None).unwrap_err().0,
            "unknown cluster preset \"x'y\" from $AB_CLUSTER; choose from ['allegro', 'bhlrn4'] or none"
        );
    }

    #[test]
    fn configparser_syntax() {
        let text = "# comment\n[dev] ; trailing\nPartition = gpu # inline\nmem: 8G\n\n[cluster]\ngpu-partitions = a,\n  b ;x\n\n  c\n\n[gpu-labels]\nRTX_A5000 = A5000;not a comment\n";
        assert_eq!(
            parse_layer(text, "f").unwrap(),
            layer(&[
                ("dev", "partition", "gpu"),
                ("dev", "mem", "8G"),
                ("cluster", "gpu_partitions", "a,\nb\n\nc"),
                ("gpu-labels", "rtx_a5000", "A5000;not a comment"),
            ])
        );
        let defaults = "[DEFAULT]\ncpus = 4\n[dev]\ngpu = 1\ncpus = 8\n[cluster]\n";
        let err = parse_layer(defaults, "f").unwrap_err();
        assert!(
            err.0.starts_with("unknown key 'cpus' in [cluster] of f;"),
            "{}",
            err.0
        );
        let layer_ok = parse_layer("[DEFAULT]\ncpus = 4\n[dev]\ngpu = 1\n", "f").unwrap();
        assert_eq!(
            layer_ok,
            layer(&[("dev", "cpus", "4"), ("dev", "gpu", "1")])
        );
    }

    #[test]
    fn configparser_errors() {
        let cases = [
            (
                "gpu = 1\n",
                "cannot parse f: File contains no section headers.\nfile: 'f', line: 1\n'gpu = 1\\n'",
            ),
            (
                "[dev]\n[dev]\n",
                "cannot parse f: While reading from 'f' [line  2]: section 'dev' already exists",
            ),
            (
                "[dev]\ngpu=1\nGPU=2\n",
                "cannot parse f: While reading from 'f' [line  3]: option 'gpu' in section 'dev' already exists",
            ),
            (
                "[dev]\njunk\n= x\ngpu = 1\n\tmore",
                "cannot parse f: Source contains parsing errors: 'f'\n\t[line  2]: 'junk\\n'\n\t[line  3]: '= x\\n'",
            ),
        ];
        for (text, message) in cases {
            assert_eq!(parse_layer(text, "f").unwrap_err().0, message, "{text:?}");
        }
    }

    #[test]
    fn gpu_tables_and_lists() {
        let settings = testing::with_preset(
            "bhlrn4",
            &[
                "[gpu-aliases]",
                "A-100 = NVIDIA A100",
                "[gpu-labels]",
                "h200 = Hopper",
                "MI-250 = MI250",
                "[cluster]",
                "gpu-partitions = a, b,,c ",
            ],
        );
        assert_eq!(settings.gpu_alias("a_100"), Some("nvidia_a100"));
        let labels: Vec<&str> = settings
            .gpu_labels
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(labels, ["a100", "1g_10gb", "h200", "intel", "mi_250"]);
        assert_eq!(settings.gpu_label("h200"), Some("Hopper"));
        assert_eq!(
            settings.get_list("cluster", "gpu_partitions"),
            ["a", "b", "c"]
        );
        assert!(settings.get_list("cluster", "missing").is_empty());
        assert_eq!(settings.section("dev").len(), 5);
    }
}
