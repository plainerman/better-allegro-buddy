//! Test hooks: the only sanctioned way to read the wall clock, the user, the user's groups,
//! the hostname and whether stdin is a terminal.
//!
//! With the cargo feature `test-hooks` (`tests/snapshots.rs`; never in a normal build) each
//! hook reads a `BAB_TEST_*` environment variable; when it is unset the real value is used,
//! exactly as Python gets it. Without the feature the variables are never read. The
//! differential harness (git tag `equivalence-harness-3`) patched the Python reference the
//! same way, so both sides saw the same clock, groups and host.
//!
//! | hook            | env var                | real behaviour (Python)                          |
//! |-----------------|------------------------|--------------------------------------------------|
//! | [`now`]         | `BAB_TEST_NOW`         | `datetime.now()` / `time.strftime` (local, naive) |
//! | [`groups`]      | `BAB_TEST_GROUPS`      | `{grp.getgrgid(g).gr_name for g in os.getgroups()}` (unknown gids skipped) |
//! | [`interactive`] | `BAB_TEST_INTERACTIVE` | `sys.stdin.isatty()`                              |
//! | [`hostname`]    | `BAB_TEST_HOSTNAME`    | `socket.gethostname()`                            |
//! | [`user`]        | (none; set `LOGNAME`/`USER`) | `getpass.getuser()`                         |
//!
//! Formats (as the reference wrapper reads them): `BAB_TEST_NOW` is an ISO local naive
//! datetime (`2026-05-04T12:00:00`; anything `datetime.fromisoformat` accepts), used when
//! non-empty; `BAB_TEST_GROUPS` is a comma list used whenever it is SET (`""` = no
//! groups, empty entries ignored); `BAB_TEST_INTERACTIVE` set: interactive iff `1`;
//! `BAB_TEST_HOSTNAME` set: the name verbatim.
//!
//! Elapsed time and sleeping (dev wait polling, `--watch`) stay real: use
//! `std::time::Instant` and [`crate::signals::sleep`].
//!
//! Values that cannot change during a run (user, groups, hostname) are cached per process;
//! the real groups are only looked up once a group name has to be checked.
//!
//! Unit tests (`cfg(test)`) replace all of this with a per-thread `testing::Ctx`
//! (frozen clock, user, groups, fake Slurm, settings, node facts) that
//! [`crate::core::slurm::parallel`] hands on to its worker threads.

use chrono::NaiveDateTime;
use std::collections::BTreeSet;
use std::sync::OnceLock;

fn env_set(name: &str) -> Option<String> {
    std::env::var_os(name).map(|v| v.to_string_lossy().into_owned())
}

/// A `BAB_TEST_*` hook variable: always `None` without the `test-hooks` feature.
#[cfg(feature = "test-hooks")]
fn test_hook(name: &str) -> Option<String> {
    env_set(name)
}

#[cfg(not(feature = "test-hooks"))]
#[inline(always)]
fn test_hook(_name: &str) -> Option<String> {
    None
}

/// Local wall-clock time as a naive datetime (Python `datetime.now()`, microsecond
/// resolution), or the frozen `BAB_TEST_NOW`. Not cached: `--watch` shows the time of each
/// refresh.
pub fn now() -> NaiveDateTime {
    #[cfg(test)]
    if let Some(now) = testing::ctx().now {
        return now;
    }
    if let Some(frozen) = test_hook("BAB_TEST_NOW").filter(|v| !v.is_empty())
        && let Some(dt) = crate::core::timefmt::iso_to_dt(&frozen)
    {
        return dt;
    }
    let now = chrono::Local::now().naive_local();
    use chrono::Timelike;
    now.with_nanosecond(now.nanosecond() / 1000 * 1000)
        .unwrap_or(now)
}

/// Python `getpass.getuser()`: first non-empty of `LOGNAME`, `USER`, `LNAME`, `USERNAME`,
/// else the passwd name of the real uid (`pwd.getpwuid(os.getuid())[0]`). Cached.
pub fn user() -> &'static str {
    #[cfg(test)]
    if let Some(user) = testing::ctx().user {
        return user;
    }
    static USER: OnceLock<String> = OnceLock::new();
    USER.get_or_init(|| {
        ["LOGNAME", "USER", "LNAME", "USERNAME"]
            .iter()
            .find_map(|name| env_set(name).filter(|v| !v.is_empty()))
            .unwrap_or_else(passwd_name)
    })
}

fn passwd_name() -> String {
    passwd_entry(None)
        .map(|(name, _)| name)
        .unwrap_or_else(|| unsafe { libc::getuid() }.to_string())
}

/// `(pw_name, pw_dir)` of `user` (`pwd.getpwnam`) or of the real uid (`pwd.getpwuid`).
fn passwd_entry(user: Option<&str>) -> Option<(String, String)> {
    let c_user = match user {
        Some(name) => Some(std::ffi::CString::new(name).ok()?),
        None => None,
    };
    let mut buf = vec![0 as libc::c_char; 4096];
    loop {
        let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
        let mut result: *mut libc::passwd = std::ptr::null_mut();
        let rc = unsafe {
            match &c_user {
                Some(name) => libc::getpwnam_r(
                    name.as_ptr(),
                    &mut pwd,
                    buf.as_mut_ptr(),
                    buf.len(),
                    &mut result,
                ),
                None => libc::getpwuid_r(
                    libc::getuid(),
                    &mut pwd,
                    buf.as_mut_ptr(),
                    buf.len(),
                    &mut result,
                ),
            }
        };
        if rc == libc::ERANGE && buf.len() < 1 << 20 {
            buf.resize(buf.len() * 2, 0);
            continue;
        }
        if rc != 0 || result.is_null() {
            return None;
        }
        let text = |ptr: *const libc::c_char| {
            unsafe { std::ffi::CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        };
        return Some((text(pwd.pw_name), text(pwd.pw_dir)));
    }
}

/// Home directory from the passwd database: of `user` (`~user`), or of the real uid
/// (`Path.home()` without `$HOME`). Not a test hook, just the OS lookup.
pub fn home_of(user: Option<&str>) -> Option<String> {
    passwd_entry(user).map(|(_, dir)| dir)
}

/// Membership in the groups of this process (`core.current_groups()`).
pub trait GroupSet {
    fn contains(&self, name: &str) -> bool;
}

impl GroupSet for BTreeSet<String> {
    fn contains(&self, name: &str) -> bool {
        BTreeSet::contains(self, name)
    }
}

/// The real groups, resolved on the first membership question only. With directory
/// services (sssd, LDAP) `getgrgid` fetches every member of the group, and with a cold
/// cache that takes seconds per big group; most partitions say `AllowGroups=ALL` and never
/// ask. Same set as Python's, just computed later.
struct SystemGroups(OnceLock<BTreeSet<String>>);

impl GroupSet for SystemGroups {
    fn contains(&self, name: &str) -> bool {
        self.0.get_or_init(system_groups).contains(name)
    }
}

/// The groups of this process (`core.current_groups()`), or `BAB_TEST_GROUPS`.
/// Uses `getgroups(2)` + `getgrgid_r(3)` (NSS, hence glibc and not musl), lazily. Cached.
pub fn groups() -> &'static dyn GroupSet {
    #[cfg(test)]
    if let Some(groups) = testing::ctx().groups {
        return groups;
    }
    static LISTED: OnceLock<Option<BTreeSet<String>>> = OnceLock::new();
    static SYSTEM: SystemGroups = SystemGroups(OnceLock::new());
    let listed = LISTED.get_or_init(|| {
        test_hook("BAB_TEST_GROUPS").map(|list| {
            list.split(',')
                .filter(|g| !g.is_empty())
                .map(str::to_owned)
                .collect()
        })
    });
    match listed {
        Some(set) => set,
        None => &SYSTEM,
    }
}

fn system_groups() -> BTreeSet<String> {
    let count = unsafe { libc::getgroups(0, std::ptr::null_mut()) };
    if count <= 0 {
        return BTreeSet::new();
    }
    let mut gids = vec![0 as libc::gid_t; count as usize];
    let count = unsafe { libc::getgroups(count, gids.as_mut_ptr()) };
    if count < 0 {
        return BTreeSet::new();
    }
    gids.truncate(count as usize);
    gids.into_iter().filter_map(group_name).collect()
}

fn group_name(gid: libc::gid_t) -> Option<String> {
    let mut buf = vec![0 as libc::c_char; 4096];
    loop {
        let mut grp: libc::group = unsafe { std::mem::zeroed() };
        let mut result: *mut libc::group = std::ptr::null_mut();
        let rc =
            unsafe { libc::getgrgid_r(gid, &mut grp, buf.as_mut_ptr(), buf.len(), &mut result) };
        if rc == libc::ERANGE && buf.len() < 1 << 24 {
            buf.resize(buf.len() * 2, 0);
            continue;
        }
        if rc != 0 || result.is_null() {
            return None;
        }
        return Some(
            unsafe { std::ffi::CStr::from_ptr(grp.gr_name) }
                .to_string_lossy()
                .into_owned(),
        );
    }
}

/// `sys.stdin.isatty()`, or `BAB_TEST_INTERACTIVE` (`1` true, anything else false).
pub fn interactive() -> bool {
    #[cfg(test)]
    if let Some(interactive) = testing::ctx().interactive {
        return interactive;
    }
    match test_hook("BAB_TEST_INTERACTIVE") {
        Some(flag) => flag == "1",
        None => unsafe { libc::isatty(0) == 1 },
    }
}

/// `socket.gethostname()`, or `BAB_TEST_HOSTNAME`. Cached.
pub fn hostname() -> &'static str {
    #[cfg(test)]
    if let Some(hostname) = testing::ctx().hostname {
        return hostname;
    }
    static HOSTNAME: OnceLock<String> = OnceLock::new();
    HOSTNAME.get_or_init(|| test_hook("BAB_TEST_HOSTNAME").unwrap_or_else(system_hostname))
}

fn system_hostname() -> String {
    let mut buf = [0 as libc::c_char; 256];
    if unsafe { libc::gethostname(buf.as_mut_ptr(), buf.len()) } != 0 {
        return String::new();
    }
    buf[buf.len() - 1] = 0;
    unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Per-thread replacements for the hooks and the process-wide state bab keeps (settings,
/// node facts, Slurm), so unit tests run in parallel without touching the environment.
#[cfg(test)]
pub mod testing {
    use crate::core::nodes::NodeFacts;
    use crate::core::slurm::fake::FakeSlurm;
    use crate::settings::Settings;
    use chrono::NaiveDateTime;
    use std::cell::RefCell;
    use std::collections::BTreeSet;
    use std::sync::{Arc, OnceLock, RwLock};

    /// What a test pins down; `None` = real behaviour.
    #[derive(Clone)]
    pub struct Ctx {
        pub now: Option<NaiveDateTime>,
        pub user: Option<&'static str>,
        pub groups: Option<&'static BTreeSet<String>>,
        pub interactive: Option<bool>,
        pub hostname: Option<&'static str>,
        /// The Slurm every command goes to; `None` panics (tests must stay hermetic).
        pub slurm: Option<Arc<FakeSlurm>>,
        /// `settings::slurm_cluster_name()` (default `allegro`).
        pub cluster_name: Option<&'static str>,
        /// `settings::load()`; defaults to the allegro preset alone (like conftest.py).
        pub settings: &'static Settings,
        /// `_NODE_FACTS`, shared with worker threads of `slurm::parallel`.
        pub node_facts: Arc<RwLock<Option<Arc<NodeFacts>>>>,
        /// `slurm::prefetch`ed queries of this test.
        pub prefetched: Arc<std::sync::Mutex<crate::core::slurm::PrefetchedForTests>>,
        /// The Slurm version `core::fast` sees (`None`: unknown, no fast queries).
        pub slurm_version: Option<(u32, u32)>,
    }

    fn allegro_settings() -> &'static Settings {
        static ALLEGRO: OnceLock<Settings> = OnceLock::new();
        ALLEGRO.get_or_init(|| crate::settings::testing::with_preset("allegro", &[]))
    }

    impl Default for Ctx {
        fn default() -> Ctx {
            Ctx {
                now: None,
                user: Some("alice"),
                groups: None,
                interactive: None,
                hostname: None,
                slurm: None,
                cluster_name: Some("allegro"),
                settings: allegro_settings(),
                node_facts: Arc::default(),
                prefetched: Arc::default(),
                slurm_version: None,
            }
        }
    }

    impl Ctx {
        pub fn now(mut self, iso: &str) -> Ctx {
            self.now = Some(crate::core::timefmt::iso_to_dt(iso).expect("ISO datetime"));
            self
        }

        pub fn user(mut self, user: &str) -> Ctx {
            self.user = Some(Box::leak(user.to_owned().into_boxed_str()));
            self
        }

        pub fn groups(mut self, groups: &[&str]) -> Ctx {
            let set: BTreeSet<String> = groups.iter().map(|g| g.to_string()).collect();
            self.groups = Some(Box::leak(Box::new(set)));
            self
        }

        pub fn hostname(mut self, hostname: &str) -> Ctx {
            self.hostname = Some(Box::leak(hostname.to_owned().into_boxed_str()));
            self
        }

        pub fn cluster_name(mut self, name: Option<&'static str>) -> Ctx {
            self.cluster_name = name;
            self
        }

        pub fn slurm_version(mut self, version: (u32, u32)) -> Ctx {
            self.slurm_version = Some(version);
            self
        }

        pub fn slurm(mut self, slurm: Arc<FakeSlurm>) -> Ctx {
            self.slurm = Some(slurm);
            self
        }

        pub fn settings(mut self, settings: Settings) -> Ctx {
            self.settings = Box::leak(Box::new(settings));
            self
        }

        /// Make this the current thread's context until the guard drops.
        pub fn install(self) -> Guard {
            let previous = CTX.with(|c| c.replace(Some(Arc::new(self))));
            Guard { previous }
        }
    }

    thread_local! {
        static CTX: RefCell<Option<Arc<Ctx>>> = const { RefCell::new(None) };
    }

    /// Restores the previous context on drop.
    pub struct Guard {
        previous: Option<Arc<Ctx>>,
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            let previous = self.previous.take();
            CTX.with(|c| *c.borrow_mut() = previous);
        }
    }

    /// The current thread's context (a default one is created on first use).
    pub fn ctx() -> Arc<Ctx> {
        CTX.with(|c| c.borrow_mut().get_or_insert_with(Default::default).clone())
    }

    /// Run `f` with `ctx` installed (worker threads of `slurm::parallel`).
    pub fn with_ctx<T>(ctx: Arc<Ctx>, f: impl FnOnce() -> T) -> T {
        let previous = CTX.with(|c| c.replace(Some(ctx)));
        let _guard = Guard { previous };
        f()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_overrides_hooks() {
        let _ctx = testing::Ctx::default()
            .now("2026-10-06T14:30:00")
            .user("bob")
            .groups(&["staff", "gpu"])
            .hostname("login1")
            .install();
        assert_eq!(now().to_string(), "2026-10-06 14:30:00");
        assert_eq!(user(), "bob");
        assert!(groups().contains("gpu") && groups().contains("staff"));
        assert!(!groups().contains("users"));
        assert_eq!(hostname(), "login1");
    }

    #[test]
    fn real_now_has_microsecond_resolution() {
        use chrono::Timelike;
        assert_eq!(now().nanosecond() % 1000, 0);
    }
}
