//! Who is running bab. Mirrors `core.current_user`, `core.current_groups`,
//! `core.user_accounts`. Owned by P2-C.

use super::Record;
use crate::error::Result;
use std::collections::BTreeSet;

/// `current_user()` = `getpass.getuser()` via [`crate::hooks::user`].
pub fn current_user() -> &'static str {
    crate::hooks::user()
}

/// `current_groups()` via [`crate::hooks::groups`].
pub fn current_groups() -> &'static dyn crate::hooks::GroupSet {
    crate::hooks::groups()
}

/// `user_accounts(user)`: `sacctmgr -nP show user <user> withassoc [cluster=<name>]
/// format=DefaultAccount,Account,QOS`, unchecked, rows with keys `DefaultAccount`,
/// `Account`, `QOS`. The cluster name comes from `settings::slurm_cluster_name()` (cached
/// file / `scontrol show config`).
pub fn user_accounts(user: &str) -> Result<Vec<Record>> {
    accounts_from(&accounts_argv(user))
}

/// [`user_accounts`] with its argv already built (building it may probe Slurm).
pub fn accounts_from(argv: &[String]) -> Result<Vec<Record>> {
    super::slurm::run_parsable_rows(argv, &["DefaultAccount", "Account", "QOS"], false)
}

/// The `sacctmgr` argv of [`user_accounts`] (may probe the cluster name).
pub fn accounts_argv(user: &str) -> Vec<String> {
    let mut argv = vec![
        "sacctmgr".to_owned(),
        "-nP".to_owned(),
        "show".to_owned(),
        "user".to_owned(),
        user.to_owned(),
        "withassoc".to_owned(),
    ];
    if let Some(cluster) = crate::settings::slurm_cluster_name() {
        argv.push(format!("cluster={cluster}"));
    }
    argv.push("format=DefaultAccount,Account,QOS".to_owned());
    argv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::slurm::fake::{FakeSlurm, fail, ok};
    use crate::hooks::testing::Ctx;

    #[test]
    fn associations_for_the_cluster() {
        let fake = FakeSlurm::always(ok("proj|proj|normal\n|other|long\n"));
        let _ctx = Ctx::default().slurm(fake.clone()).install();
        let rows = user_accounts("alice").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["Account"], "other");
        assert_eq!(rows[1]["DefaultAccount"], "");
        assert_eq!(
            fake.calls()[0],
            [
                "sacctmgr",
                "-nP",
                "show",
                "user",
                "alice",
                "withassoc",
                "cluster=allegro",
                "format=DefaultAccount,Account,QOS"
            ]
        );
        let fake = FakeSlurm::always(fail(1, "down"));
        let _ctx = Ctx::default()
            .cluster_name(None)
            .slurm(fake.clone())
            .install();
        assert!(user_accounts("bob").unwrap().is_empty());
        assert_eq!(fake.calls()[0][6], "format=DefaultAccount,Account,QOS");
    }
}
