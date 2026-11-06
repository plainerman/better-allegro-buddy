//! Reservations. Mirrors `core.load_reservations`, `active_reservations`,
//! `recent_active_reservations`, `upcoming_reservations`,
//! `reservation_blocked_pending_jobs`, `reserved_nodes_map`, `user_can_use_reservation`.
//! Owned by P2-C.

use super::parse::{expand_nodelist, is_nullish, parse_prop_line};
use super::timefmt::{delta_micros, iso_to_dt};
use super::{Record, RecordExt};
use crate::error::Result;
use crate::hooks::GroupSet;
use crate::pyfmt;
use chrono::NaiveDateTime;
use std::collections::{BTreeSet, HashMap};

/// Active reservations older than this (by StartTime) are not displayed.
pub const RESERVATION_MAX_ACTIVE_DISPLAY_AGE_DAYS: i64 = 30;

/// `scontrol --oneliner show reservation`.
pub const ARGV: [&str; 4] = ["scontrol", "--oneliner", "show", "reservation"];

/// `load_reservations()`: `scontrol --oneliner show reservation` (unchecked; failure ->
/// `[]`), lines containing `ReservationName=`.
pub fn load_reservations() -> Result<Vec<Record>> {
    let output = super::slurm::run(&ARGV, false)?;
    if !output.success() {
        return Ok(Vec::new());
    }
    Ok(pyfmt::splitlines(&output.stdout)
        .into_iter()
        .filter(|line| !pyfmt::strip(line).is_empty() && line.contains("ReservationName="))
        .map(parse_prop_line)
        .collect())
}

/// `active_reservations(reservations)`: `State` upper == `ACTIVE`.
pub fn active_reservations(reservations: &[Record]) -> Vec<Record> {
    reservations
        .iter()
        .filter(|r| r.get_str("State").to_uppercase() == "ACTIVE")
        .cloned()
        .collect()
}

/// `recent_active_reservations(reservations, now=now)`.
pub fn recent_active_reservations(reservations: &[Record], now: NaiveDateTime) -> Vec<Record> {
    let max_age = RESERVATION_MAX_ACTIVE_DISPLAY_AGE_DAYS * 86_400 * 1_000_000;
    active_reservations(reservations)
        .into_iter()
        .filter(|r| match iso_to_dt(r.get_str("StartTime")) {
            None => true,
            Some(start) => delta_micros(now, start) <= max_age,
        })
        .collect()
}

/// `upcoming_reservations(reservations, now=now)`: INACTIVE and starting in the future.
pub fn upcoming_reservations(reservations: &[Record], now: NaiveDateTime) -> Vec<Record> {
    reservations
        .iter()
        .filter(|r| r.get_str("State").to_uppercase() == "INACTIVE")
        .filter(|r| iso_to_dt(r.get_str("StartTime")).is_some_and(|start| start > now))
        .cloned()
        .collect()
}

/// `reservation_blocked_pending_jobs(resvs, pending_jobs)`: the reason mentions
/// maintenance or a reservation name as a whole word (`re.split(r"[\s,():]+")`).
pub fn reservation_blocked_pending_jobs(resvs: &[Record], pending_jobs: &[Record]) -> Vec<Record> {
    let names: BTreeSet<String> = resvs
        .iter()
        .map(|r| r.get_str("ReservationName"))
        .filter(|name| !name.is_empty())
        .map(str::to_lowercase)
        .collect();
    pending_jobs
        .iter()
        .filter(|job| {
            let reason = job.get_str("Reason").to_lowercase();
            if reason.is_empty() {
                return false;
            }
            reason.contains("reserved for maintenance")
                || reason
                    .split(|c: char| pyfmt::is_space(c) || matches!(c, ',' | '(' | ')' | ':'))
                    .any(|word| names.contains(word))
        })
        .cloned()
        .collect()
}

/// `reserved_nodes_map(reservations)`: node -> reservation name (later reservations win).
pub fn reserved_nodes_map(reservations: &[Record]) -> HashMap<String, String> {
    let mut mapping = HashMap::new();
    for resv in reservations {
        let nodes = resv.get_str("Nodes");
        if is_nullish(Some(nodes)) {
            continue;
        }
        let name = resv.get_opt("ReservationName").unwrap_or("resv");
        for node in expand_nodelist(nodes) {
            mapping.insert(node, name.to_owned());
        }
    }
    mapping
}

/// `user_can_use_reservation(reservation, user, groups)`.
pub fn user_can_use_reservation(
    reservation: &Record,
    user: &str,
    groups: &(impl GroupSet + ?Sized),
) -> bool {
    let users = reservation.get_str("Users");
    if !is_nullish(Some(users)) {
        return users.split(',').any(|u| u == user);
    }
    let res_groups = reservation.get_str("Groups");
    if !is_nullish(Some(res_groups)) {
        return res_groups.split(',').any(|g| groups.contains(g));
    }
    false
}

#[cfg(test)]
mod tests {
    //! Port of the reservation parts of `tests/test_stats.py`.
    use super::*;

    fn rec(pairs: &[(&str, &str)]) -> Record {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn who_can_use_a_reservation() {
        let r = rec(&[("Users", "alice,bob,carol"), ("Groups", "")]);
        assert!(user_can_use_reservation(&r, "alice", &set(&[])));
        let r = rec(&[("Users", "alice,bob"), ("Groups", "")]);
        assert!(!user_can_use_reservation(&r, "carol", &set(&[])));
        let r = rec(&[("Users", ""), ("Groups", "researchers")]);
        assert!(user_can_use_reservation(
            &r,
            "alice",
            &set(&["researchers"])
        ));
        assert!(!user_can_use_reservation(&r, "alice", &set(&["students"])));
        let r = rec(&[("Users", "(null)"), ("Groups", "researchers")]);
        assert!(user_can_use_reservation(
            &r,
            "alice",
            &set(&["researchers"])
        ));
        let r = rec(&[("Users", "(null)"), ("Groups", "(null)")]);
        assert!(!user_can_use_reservation(&r, "alice", &set(&["everyone"])));
    }

    #[test]
    fn active() {
        assert_eq!(
            active_reservations(&[rec(&[("ReservationName", "maint1"), ("State", "ACTIVE")])])
                .len(),
            1
        );
        assert!(
            active_reservations(&[rec(&[("ReservationName", "future"), ("State", "INACTIVE")])])
                .is_empty()
        );
        assert_eq!(
            active_reservations(&[rec(&[("ReservationName", "x"), ("State", "active")])]).len(),
            1
        );
        let mixed = [
            rec(&[("ReservationName", "maint"), ("State", "ACTIVE")]),
            rec(&[("ReservationName", "future"), ("State", "INACTIVE")]),
        ];
        let result = active_reservations(&mixed);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["ReservationName"], "maint");
    }

    #[test]
    fn blocked_pending_jobs() {
        let jobs = [rec(&[
            ("JobID", "101"),
            ("Reason", "reserved for maintenance"),
        ])];
        assert_eq!(reservation_blocked_pending_jobs(&[], &jobs).len(), 1);
        let jobs = [rec(&[
            ("JobID", "102"),
            ("Reason", "ReqNodeNotAvail,maint2025"),
        ])];
        assert_eq!(
            reservation_blocked_pending_jobs(&[rec(&[("ReservationName", "maint2025")])], &jobs)
                .len(),
            1
        );
        let jobs = [rec(&[("JobID", "103"), ("Reason", "Resources")])];
        assert!(
            reservation_blocked_pending_jobs(&[rec(&[("ReservationName", "maintenance")])], &jobs)
                .is_empty()
        );
        assert!(reservation_blocked_pending_jobs(&[], &[]).is_empty());
        for (name, reason) in [
            ("res", "Resources"),
            ("gpu", "ReqNodeNotAvail, UnavailableNodes:gpu[101-102]"),
        ] {
            let jobs = [rec(&[("JobID", "104"), ("Reason", reason)])];
            assert!(
                reservation_blocked_pending_jobs(&[rec(&[("ReservationName", name)])], &jobs)
                    .is_empty()
            );
        }
    }
}
