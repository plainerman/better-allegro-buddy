//! Completion with a broken config: click swallows exceptions from parameter callbacks
//! during resilient parsing, so `-G` (normalised through the settings' GPU aliases) must
//! not crash completion. Expected output recorded from the Python reference.

use std::process::Command;

#[test]
fn gpu_type_completion_survives_an_invalid_config() {
    let home = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("completion-config");
    std::fs::create_dir_all(&home).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_bab"))
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", &home)
        .env("AB_CLUSTER", "not-a-preset")
        .env("_BAB_COMPLETE", "bash_complete")
        .env("COMP_WORDS", "bab free -G a100 --")
        .env("COMP_CWORD", "4")
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&output.stderr), "");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "plain,--kind\nplain,--nodes\nplain,--partition\nplain,--watch\nplain,--help\n"
    );
    assert_eq!(output.status.code(), Some(0));
}
