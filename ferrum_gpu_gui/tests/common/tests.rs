use super::*;

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;

    /// A throwaway process group that is always killed, so a failing assertion cannot leave
    /// `sleep 60` helpers behind.
    struct Group(Child);
    impl Group {
        fn new(script: &str) -> Group {
            Group(
                Command::new("sh")
                    .args(["-c", script])
                    .process_group(0)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                    .expect("start sh"),
            )
        }
        fn pgid(&self) -> u32 {
            self.0.id()
        }
    }
    impl Drop for Group {
        fn drop(&mut self) {
            signal_group(self.0.id(), "KILL");
            let _ = self.0.wait();
        }
    }

    /// Regression: the group id used to be passed as a bare `-<pgid>`, which procps-ng's `kill`
    /// turned into `kill(-1, SIGTERM)` for any pid starting with 1 (the desktop session died).
    #[test]
    fn the_group_id_is_always_an_operand_after_a_double_dash() {
        // pids from real session crashes: all begin with the digit 1
        for pgid in [2u32, 7654321, 19207, 1149968, 1362056, 1908550] {
            for sig in ["TERM", "KILL", "0"] {
                let args = group_signal_args(pgid, sig).unwrap();
                assert_eq!(args, ["-s", sig, "--", &format!("-{pgid}")], "{pgid}");
            }
        }
        // group 0 (ourselves) and group 1 (as -1: everything we own) are never signalled
        for pgid in [0u32, 1] {
            assert_eq!(group_signal_args(pgid, "TERM"), None);
            assert!(!signal_group(pgid, "TERM"));
            assert!(!group_alive(pgid));
        }
        terminate_group(0);
        terminate_group(1);
    }

    #[test]
    fn terminating_a_group_ends_that_group_and_nothing_else() {
        let mut victim = Group::new("sleep 60 & wait"); // a leader with a helper, like Chrome
        let bystander = Group::new("sleep 60 & wait"); // someone else's group
        std::thread::sleep(Duration::from_millis(100));
        let pgid = victim.pgid();
        terminate_group(pgid);
        assert_eq!(victim.0.wait().unwrap().signal(), Some(15), "SIGTERM");
        let gone = wait_for(Duration::from_secs(5), || {
            (!group_alive(pgid)).then(String::new)
        });
        assert!(gone.is_some(), "the helper in the group was terminated too");
        assert!(
            group_alive(bystander.pgid()),
            "a bystander group is untouched"
        );
    }

    /// Hazard: helpers that ignore SIGTERM used to survive `drop`.
    #[test]
    fn reap_group_escalates_to_sigkill_for_leaders_and_helpers_that_ignore_term() {
        let mut stubborn = Group::new("trap '' TERM; sleep 60 & wait");
        let bystander = Group::new("sleep 60 & wait");
        std::thread::sleep(Duration::from_millis(200));
        let pgid = stubborn.pgid();
        let t0 = Instant::now();
        reap_group(
            &mut stubborn.0,
            Duration::from_millis(300),
            Duration::from_millis(300),
        );
        assert!(
            t0.elapsed() < Duration::from_secs(5),
            "bounded by the graces"
        );
        let gone = wait_for(Duration::from_secs(5), || {
            (!group_alive(pgid)).then(String::new)
        });
        assert!(gone.is_some(), "leader and helper are both gone");
        assert!(
            group_alive(bystander.pgid()),
            "a bystander group is untouched"
        );
    }

    #[test]
    fn reap_group_is_prompt_when_everything_obeys_sigterm() {
        let mut polite = Group::new("sleep 60 & wait");
        std::thread::sleep(Duration::from_millis(100));
        let t0 = Instant::now();
        reap_group(
            &mut polite.0,
            Duration::from_secs(3),
            Duration::from_secs(2),
        );
        assert!(
            t0.elapsed() < Duration::from_secs(2),
            "no grace is waited out"
        );
        assert!(!group_alive(polite.pgid()));
    }

    /// Hazard: a process whose parent died used to run on forever. `timeout` is the watchdog.
    #[test]
    fn with_deadline_ends_a_hung_program_and_its_helpers() {
        if !has_timeout() {
            eprintln!(
                "SKIP: no coreutils `timeout` here (macOS has none unless coreutils is installed)"
            );
            return;
        }
        let mut c = with_deadline("sh", 1);
        let mut child = c
            .args(["-c", "sleep 60 & wait"])
            .process_group(0)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("start");
        let pgid = child.id();
        let t0 = Instant::now();
        let status = child.wait().unwrap();
        assert!(
            t0.elapsed() < Duration::from_secs(8),
            "ended by the deadline"
        );
        assert_eq!(status.code(), Some(124), "timeout's own 'timed out' status");
        let gone = wait_for(Duration::from_secs(5), || {
            (!group_alive(pgid)).then(String::new)
        });
        assert!(gone.is_some(), "the helper died with it");
    }

    #[test]
    fn with_deadline_passes_the_program_through_unchanged_when_it_finishes() {
        let out = with_deadline("echo", 30).arg("hi").output().unwrap();
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hi");
    }
}
