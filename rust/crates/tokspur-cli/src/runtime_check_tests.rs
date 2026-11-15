use super::{Command, parse_args};
use crate::serve::{SpeculativeAsset, build_carrier_flags};
use std::path::Path;

#[test]
fn retired_prefill_controls_refuse_before_launch() {
    for args in [
        vec!["serve", "--gdn-prefill-only"],
        vec!["serve", "--native-prefill-router-fast", "on"],
        vec![
            "serve",
            "--speed-preset",
            "--native-prefill-router-fast",
            "on",
        ],
    ] {
        let Command::Serve(opts) = parse_args(args).unwrap().command else {
            panic!("serve");
        };
        assert!(
            build_carrier_flags(
                &opts,
                SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
                Path::new("/tmp/state")
            )
            .is_err()
        );
    }
}

#[test]
fn runtime_controls_are_absent_by_default_and_off_is_not_forwarded() {
    for args in [
        vec!["serve"],
        vec![
            "serve",
            "--native-prepared-validation",
            "off",
            "--native-dflash2-draft-policy",
            "always",
            "--worker-burst-pacing",
            "off",
        ],
    ] {
        let Command::Serve(opts) = parse_args(args).unwrap().command else {
            panic!("serve");
        };
        let flags = build_carrier_flags(
            &opts,
            SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
            Path::new("/tmp/state"),
        )
        .unwrap();
        for flag in [
            "--native-prepared-validation",
            "--native-verify-cost-profile",
            "--native-dflash2-draft-policy",
            "--native-verify-m8-splits",
            "--worker-burst-pacing",
        ] {
            assert!(!flags.iter().any(|f| f == flag));
        }
    }
}

#[test]
fn worker_burst_pacing_is_forwarded_only_when_enabled() {
    let Command::Serve(opts) = parse_args(["serve", "--worker-burst-pacing", "on"])
        .unwrap()
        .command
    else {
        panic!("serve");
    };
    let flags = build_carrier_flags(
        &opts,
        SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
        Path::new("/tmp/state"),
    )
    .unwrap();
    assert!(
        flags
            .windows(2)
            .any(|pair| pair == ["--worker-burst-pacing", "on"])
    );
    assert!(parse_args(["serve", "--worker-burst-pacing", "auto"]).is_err());
}

#[test]
fn prepared_validation_is_forwarded_only_when_enabled() {
    let Command::Serve(opts) = parse_args(["serve", "--native-prepared-validation", "on"])
        .unwrap()
        .command
    else {
        panic!("serve");
    };
    let flags = build_carrier_flags(
        &opts,
        SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
        Path::new("/tmp/state"),
    )
    .unwrap();
    assert!(
        flags
            .windows(2)
            .any(|pair| pair == ["--native-prepared-validation", "on"])
    );
    assert!(parse_args(["serve", "--native-prepared-validation", "auto"]).is_err());
}

#[test]
fn drafting_and_numerical_controls_refuse_missing_prerequisites() {
    for args in [
        vec!["serve", "--native-dflash2-draft-policy", "auto"],
        vec!["serve", "--native-verify-m8-splits", "/tmp/splits.json"],
    ] {
        let Command::Serve(opts) = parse_args(args).unwrap().command else {
            panic!("serve");
        };
        assert!(
            build_carrier_flags(
                &opts,
                SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
                Path::new("/tmp/state")
            )
            .is_err()
        );
    }
}

#[test]
fn controller_dflash2_explicit_off_survives_preset_expansion() {
    let Command::Serve(opts) = parse_args([
        "serve",
        "--speed-preset",
        "--drafter",
        "dflash2",
        "--dflash2-dir",
        "/tmp/dflash2",
        "--no-verify-nax",
        "--native-verify-hc-gdn-fused",
        "off",
    ])
    .unwrap()
    .command
    else {
        panic!("serve");
    };
    let flags = build_carrier_flags(
        &opts,
        SpeculativeAsset::DFlash2Dir(Path::new("/tmp/dflash2")),
        Path::new("/tmp/state"),
    )
    .unwrap();
    assert!(
        flags
            .windows(2)
            .any(|p| p == ["--native-verify-nax", "off"])
    );
    assert!(!flags.iter().any(|f| f == "--native-verify-nax-m8"));
    assert!(
        flags
            .windows(2)
            .any(|p| p == ["--native-prefill-dense", "nax"])
    );
}

#[test]
fn controller_mtp_draft_slice_explicit_off_survives_preset_expansion() {
    let Command::Serve(opts) = parse_args(["serve", "--speed-preset", "--no-mtp-draft-slice"])
        .unwrap()
        .command
    else {
        panic!("serve");
    };
    assert!(!opts.mtp_draft_slice);
    assert_eq!(opts.mtp_depth, Some(3));
    let flags = build_carrier_flags(
        &opts,
        SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
        Path::new("/tmp/state"),
    )
    .unwrap();
    assert!(
        flags
            .windows(2)
            .any(|p| p == ["--native-mtp-draft-slice", "off"])
    );
    assert!(flags.windows(2).any(|p| p == ["--native-mtp-depth", "3"]));
    assert!(flags.windows(2).any(|p| p == ["--native-verify-nax", "on"]));
    assert!(
        flags
            .windows(2)
            .any(|p| p == ["--native-verify-nax-m8", "on"])
    );
    assert!(
        flags
            .windows(2)
            .any(|p| p == ["--native-mtp-fast-round", "on"])
    );
}

#[test]
fn controller_mtp_preset_explicit_off_survives_preset_expansion() {
    let Command::Serve(opts) = parse_args([
        "serve",
        "--speed-preset",
        "--no-verify-nax-m8",
        "--native-mtp-fast-round",
        "off",
    ])
    .unwrap()
    .command
    else {
        panic!("serve");
    };
    let flags = build_carrier_flags(
        &opts,
        SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
        Path::new("/tmp/state"),
    )
    .unwrap();
    assert!(flags.windows(2).any(|p| p == ["--native-verify-nax", "on"]));
    assert!(
        flags
            .windows(2)
            .any(|p| p == ["--native-verify-nax-m8", "off"])
    );
    assert!(
        flags
            .windows(2)
            .any(|p| p == ["--native-mtp-fast-round", "off"])
    );
}

#[test]
fn controller_mtp_draft_slice_selection_is_refused_for_dflash2_with_the_flag_it_names() {
    let refusal = |extra: &[&str]| {
        let mut args = vec![
            "serve",
            "--speed-preset",
            "--drafter",
            "dflash2",
            "--dflash2-dir",
            "/tmp/dflash2",
        ];
        args.extend_from_slice(extra);
        let Command::Serve(opts) = parse_args(args).unwrap().command else {
            panic!("serve");
        };
        build_carrier_flags(
            &opts,
            SpeculativeAsset::DFlash2Dir(Path::new("/tmp/dflash2")),
            Path::new("/tmp/state"),
        )
        .unwrap_err()
        .to_string()
    };
    for on in [
        &["--mtp-draft-slice"][..],
        &["--native-mtp-draft-slice", "on"],
        &["--no-mtp-draft-slice", "--mtp-draft-slice"],
    ] {
        assert_eq!(
            refusal(on),
            "--mtp-draft-slice is incompatible with --drafter dflash2"
        );
    }
    for off in [
        &["--no-mtp-draft-slice"][..],
        &["--native-mtp-draft-slice", "off"],
        &["--mtp-draft-slice", "--no-mtp-draft-slice"],
    ] {
        let message = refusal(off);
        assert!(
            message.starts_with("--no-mtp-draft-slice and --native-mtp-draft-slice do not apply")
                && message.contains("--dflash2-vocab-slice 0"),
            "unexpected refusal for {off:?}: {message}"
        );
    }
    // The DFlash2 setting the message points at is accepted and forwarded.
    let Command::Serve(full) = parse_args([
        "serve",
        "--speed-preset",
        "--drafter",
        "dflash2",
        "--dflash2-dir",
        "/tmp/dflash2",
        "--dflash2-vocab-slice",
        "0",
    ])
    .unwrap()
    .command
    else {
        panic!("serve");
    };
    assert_eq!(full.dflash2_vocab_slice, Some(0));
    assert!(
        build_carrier_flags(
            &full,
            SpeculativeAsset::DFlash2Dir(Path::new("/tmp/dflash2")),
            Path::new("/tmp/state"),
        )
        .is_ok()
    );
}
