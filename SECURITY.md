# Security

## Reporting a vulnerability

Use this repository's **Security → Report a vulnerability** page. Do not open a public issue.

Include the commit or build, macOS version, reproduction steps and observed impact. Remove keys, private paths, prompts and conversations from attachments.

## Scope

tokspur serves a local API protected by a generated key. Report problems such as:

- API access without the key, or another user reading private keys, settings or state
- adoption or configuration accepting changed, substituted or symlinked model files, or writing outside `.tokspur/` other than the three writes described below
- outbound requests from the server, or prompts and responses leaking into logs or shared state
- workers left running after `tokspur serve` stops, or the supervisor stopping unrelated processes
- an installer or update accepting a release that fails checksum, Developer ID signature or build checks

Three writes outside `.tokspur/` are deliberate. Adoption makes the derived pack read-only where it is stored, and when the adopted path is a symlink, such as a Hugging Face cache entry, the file it resolves to is the one made read-only. `tokspur pull` saves model files in the directory you name or, when `--state-root`, `TOKSPUR_STATE_ROOT` or an installed release sets the state root, in `models/` beside `.tokspur/`. With `--derived-pack-trust-mode local-seal`, a start that has to verify the derived pack in full writes `<stem>.seal` beside it once that verification passes, creating the seal or replacing an earlier one. The seal is a small read-only text file, written under a temporary name in the same directory and renamed into place.

Client coding agents choose which files to read and commands to run. Report problems in those choices to the client project.

## Supported versions

Fixes land on `main`. There are no maintained release branches yet.
