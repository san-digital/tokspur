# Rust serving, protocol, derived-pack and CLI build targets. GPU execution and Metal command encoding stay in Objective-C++ and C++. Outputs stay under build/rust-*.
#
#   make -C native -f rust.mk rust-check          fmt, clippy (every feature level), tests
#   make -C native -f rust.mk rust-release        release tokspur-server (fast-math) and tokspur CLI
#   make -C native -f rust.mk rust-server         release server, FEATURES=fast-math by default
#   make -C native -f rust.mk rust-server FEATURES=   ordinary (no private wire) server
#   make -C native -f rust.mk rust-diff           byte-level differential against the C++ server
#   make -C native -f rust.mk rust-supply         cargo-audit, cargo-deny, cargo-vet and the OSV check
#   make -C native -f rust.mk rust-fuzz           build, seed and run every libFuzzer target
#   make -C native -f rust.mk rust-fuzz FUZZ_SECONDS=300 FUZZ_TARGETS="json_diff gguf_diff"
#
# The toolchain is pinned by rust/rust-toolchain.toml; builds are --frozen (locked and offline). Kernels and the metallib are untouched.
RUST_DIR := ../rust
CARGO := cargo
FROZEN := --frozen
FEATURES ?= fast-math
FEATURE_LEVELS := verify8 branch8 qsa-stage-v8 fast-math
ARM := $(if $(FEATURES),$(FEATURES),ordinary)
RUST_SERVER_DIR ?= ../build/rust-$(ARM)
CPP_BUILD ?= ../build/owned-native
CPP_FAST_MATH ?= ../build/http-fast-math
DIFF_WORK ?= ../build/rust-diff
DIFF_RUST_DIR ?= $(DIFF_WORK)/rust-$(ARM)
PYTHON ?= python3
RUST_RECEIPT_CACHE ?= 1
RUST_RECEIPT_CACHE_DIR ?= ../build/rust-receipt-cache
RECEIPT_RUN = $(PYTHON) -I ../scripts/rust_receipt_cache.py \
	--repo-root $(abspath ..) \
	--cache-dir "$(abspath $(RUST_RECEIPT_CACHE_DIR))" \
	--enabled "$(RUST_RECEIPT_CACHE)" \
	--cpp-oracle "$(CPP_ORACLE)"

# Pinned supply-chain tools built with --locked into build/rust-tools.
#   cargo install --locked --root build/rust-tools cargo-audit@0.22.0 cargo-deny@0.20.2 cargo-vet@0.10.2
RUST_TOOLS := $(abspath ../build/rust-tools/bin)

.PHONY: rust-check rust-test rust-release rust-server rust-diff rust-supply rust-fuzz rust-fuzz-build rust-fuzz-seeds rust-fuzz-run

# The differential fuzz targets link the unchanged C++ parsers, which need a libc++ with floating-point std::from_chars. CPP_ORACLE=0 skips them where the toolchain lacks it (CI's hosted macOS image); locally they are checked too.
CPP_ORACLE ?= 1

rust-check:
	cd $(RUST_DIR) && $(CARGO) fmt --all --check
	$(RECEIPT_RUN) --action clippy --arm ordinary --features "" -- \
		$(CARGO) clippy $(FROZEN) --workspace --all-targets -- -D warnings
	if [ "$(CPP_ORACLE)" = 1 ]; then \
		$(RECEIPT_RUN) --action clippy --arm cpp-oracle --features cpp-oracle -- \
			$(CARGO) clippy $(FROZEN) -p tokspur-fuzz --all-targets --features cpp-oracle -- -D warnings; \
	fi
	for level in $(FEATURE_LEVELS); do \
		$(RECEIPT_RUN) --action clippy --arm "$$level" --features "$$level" -- \
			$(CARGO) clippy $(FROZEN) -p tokspur-wire -p tokspur-server -p tokspur-fuzz --all-targets --features "$$level" -- -D warnings || exit 1; \
	done
	$(MAKE) -f rust.mk rust-test

rust-test:
	$(RECEIPT_RUN) --action test --arm ordinary --features "" -- \
		$(CARGO) test $(FROZEN) --workspace
	if [ "$(CPP_ORACLE)" = 1 ]; then \
		$(RECEIPT_RUN) --action test --arm cpp-oracle --features cpp-oracle -- \
			$(CARGO) test $(FROZEN) -p tokspur-fuzz --features cpp-oracle; \
	fi
	for level in $(FEATURE_LEVELS); do \
		$(RECEIPT_RUN) --action test --arm "$$level" --features "$$level" -- \
			$(CARGO) test $(FROZEN) -p tokspur-wire -p tokspur-server -p tokspur-fuzz --features "$$level" || exit 1; \
	done

rust-server:
	cd $(RUST_DIR) && $(CARGO) build $(FROZEN) --release -p tokspur-server \
		$(if $(FEATURES),--features $(FEATURES)) --target-dir $(abspath $(RUST_SERVER_DIR))

rust-release: rust-server
	cd $(RUST_DIR) && $(CARGO) build $(FROZEN) --release -p tokspur-cli \
		--target-dir $(abspath $(RUST_SERVER_DIR))

# Builds the C++ http-fast-math oracle into $(CPP_FAST_MATH) and runs the byte-level server differential into $(DIFF_WORK).
SERVER_DIFF = $(PYTHON) -I $(RUST_DIR)/tools/diff/server_diff.py \
	--cpp $(CPP_FAST_MATH)/tokspur-server --fake $(CPP_FAST_MATH)/test-http_fast_math \
	--rust $(DIFF_RUST_DIR)/release/tokspur-server
rust-diff:
	$(MAKE) -f rust.mk rust-server RUST_SERVER_DIR="$(DIFF_RUST_DIR)"
	$(MAKE) -C . http-fast-math HTTP_FAST_MATH_BUILD="$(CPP_FAST_MATH)"
	for arm in channel staged; do \
		$(SERVER_DIFF) --channel $$arm --work $(DIFF_WORK)/server-$$arm \
			--report $(DIFF_WORK)/server-report-$$arm.json || exit 1; \
	done
	$(SERVER_DIFF) --channel channel --rust-channel staged --work $(DIFF_WORK)/server-negative \
		--report $(DIFF_WORK)/server-report-negative.json
	$(SERVER_DIFF) --startup --work $(DIFF_WORK)/server-startup \
		--report $(DIFF_WORK)/server-report-startup.json

# Advisory databases are fetched (RustSec for cargo-audit and cargo-deny, OSV for the lockfile check). Other checks run offline against Cargo.lock. Licences are reviewed outside cargo-deny, so it selects only advisories, bans and sources.
rust-supply:
	cd $(RUST_DIR) && PATH="$(RUST_TOOLS):$$PATH" $(CARGO) audit
	cd $(RUST_DIR) && $(RUST_TOOLS)/cargo-deny fetch db && $(RUST_TOOLS)/cargo-deny --frozen check advisories bans sources
	cd $(RUST_DIR) && PATH="$(RUST_TOOLS):$$PATH" $(CARGO) vet --locked
	$(PYTHON) -I ../scripts/check_rust_advisories.py

# Supervised real-model cohort: run only through the one-owner model queue. The C++ and Rust fast-math servers drive the same unchanged C++ target in turn (one model resident at a time) over the scripted requests; the worker wire both ways, stdout and HTTP must match byte for byte, except the named host-memory telemetry fields, which are masked and counted. Generated IDs and hc_exact_final_logit_sha256 values are compared exactly.
#   make -C native all ci-components
#   make -C native -f rust.mk rust-cohort COHORT_POLICY=<policy file>
COHORT_TARGET ?= $(CPP_BUILD)/tokspur-native
COHORT_LIBRARY ?= $(CPP_BUILD)/tokspur-owned.metallib
.PHONY: rust-cohort
rust-cohort:
	test -n "$(COHORT_POLICY)" || { echo "rust-cohort needs COHORT_POLICY"; exit 2; }
	$(MAKE) -f rust.mk rust-server RUST_SERVER_DIR="$(DIFF_RUST_DIR)"
	$(SERVER_DIFF) --channel channel --target $(COHORT_TARGET) --policy $(COHORT_POLICY) \
		--library $(COHORT_LIBRARY) --work ../build/rust-diff/cohort --report ../build/rust-diff/cohort-report.json

# libFuzzer targets of rust/crates/tokspur-fuzz, on the pinned stable toolchain: no nightly and no cargo-fuzz. RUSTFLAGS add sanitizer-coverage instrumentation, overflow checks and debug assertions; --target keeps them off build scripts, and build.rs instruments the C++ oracle the same way under --cfg fuzzing. The wire target is built once per feature level. Seeds come from the repository's existing tests (the tokspur-fuzz and tokspur-gguf seed writers). Each target runs as a single libFuzzer job for FUZZ_SECONDS under nice -n FUZZ_NICE, at most FUZZ_PARALLEL at a time. Corpora, crash inputs, logs, dictionaries and the summary stay under build/rust-fuzz (Git-ignored). Model-free: no weights, no GPU.
FUZZ_SECONDS ?= 60
FUZZ_PARALLEL ?= 4
FUZZ_NICE ?= 10
FUZZ_DIR := $(abspath ../build/rust-fuzz)
RUST_ABS := $(abspath $(RUST_DIR))
FUZZ_TRIPLE := aarch64-apple-darwin
FUZZ_RUSTFLAGS := -Cpasses=sancov-module -Cllvm-args=-sanitizer-coverage-level=4 \
	-Cllvm-args=-sanitizer-coverage-inline-8bit-counters \
	-Cllvm-args=-sanitizer-coverage-pc-table \
	-Cllvm-args=-sanitizer-coverage-trace-compares \
	--cfg fuzzing -Cdebug-assertions -Coverflow-checks
FUZZ_BUILD = cd $(RUST_ABS) && RUSTFLAGS="$(FUZZ_RUSTFLAGS)" $(CARGO) build $(FROZEN) --release \
	-p tokspur-fuzz --target $(FUZZ_TRIPLE) --target-dir $(FUZZ_DIR)/target
FUZZ_OUT := $(FUZZ_DIR)/target/$(FUZZ_TRIPLE)/release
FUZZ_BINS := json json_diff http gguf gguf_diff response response_diff bpe
FUZZ_TARGETS ?= $(FUZZ_BINS) $(addprefix wire-,$(FEATURE_LEVELS))

rust-fuzz-build:
	$(RECEIPT_RUN) --action fuzz-build --arm cpp-oracle --features cpp-oracle \
		--fuzz-dir "$(FUZZ_DIR)" --fuzz-out "$(FUZZ_OUT)" \
		--fuzz-bins "$(FUZZ_BINS)" --fuzz-triple "$(FUZZ_TRIPLE)" \
		--fuzz-rustflags="$(FUZZ_RUSTFLAGS)" -- \
		$(CARGO) build $(FROZEN) --release -p tokspur-fuzz --target $(FUZZ_TRIPLE) --target-dir $(FUZZ_DIR)/target --features cpp-oracle $(addprefix --bin ,$(FUZZ_BINS))
	for level in $(FEATURE_LEVELS); do \
		$(RECEIPT_RUN) --action fuzz-build --arm "$$level" --features "$$level" \
			--fuzz-dir "$(FUZZ_DIR)" --fuzz-out "$(FUZZ_OUT)" \
			--fuzz-triple "$(FUZZ_TRIPLE)" --fuzz-rustflags="$(FUZZ_RUSTFLAGS)" -- \
			$(CARGO) build $(FROZEN) --release -p tokspur-fuzz --target $(FUZZ_TRIPLE) --target-dir $(FUZZ_DIR)/target --bin wire --features "$$level" || exit 1; \
	done

rust-fuzz-seeds:
	cd $(RUST_ABS) && TOKSPUR_FUZZ_SEEDS=$(FUZZ_DIR) $(CARGO) test $(FROZEN) -p tokspur-fuzz \
		--test seeds -- --ignored write_seed_corpora
	cd $(RUST_ABS) && TOKSPUR_GGUF_FUZZ_SEEDS=$(FUZZ_DIR)/corpus/gguf $(CARGO) test $(FROZEN) \
		-p tokspur-gguf --test gguf -- --ignored seeds::write_fuzz_seeds
	mkdir -p $(FUZZ_DIR)/corpus/gguf_diff
	cp $(FUZZ_DIR)/corpus/gguf/seed-* $(FUZZ_DIR)/corpus/gguf_diff/

rust-fuzz-run:
	rm -f $(FUZZ_DIR)/logs/*.status
	{ \
		echo "source: $$(git -C $(RUST_ABS) rev-parse HEAD)$$(git -C $(RUST_ABS) status --porcelain -- . ../native | grep -q . && echo ' plus uncommitted changes')"; \
		echo "toolchain file: $$(shasum -a 256 $(RUST_ABS)/rust-toolchain.toml | cut -d ' ' -f 1)"; \
		echo "Cargo.lock: $$(shasum -a 256 $(RUST_ABS)/Cargo.lock | cut -d ' ' -f 1)"; \
		cd $(RUST_ABS) && rustc -Vv && $(CARGO) -V; \
		c++ --version | head -n 1; \
		echo "RUSTFLAGS: $(FUZZ_RUSTFLAGS)"; \
		echo "seconds per target: $(FUZZ_SECONDS); parallel: $(FUZZ_PARALLEL); nice: $(FUZZ_NICE)"; \
		cd $(FUZZ_DIR)/bin && shasum -a 256 $(FUZZ_TARGETS); \
	} > $(FUZZ_DIR)/provenance.txt
	printf '%s\n' $(FUZZ_TARGETS) | xargs -P $(FUZZ_PARALLEL) -I {} \
		sh $(RUST_ABS)/tools/fuzz/run.sh $(FUZZ_DIR) {} $(FUZZ_SECONDS) $(FUZZ_NICE)
	sh $(RUST_ABS)/tools/fuzz/summary.sh $(FUZZ_DIR) $(FUZZ_TARGETS)

rust-fuzz: rust-fuzz-build rust-fuzz-seeds
	$(MAKE) -f rust.mk rust-fuzz-run
