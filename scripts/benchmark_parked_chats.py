#!/usr/bin/env python3
r"""Benchmark parked Japanese reasoning and Rust chats on a local Tokspur server.

Use identical weights, MTP3, shared-prefix settings and a fresh server for every
leg. On the serving Mac, start the selected build in a separate terminal:

  tokspur serve --speed-preset --drafter mtp --mtp-depth 3 --shared-prefix-cache \
      --parked-lanes-mib 4096 --telemetry-stdout > server.jsonl

Set TOKSPUR_API_KEY or pass --api-key-file pointing to the server's state/api-key.
Run from the checkout containing this script:

  python3 scripts/benchmark_parked_chats.py oracle --gap 0 --log server.jsonl --out oracle.jsonl
  # Restart with an empty server log before each interleaved leg.
  python3 scripts/benchmark_parked_chats.py interleaved --oracle oracle.jsonl \
      --log server.jsonl --out A1.jsonl

Create the oracle on main. Measure fresh legs in order A1 B1 B2 A2 B3 A3 A4 B4,
where A is main and B is the candidate. Keep every result, including slower legs.
Compare the full six requests and the four returns separately, within each pair:

  python3 scripts/benchmark_parked_chats.py compare --pair A1.jsonl B1.jsonl \
      --pair A2.jsonl B2.jsonl --pair A3.jsonl B3.jsonl --pair A4.jsonl B4.jsonl

HTTP seconds run from the client request through the final [DONE] event. Each
sum excludes startup and pacing. The default gap is seven seconds plus five
consecutive nominal seconds. Use --gap 0 for a separate hot diagnostic.
Full-leg eligibility requires nominal thermal/pressure readings and unchanged
system swap-in, swap-out and used-swap counters during every request. Returning
legs also require stable swap-outs and used swap across gaps, but check swap-ins
only during the four returns. Unknown host readings produce null eligibility.
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import stream_collector

ORACLE = ("J1", "J2", "J3", "R1", "R2", "R3")
INTERLEAVED = ("J1", "R1", "J2", "R2", "J3", "R3")
RETURNS = INTERLEAVED[2:]
PROMPTS = {
    "J1": "以下は運行記録です。記録でいう検品時間と荷物個数の違いを、日本語で1文で答えてください。\n"
    + "\n".join(
        f"記録{i:04d}：配送先は第{i % 17}倉庫。開始から{i % 47}分で検品し、{i % 13 + 1}個の荷物を登録した。"
        for i in range(284)
    ),
    "J2": "ここからは予定の妥当性を考えます。1台の機械が時刻0から休まず動きます。Aは3分で締切4、Bは2分で締切7、Cは1分で締切2。中断せずに全締切を守れる順序と3つの完了時刻を、日本語で理由を添えて簡潔に示してください。",
    "J3": "先ほどの予定で、Bの締切だけを時刻5にすると全件間に合いますか。日本語で一文の根拠を添えて答えてください。",
    "R1": "Review this Rust module. Describe the recurrence in one sentence.\n\n"
    + "\n".join(
        f"pub fn step_{i}(x: u32) -> u32 {{ x.wrapping_mul({i % 17 + 1}).wrapping_add({i * 3 + 5}) }}"
        for i in range(147)
    ),
    "R2": "The module needs an independent helper. Write only Rust code for pub fn clamped_add(x: u32, delta: u32) -> u32 that adds without wraparound, returning u32::MAX if the sum is too large. No main or test module.",
    "R3": "Write a Rust unit test named overflow_and_identity that tests clamped_add(u32::MAX, 1) and clamped_add(7, 0). Return only the test, without reimplementing the helper.",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def request_body(label, history):
    japanese = label.startswith("J")
    body = dict(
        model="tokspur",
        messages=[*history, {"role": "user", "content": PROMPTS[label]}],
        seed=24319 if japanese else 9983,
        temperature=0.65 if japanese else 0.0,
        top_p=0.90 if japanese else 1.0,
        top_k=20 if japanese else 1,
        max_tokens=190 if japanese else 166,
        enable_thinking=japanese,
        stream=True,
        stream_options={"include_usage": True},
    )
    if japanese:
        body["thinking_budget"] = 48
    return body


def collect_http(endpoint, body, key):
    start = time.monotonic()
    exchange = stream_collector.request(endpoint, body, key, timeout=300)
    events = exchange["events"]
    require(
        exchange["status"] == 200,
        f"HTTP {exchange['status']}, {exchange['connect_error']}",
    )
    require(events and events[-1][1] == b"[DONE]", "missing final SSE event")
    message = {}
    response_id = usage = finish = None
    for _, raw in events[:-1]:
        chunk = json.loads(raw)
        require("error" not in chunk and chunk["id"], "failed SSE chunk")
        if response_id is None:
            response_id = chunk["id"]
        require(chunk["id"] == response_id, "mixed response IDs")
        if chunk.get("usage") is not None:
            require(usage is None, "duplicate usage")
            usage = chunk["usage"]
        for choice in chunk["choices"]:
            require(
                choice["index"] == 0 and finish is None,
                "unexpected choice after finish",
            )
            for field, value in choice["delta"].items():
                require(
                    field in ("role", "content", "reasoning_content")
                    and isinstance(value, str),
                    f"unexpected assistant delta {field}",
                )
                message[field] = message.get(field, "") + value
            finish = choice.get("finish_reason")
    require(
        message.get("role") == "assistant" and message.get("content"),
        "missing assistant content",
    )
    require(
        not body["enable_thinking"] or message.get("reasoning_content"),
        "missing reasoning",
    )
    require(
        isinstance(finish, str) and usage and usage["completion_tokens"] > 0,
        "missing finish/usage",
    )
    require(
        usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
        "invalid token total",
    )
    return response_id, message, usage, finish, events[-1][0] - start


def next_event(log, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pos = log.tell()
        line = log.readline()
        if line.endswith("\n"):
            if line.lstrip().startswith("{"):
                return json.loads(line)
            continue
        log.seek(pos)
        time.sleep(0.025)
    raise ValueError("timed out awaiting complete native telemetry in --log")


def receipt(log, response_id):
    event = next_event(log)
    require(
        event["event"] == "done",
        "expected a single native done event from a fresh exclusive server",
    )
    done = event.get("native_done", event)
    timing = next_event(log)
    require(
        timing["event"] == "request_timing"
        and timing["outcome"] == "ok"
        and timing["response_id"] == response_id,
        "HTTP/native telemetry association failed",
    )
    return done, timing


class MacHost:
    """Read the host's thermal, memory pressure and exact global swap counters."""

    def __init__(self):
        self.lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        ptr = ctypes.c_void_p
        objc.objc_getClass.restype = objc.sel_registerName.restype = ptr
        send = ctypes.CFUNCTYPE(ptr, ptr, ptr)(("objc_msgSend", objc))
        self.thermal_state = ctypes.CFUNCTYPE(ctypes.c_long, ptr, ptr)(
            ("objc_msgSend", objc)
        )
        self.info = send(
            objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo")
        )
        require(self.info, "could not access NSProcessInfo")
        self.selector = objc.sel_registerName(b"thermalState")

    def sysctl(self, name, value):
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if self.lib.sysctlbyname(
            name, ctypes.byref(value), ctypes.byref(size), None, 0
        ):
            raise OSError(f"cannot read {name.decode()}")
        return value

    def read(self):
        # xsw_usage starts with three uint64s (total, available, used), then two uint32s.
        swap = self.sysctl(b"vm.swapusage", (ctypes.c_uint64 * 4)())
        pressure = self.sysctl(
            b"kern.memorystatus_vm_pressure_level", ctypes.c_int()
        ).value
        vm = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=5)
        counts = dict(re.findall(r"^(Swapins|Swapouts):\s+(\d+)\.", vm, re.M))
        return dict(
            thermal=self.thermal_state(self.info, self.selector),
            pressure=pressure,
            swapins=int(counts["Swapins"]),
            swapouts=int(counts["Swapouts"]),
            swap_used=swap[2],
        )


def nominal(sample):
    return sample["thermal"] == 0 and sample["pressure"] == 1


def pace(host, gap):
    quiet = min(gap, 5.0)
    time.sleep(gap - quiet)
    if not host or not gap:
        time.sleep(quiet)
        return
    start = stable = time.monotonic()
    while True:
        if not nominal(host.read()):
            stable = time.monotonic()
        if time.monotonic() - stable >= quiet:
            return
        require(time.monotonic() - start < 240, "host did not become nominal")
        time.sleep(min(1, quiet))


def summary(rows):
    totals = {}
    eligible = {}
    samples = [r[k] for r in rows for k in ("before", "after")]
    for name, labels in (("full", INTERLEAVED), ("returns", RETURNS)):
        turns = [r for r in rows if r["turn"] in labels]
        totals[name] = {
            lang: sum(r["http_s"] for r in turns if r["turn"][0] in lanes)
            for lang, lanes in (("Japanese", "J"), ("Rust", "R"), ("all", "JR"))
        }
        if None in samples:
            eligible[name] = None
        elif name == "full":
            eligible[name] = all(
                nominal(r["before"]) and r["before"] == r["after"] for r in turns
            )
        else:
            steady = all(
                nominal(s)
                and s["swapouts"] == samples[0]["swapouts"]
                and s["swap_used"] == samples[0]["swap_used"]
                for s in samples
            )
            eligible[name] = steady and all(
                r["before"]["swapins"] == r["after"]["swapins"] for r in turns
            )
    return {"http_seconds": totals, "resource_eligible": eligible}


def load_leg(path, mode):
    with path.open(encoding="utf-8") as stream:
        header, *rows, result = [json.loads(line) for line in stream]
    order = ORACLE if mode == "oracle" else INTERLEAVED
    require(
        header["mode"] == mode
        and header["schema"] == 1
        and [r["turn"] for r in rows] == list(order)
        and summary(rows) == result,
        f"incomplete or inconsistent {mode} leg in {path}",
    )
    return header, rows, result


def run(args):
    baseline = {}
    if args.mode == "interleaved":
        _, rows, _ = load_leg(args.oracle, "oracle")
        baseline = {r["turn"]: r for r in rows}
    key = (
        args.api_key_file.read_text().strip()
        if args.api_key_file
        else os.environ.get("TOKSPUR_API_KEY")
    )
    require(not key or not re.search(r"[\r\n]", key), "invalid API key")
    require(
        math.isfinite(args.gap) and args.gap >= 0,
        "--gap must be finite and nonnegative",
    )
    stream_collector.local_endpoint(args.endpoint)
    host = None
    if sys.platform == "darwin":
        try:
            host = MacHost()
            host.read()
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            print(f"Host observations unavailable: {error}", file=sys.stderr)
            host = None
    history = {"J": [], "R": []}
    prefixes = {"J": 0, "R": 0}
    old_restores = old_tokens = 0
    rows = []
    with (
        args.log.open(encoding="utf-8") as log,
        args.out.open("x", encoding="utf-8") as out,
    ):

        def save(value):
            print(
                json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                file=out,
                flush=True,
            )

        ready = next_event(log, timeout=120)
        require(
            ready["event"] == "ready"
            and ready["native_ready"]["native_mtp_depth"] == 3,
            "expected a fresh MTP3 server log with --telemetry-stdout",
        )
        require(
            not any(line.lstrip().startswith("{") for line in log.readlines()),
            "server log already contains events after ready; start a fresh server for this leg",
        )
        header = {"schema": 1, "mode": args.mode, "gap_s": args.gap}
        if baseline:
            header["oracle_sha256"] = hashlib.sha256(
                args.oracle.read_bytes()
            ).hexdigest()
        save(header)
        for label in INTERLEAVED if baseline else ORACLE:
            lane = label[0]
            body = request_body(label, history[lane])
            pace(host, args.gap)
            before = host.read() if host else None
            rid, assistant, usage, finish, wall = collect_http(args.endpoint, body, key)
            after = host.read() if host else None
            done, timing = receipt(log, rid)
            parked = done["parked_lanes"]
            require(
                parked["budget_bytes"] == 4096 * 1024**2,
                "expected a 4096 MiB parked-lane budget",
            )
            restores = parked["restores"] - old_restores
            restored = parked["restored_tokens"] - old_tokens
            old_restores, old_tokens = parked["restores"], parked["restored_tokens"]
            delta = (
                {k: after[k] - before[k] for k in ("swapins", "swapouts", "swap_used")}
                if host
                else None
            )
            row = dict(
                turn=label,
                request=body,
                assistant=assistant,
                usage=usage,
                finish=finish,
                native={
                    k: done[k]
                    for k in (
                        "committed_prefix",
                        "finish_reason",
                        "stop_id",
                        "hc_exact_final_logit_sha256",
                        "sampling",
                    )
                },
                response_id=rid,
                http_s=wall,
                prefill_s=timing["prefill_us"] / 1e6,
                restores=restores,
                restored_tokens=restored,
                before=before,
                after=after,
                swap_delta=delta,
            )
            save(row)
            sampler = done["sampling"]
            require(
                sampler["valid"] is True
                and sampler["entropy_prefix"] == done["committed_prefix"]
                and sampler["entropy_words"] >= 0
                and sampler["min_p"] == 0
                and all(
                    sampler[k] == body[k]
                    for k in ("temperature", "top_k", "top_p", "seed")
                )
                and re.fullmatch(r"[a-f0-9]{64}", done["hc_exact_final_logit_sha256"]),
                f"{label}: invalid native state",
            )
            require(
                timing["completion_tokens"] == usage["completion_tokens"],
                f"{label}: HTTP/native usage differs",
            )
            expected = prefixes[lane] if baseline else 0
            require(
                (restores, restored) == (int(expected > 0), expected),
                f"{label}: unexpected restore ({restores}, {restored})",
            )
            require(
                usage["prompt_tokens_details"]["cached_tokens"] == prefixes[lane],
                f"{label}: incomplete cached prefix",
            )
            if baseline:
                for field in ("request", "assistant", "usage", "finish", "native"):
                    require(
                        row[field] == baseline[label][field],
                        f"{label}: {field} differs from uninterrupted oracle",
                    )
            prefixes[lane] = done["committed_prefix"]
            history[lane] = [*body["messages"], assistant]
            rows.append(row)
            print(
                f"{label} {wall:.3f}s HTTP, {row['prefill_s']:.3f}s prefill, {restores} restores / {restored} tokens",
                flush=True,
            )
        result = summary(rows)
        save(result)
        print(json.dumps(result, indent=2))


def compare(pairs):
    seen = set()
    oracle = gap = None
    for number, pair in enumerate(pairs, 1):
        legs = []
        for path in pair:
            require(path.resolve() not in seen, f"leg used more than once: {path}")
            seen.add(path.resolve())
            header, _, totals = load_leg(path, "interleaved")
            if oracle is None:
                oracle, gap = header["oracle_sha256"], header["gap_s"]
            require(
                (header["oracle_sha256"], header["gap_s"]) == (oracle, gap),
                "mixed oracle or pacing",
            )
            legs.append(totals)
        for kind in ("full", "returns"):
            a, b = (leg["http_seconds"][kind] for leg in legs)
            gains = {k: round(100 * (a[k] - b[k]) / a[k], 3) for k in a}
            print(
                json.dumps(
                    dict(
                        pair=number,
                        files=[str(p) for p in pair],
                        outcome=kind,
                        main_seconds=a,
                        candidate_seconds=b,
                        reduction_pct=gains,
                        eligible_by_arm=[
                            leg["resource_eligible"][kind] for leg in legs
                        ],
                    )
                )
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("oracle", "interleaved"):
        sub = modes.add_parser(mode)
        sub.add_argument(
            "--endpoint", "--base-url", default="http://127.0.0.1:18081/v1"
        )
        sub.add_argument("--api-key-file", type=Path)
        sub.add_argument(
            "--log", type=Path, required=True, help="fresh server's JSONL stdout"
        )
        sub.add_argument("--out", type=Path, required=True, help="new JSONL receipt")
        sub.add_argument(
            "--gap",
            type=float,
            default=12,
            help="minimum seconds before each request (default 12)",
        )
        if mode == "interleaved":
            sub.add_argument("--oracle", type=Path, required=True)
    modes.add_parser("compare").add_argument(
        "--pair",
        nargs=2,
        action="append",
        type=Path,
        required=True,
        metavar=("MAIN", "CANDIDATE"),
    )
    args = parser.parse_args()
    if args.mode == "compare":
        compare(args.pair)
    else:
        run(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, stream_collector.CollectorError) as error:
        sys.exit(f"parked chat benchmark: {error}")
