<p align="center">
  <a href="https://tokspur.com/#gh-light-mode-only"><img src="https://tokspur.com/assets/brand/tokspur-wordmark-960.png" width="320" alt="tokspur"></a>
  <a href="https://tokspur.com/#gh-dark-mode-only"><img src="https://tokspur.com/assets/brand/tokspur-wordmark-reversed-960.png" width="320" alt="tokspur"></a>
</p>

<p align="center"><strong>Private chat and code on your Mac.</strong></p>

<p align="center">
  <a href="https://tokspur.com/">Website</a> |
  <a href="https://tokspur.com/get-started/">Get started</a> |
  <a href="https://tokspur.com/docs/">Docs</a> |
  <a href="https://tokspur.com/engine/">Engine</a> |
  <a href="https://github.com/san-digital/tokspur">GitHub</a>
</p>

Tokspur runs Qwen3.8 Flash-Next locally on a Mac. It uses a native Metal worker and serves an OpenAI-compatible chat API.

Tokspur needs an Apple M5 Max with a 40-core GPU, 128 GB of unified memory and macOS 27 or later. Run the installer in Terminal, then follow the [setup guide](https://tokspur.com/get-started/) to verify the installation and connect a client.

```sh
curl -fsSL https://tokspur.com/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

## Add the model

Use the pinned base GGUF, pre-built Planar Pack v3 and shared Q8 MTP sidecar from [`san-digital/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF`](https://huggingface.co/san-digital/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF) (see the [model guide](https://tokspur.com/docs/models/)). Adoption verifies every file by SHA-256 and reuses the downloaded pack directly in place with no extra `tokspur pack build` step.

```sh
hf download san-digital/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF --local-dir /path/to/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF
tokspur adopt --target /path/to/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF --draft /path/to/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF
tokspur configure
tokspur serve --speed-preset
```

Connect to `http://127.0.0.1:18081/v1` with model `tokspur` and the generated API key.

For faster block-8 speculative decoding across code, `<think>` reasoning and multilingual workloads, pair with the optional drafter [`san-digital/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/san-digital/Qwen3.8-Flash-Next-DFlash2):

```sh
hf download san-digital/Qwen3.8-Flash-Next-DFlash2 --local-dir /path/to/Qwen3.8-Flash-Next-DFlash2
tokspur serve --speed-preset --drafter dflash2 --dflash2-dir /path/to/Qwen3.8-Flash-Next-DFlash2 --shared-prefix-cache
```

Add `--dflash2-vocab-gather cjk64k --dflash2-lookup hybrid --ple-ram-resident` when serving multilingual or CJK prompts alongside `<think>` reasoning.

### Pack formats

[`san-digital/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF`](https://huggingface.co/san-digital/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF) already includes Planar Pack v3 (`Qwen3.8-Flash-Next-Q4_K.tokspur.gguf`) and needs no local pack build. When adopting raw split-shard GGUF files instead, `tokspur pack build` writes the planar Pack v3 from the adopted target and MTP sidecar (`--format v2` selects the older layout, and `--pack-v3` on `serve` refuses fallback). Allow about **130 GiB free** when building a local pack from raw shards.

## Build

Install Xcode with the Metal toolchain, Git, Python 3.12 and Rust 1.99.1. Run these commands from the checkout root.

```sh
python3.12 scripts/bootstrap.py
.venv/bin/tokspur doctor
```

## Licences

[Apache 2.0](LICENSE). See [NOTICE](NOTICE) for third-party credits and [model licences](MODEL_LICENCES.md) for the weights. Report vulnerabilities through the [Security page](https://github.com/san-digital/tokspur/security).
