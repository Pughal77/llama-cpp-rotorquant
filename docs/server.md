# llama-server and Web UI Guide

This guide covers how to start and use `llama-server`, interact with the built-in Web UI, configure multi-model setups using `.ini` preset files, and understand common command-line parameters such as `-fa` and `-ngl`.

For the full REST API specification and advanced developer notes, see [tools/server/README.md](file:///home/raijinshounen/myRoot/mySEJourney/model-inference/llama-cpp-rotorquant/tools/server/README.md).

---

## 1. Starting the Web UI with a GGUF Model

`llama-server` includes an embedded web user interface (llama-ui) built with SvelteKit. The Web UI is enabled by default.

When you run `llama-server`, it binds by default to `127.0.0.1:9931`. You can access the interface by opening your web browser at:

```
http://127.0.0.1:9931
```
(or `http://localhost:9931`)

### Basic Commands

#### Run a local GGUF model on CPU
```sh
./llama-server -m /path/to/model.gguf -c 4096
```

#### Run with full GPU acceleration and Flash Attention
```sh
# -ngl 99 (or -ngl all) offloads all layers to VRAM
# -fa on enables Flash Attention to minimize KV cache memory and speed up inference
./llama-server -m /path/to/model.gguf -ngl 99 -fa on -c 8192
```

#### Run directly from Hugging Face Hub
If you do not have the model downloaded locally, `llama-server` can fetch it directly from Hugging Face:
```sh
./llama-server -hf ggml-org/gemma-3-4b-it-GGUF:Q4_K_M -ngl 99 -fa on
```
The server downloads the required GGUF file to the local cache directory (`~/.cache/llama.cpp` by default) and starts serving immediately.

#### Bind to custom host and port
To allow other devices on your local network to access the server and Web UI:
```sh
./llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 8080
```

#### Run a multimodal (vision) model
Provide both the base model and the multimodal projector (`mmproj`):
```sh
./llama-server -m /path/to/vision-model.gguf --mmproj /path/to/mmproj.gguf -ngl 99
```
When using `-hf`, `mmproj` files are detected and downloaded automatically.

#### Enable built-in tools for agentic interactions
The Web UI can execute built-in tools (such as reading files or executing terminal commands) if enabled:
```sh
./llama-server -m /path/to/model.gguf --tools all
```
*Note: For security reasons, enabling tools automatically restricts CORS origins to `localhost`.*

#### Disabling the Web UI or using custom frontend assets
If you only want headless API access, disable the UI to save resources:
```sh
./llama-server -m /path/to/model.gguf --no-ui
```
To serve a custom frontend directory instead:
```sh
./llama-server -m /path/to/model.gguf --path /path/to/custom/web/dist
```

---

## 2. Loading Multiple Models with an INI File (Router Mode)

`llama-server` can run in **Router mode**, where a single master server manages multiple model instances. The router exposes the same OpenAI-compatible API and Web UI, dynamically loading and unloading models upon request.

To start in router mode, launch `llama-server` with `--models-preset` pointing to an `.ini` file:

```sh
llama-server --models-preset ./models.ini
```

### INI File Format and Structure

The configuration uses standard INI syntax:
- `[*]` defines global defaults applied to all models.
- `[section_name]` defines a model configuration. The section name serves as the model identifier/alias in the API and Web UI.
- Parameter keys inside each section match command-line arguments without the leading dashes (for example, `--n-gpu-layers 99` becomes `n-gpu-layers = 99` or `ngl = 99`).
- Boolean flags are written as `true`/`false` or `1`/`0`.

### Example `models.ini`

```ini
version = 1

; Global settings inherited by all models unless overridden
[*]
ctx-size = 8192
flash-attn = on
n-gpu-layers = 99
load-mode = mmap

; Model 1: Local Qwen 2.5 Coder
[qwen-coder]
model = /models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
ctx-size = 16384
batch-size = 2048
load-on-startup = true

; Model 2: Hugging Face Gemma 3
[gemma-3-4b]
hf = ggml-org/gemma-3-4b-it-GGUF:Q4_K_M
ctx-size = 8192
top-p = 0.95
temp = 0.7

; Model 3: Small reasoning model with custom chat template arguments
[deepseek-r1-distill]
model = /models/DeepSeek-R1-Distill-Qwen-8B-Q4_K_M.gguf
ctx-size = 32768
n-gpu-layers = all
chat-template-kwargs = {"reasoning_effort": "high"}
```

### Preset-Exclusive Options

The following settings are specific to model presets in `.ini` files:
- `load-on-startup` (`true` / `false`): Preloads the model into memory immediately when the server starts. If set to `false`, the model is loaded on demand when first requested.
- `stop-timeout` (integer, default: 10): Seconds to wait before forcibly terminating an unloaded model process.
- `dedup-cache-models` (`true` / `false`): When using `hf` sources, hides duplicate raw cache items from `GET /models` so only the preset alias is visible.

### Router Command-Line Options

When running with presets, you can control memory management and concurrency via router CLI arguments:

```sh
llama-server --models-preset ./models.ini --models-max 2 --models-memory-margin 2048
```

- `--models-preset <path>`: Path to the `.ini` preset file.
- `--models-max <N>`: Maximum number of models loaded simultaneously (default: 4, 0 = unlimited). Idle models are unloaded according to Least Recently Used (LRU) policy.
- `--models-memory-margin <N>`: Keeps `N` MiB of VRAM free per GPU. The router measures model memory and unloads idle models automatically before loading a new one to prevent out-of-memory errors.
- `--models-autoload` / `--no-models-autoload`: Controls whether models load automatically upon receiving a query if not already in memory (default: enabled).

### Using Multiple Models in the Web UI

When router mode is active, the Web UI automatically displays the **Model Selector** in the chat header and the **Models Manager** table in the sidebar:
- You can switch models between conversations or regenerate any answer using a different model.
- You can inspect loaded models, inspect VRAM consumption, and trigger manual load/unload operations directly from the UI.

### Querying Specific Models via API

For OpenAI-compatible endpoints, specify the preset section name in the `"model"` field:

```sh
curl http://127.0.0.1:9931/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-coder",
    "messages": [{"role": "user", "content": "Write a python fibonacci function."}]
  }'
```

For monitoring and properties endpoints, pass the model name as a query parameter:
```sh
curl "http://127.0.0.1:9931/props?model=qwen-coder"
```

---

## 3. Parameter Reference Guide

Below is a reference of essential parameters accepted by `llama-server` and `llama-cli`. Short flags can be used interchangeably with long options on the command line, or without leading dashes in `.ini` preset files.

### GPU Offloading and Hardware

| Parameter | Type / Values | Description |
| --------- | ------------- | ----------- |
| `-ngl, --gpu-layers, --n-gpu-layers` | integer, `auto`, `all` | Number of model layers to offload to GPU VRAM. Using `all` or a number greater than or equal to total layers offloads the entire model. `auto` lets the backend determine the layer count. |
| `-fa, --flash-attn` | `on`, `off`, `auto` | Flash Attention mode. Drastically reduces memory consumption of the attention mechanism and KV cache while accelerating prompt processing on supported GPUs. |
| `-sm, --split-mode` | `none`, `layer`, `row`, `tensor` | Multi-GPU split strategy. `layer` (default) splits transformer layers across GPUs in a pipeline. `row` splits weight tensors across GPUs. |
| `-ts, --tensor-split` | comma-separated list | Fraction of model offloaded to each GPU (for example, `3,1` allocates 75% to GPU 0 and 25% to GPU 1). |
| `-mg, --main-gpu` | integer (default: 0) | The primary GPU used for intermediate scratch buffers and KV cache. |
| `-fit, --fit` | `on`, `off` (default: `on`) | Automatically adjusts unset parameters (such as context size) so the model fits in available GPU memory. |
| `-fitt, --fit-target` | integer (default: 1024) | Target VRAM margin in MiB per GPU left free when `--fit` calculates memory allocation. |
| `-kvo, --kv-offload` | flag (default: enabled) | Whether to offload the Key/Value (KV) attention cache to GPU memory. Disable with `--no-kv-offload` to keep KV cache in CPU RAM. |
| `-lm, --load-mode` | `auto`, `mmap`, `mlock`, `none`, `dio` | Memory loading method. `mmap` maps files into virtual memory; `mlock` prevents memory from swapping to disk. |

### Context and Batch Sizes

| Parameter | Type / Values | Description |
| --------- | ------------- | ----------- |
| `-c, --ctx-size` | integer (default: 0) | Context window size in tokens. `0` loads the maximum native context defined in the model metadata. Lower values reduce VRAM usage. |
| `-b, --batch-size` | integer (default: 2048) | Logical maximum batch size for prompt processing. |
| `-ub, --ubatch-size` | integer (default: 512) | Physical micro-batch size. Controls the batch size processed in a single compute pass. Lower values save VRAM during long prompt evaluation; higher values improve throughput on large GPUs. |
| `-ctk, --cache-type-k` | `f16`, `q8_0`, `q4_0`, `turbo3_0`, `planar3_0`, etc. | Data type for the K (Key) cache. Quantizing the cache to `q8_0`, `q4_0`, or rotated quants (`turbo2_0`-`turbo4_0`, `planar3_0`-`planar4_0`, `iso3_0`-`iso4_0`) significantly reduces VRAM usage for long contexts. |
| `-ctv, --cache-type-v` | `f16`, `q8_0`, `q4_0`, `turbo3_0`, `planar3_0`, etc. | Data type for the V (Value) cache. Pair with `-ctk` for compact KV cache storage. |
| `-np, --parallel` | integer (default: -1) | Number of independent execution slots running concurrently. Default `-1` auto-tunes based on hardware. |
| `-cb, --cont-batching` | flag (default: enabled) | Enables continuous/dynamic batching across multiple client requests. |
| `-kvu, --kv-unified` | flag | Uses a shared unified KV cache pool across all execution slots. |

### CPU and Threads

| Parameter | Type / Values | Description |
| --------- | ------------- | ----------- |
| `-t, --threads` | integer (default: -1) | Number of CPU threads to use during token generation. `-1` uses the number of physical CPU cores. |
| `-tb, --threads-batch` | integer | Number of CPU threads used during prompt processing and batch evaluation (defaults to the same value as `--threads`). |
| `--cpu-mask` | hex mask | Hexadecimal CPU affinity mask to bind threads to specific CPU cores. |

### Model Sources

| Parameter | Type / Values | Description |
| --------- | ------------- | ----------- |
| `-m, --model` | file path | Path to local GGUF model file. |
| `-hf, --hf-repo` | `user/repo[:quant]` | Hugging Face repository and quantization tag. Default quantization is `Q4_K_M`. |
| `-hff, --hf-file` | file name | Specific GGUF file name within the Hugging Face repository. |
| `-mm, --mmproj` | file path | Path to the multimodal projector GGUF for vision or audio models. |

### Networking and Web UI

| Parameter | Type / Values | Description |
| --------- | ------------- | ----------- |
| `--host` | IP address (default: `127.0.0.1`) | IP address to bind to. Use `0.0.0.0` to listen on all network interfaces. |
| `--port` | integer (default: `9931`) | TCP port to listen on. |
| `--ui, --no-ui` | flag (default: enabled) | Enables or disables the embedded Web UI. |
| `--path` | directory path | Serves custom static HTML/JS files instead of the embedded UI assets. |
| `--api-key` | string | Shared secret API key required in the `Authorization: Bearer <key>` header for API requests. |
| `--cors-origins` | comma-separated list | Allowed CORS origins. Defaults to `*` for standard API use, or `localhost` when tools/agents are enabled. |
| `--tools` | `all` or list | Enables built-in server tools for function calling and agentic tasks. |

### Sampling Options

| Parameter | Type / Values | Description |
| --------- | ------------- | ----------- |
| `--temp, --temperature` | float (default: 0.80) | Randomness of predictions. Higher values increase variability; `0` produces greedy deterministic generation. |
| `--top-p` | float (default: 0.95) | Top-p (nucleus) sampling. Only tokens comprising the top cumulative probability `p` are considered. `1.0` disables. |
| `--top-k` | integer (default: 40) | Top-k sampling. Limits candidate tokens to the `k` most likely tokens. `0` disables. |
| `--min-p` | float (default: 0.05) | Min-p sampling. Discards tokens whose probability is lower than `min_p * top_token_probability`. |
| `--repeat-penalty` | float (default: 1.00) | Penalizes tokens that have already appeared in the recent context. `1.0` means no penalty. |
| `-n, --n-predict` | integer (default: -1) | Maximum number of tokens to predict. `-1` denotes infinity (up to context limit). |
| `--jinja` | flag (default: enabled) | Uses Jinja chat templating engine for formatting chat conversations according to model specifications. |

---

## 4. Useful Links

- [tools/server/README.md](file:///home/raijinshounen/myRoot/mySEJourney/model-inference/llama-cpp-rotorquant/tools/server/README.md) - Full server documentation and endpoint details.
- [docs/preset.md](file:///home/raijinshounen/myRoot/mySEJourney/model-inference/llama-cpp-rotorquant/docs/preset.md) - Documentation on system-level and Hugging Face INI presets.
- [docs/multi-gpu.md](file:///home/raijinshounen/myRoot/mySEJourney/model-inference/llama-cpp-rotorquant/docs/multi-gpu.md) - Multi-GPU offload and configuration guide.
- [docs/multimodal.md](file:///home/raijinshounen/myRoot/mySEJourney/model-inference/llama-cpp-rotorquant/docs/multimodal.md) - Multimodal model setup and usage guide.
