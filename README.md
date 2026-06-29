# ark-cli

Rust CLI for Volcengine Ark Agent/Coding Plan endpoints.

```bash
export ARK_API_KEY=...
ark-cli list --kind text
ark-cli env --tool anthropic --model doubao-seed-2.0-code
ark-cli chat --model doubao-seed-2.0-code --message "hello"
ark-cli image --model doubao-seedream-5.0-lite --prompt "a clean icon"
ark-cli video create --model doubao-seedance-2.0-fast --prompt "a sunrise time-lapse"
ark-cli tts --text "hello" --dryrun
```

Configuration may be TOML or JSON:

```toml
api_key = "..."
protocol = "openai"
model = "doubao-seed-2.0-code"
base_url = "https://ark.cn-beijing.volces.com/api/plan/v3"
resource_id = "seed-tts-2.0"
```

Auto mode is intentionally rejected. Use a concrete model/resource id listed by
`ark-cli list`.

