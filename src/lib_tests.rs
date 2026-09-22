use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Write `content` to a unique temp file whose name ends in `filename_with_ext`
/// (extension included, e.g. `"data.json"`) so dispatch-by-extension works.
fn temp_file(filename_with_ext: &str, content: &str) -> PathBuf {
    let n = TEMP_COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir()
        .join(format!("ark-cli-test-{n}-{filename_with_ext}"));
    fs::write(&path, content).expect("write temp file");
    path
}

// ---------------------------------------------------------------------------
// endpoint_url
// ---------------------------------------------------------------------------

#[test]
fn endpoint_url_joins_base_without_double_slash() {
    let url = endpoint_url(
        "https://ark.cn-beijing.volces.com/api/plan/v3/",
        Endpoint::Images,
        None,
    );
    assert_eq!(
        url,
        "https://ark.cn-beijing.volces.com/api/plan/v3/images/generations"
    );
}

#[test]
fn endpoint_url_keeps_base_without_trailing_slash_intact() {
    // A base with no trailing slash must not lose its last character when the
    // path is appended (regression guard against accidental double-trimming).
    let url = endpoint_url(
        "https://ark.cn-beijing.volces.com/api/plan/v3",
        Endpoint::Images,
        None,
    );
    assert_eq!(
        url,
        "https://ark.cn-beijing.volces.com/api/plan/v3/images/generations"
    );
}

#[test]
fn endpoint_url_table_covers_every_path_endpoint() {
    // Every non-TTS endpoint joins its literal path onto the (trimmed) base.
    let base = "https://x.example/p";
    for (endpoint, expected) in [
        (
            Endpoint::AnthropicMessages,
            "https://x.example/p/v1/messages",
        ),
        (
            Endpoint::OpenaiChat,
            "https://x.example/p/chat/completions",
        ),
        (Endpoint::Embeddings, "https://x.example/p/embeddings"),
        (
            Endpoint::Images,
            "https://x.example/p/images/generations",
        ),
        (
            Endpoint::VideoTasks,
            "https://x.example/p/contents/generations/tasks",
        ),
    ] {
        assert_eq!(endpoint_url(base, endpoint, None), expected);
    }
}

#[test]
fn endpoint_url_appends_task_id_for_non_video_endpoints() {
    // task_id appending is generic, not specific to VideoTasks.
    let url = endpoint_url(
        "https://x.example/p",
        Endpoint::AnthropicMessages,
        Some("msg_7"),
    );
    assert_eq!(url, "https://x.example/p/v1/messages/msg_7");
}

#[test]
fn video_task_endpoint_can_include_id() {
    let url = endpoint_url(OPENAI_BASE_URL, Endpoint::VideoTasks, Some("task-123"));
    assert_eq!(
        url,
        "https://ark.cn-beijing.volces.com/api/plan/v3/contents/generations/tasks/task-123"
    );
}

#[test]
fn endpoint_url_tts_endpoints_ignore_base_and_task_id() {
    // The three TTS endpoints return fixed constants regardless of the base
    // url or task id supplied; they never touch base_url trimming logic.
    let custom_base = "https://custom.example/whatever/";
    for (endpoint, expected) in [
        (Endpoint::TtsHttp, TTS_HTTP_URL),
        (Endpoint::TtsBidirectionalWs, TTS_BIDIRECTIONAL_WS_URL),
        (Endpoint::TtsUnidirectionalWs, TTS_UNIDIRECTIONAL_WS_URL),
    ] {
        assert_eq!(endpoint_url(custom_base, endpoint, None), expected);
        // task_id must also be ignored for these endpoints.
        assert_eq!(endpoint_url(custom_base, endpoint, Some("ignored")), expected);
    }
}

// ---------------------------------------------------------------------------
// validate_model
// ---------------------------------------------------------------------------

#[test]
fn auto_model_is_rejected() {
    let error = validate_model("Auto", ModelKind::Text).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Auto mode is not supported by these Ark Plan endpoints; choose a concrete model"
    );
}

#[test]
fn auto_rejection_is_case_insensitive_across_kinds() {
    // eq_ignore_ascii_case: any casing on any kind hits the Auto guard first.
    for (name, kind) in [
        ("auto", ModelKind::Text),
        ("AUTO", ModelKind::Embedding),
        ("AuTo", ModelKind::Speech),
    ] {
        let error = validate_model(name, kind).unwrap_err();
        assert!(
            error
                .to_string()
                .starts_with("Auto mode is not supported"),
            "expected Auto rejection for {name:?}"
        );
    }
}

#[test]
fn listed_text_model_is_accepted() {
    validate_model("doubao-seed-2.0-code", ModelKind::Text).unwrap();
}

#[test]
fn validate_model_accepts_one_entry_per_kind() {
    // Pin one catalog member for each ModelKind so a future catalog change
    // that drops a kind is caught here rather than at runtime.
    for (name, kind) in [
        ("doubao-seed-2.0-code", ModelKind::Text),
        ("doubao-embedding-vision", ModelKind::Embedding),
        ("doubao-seedream-5.0-lite", ModelKind::Image),
        ("doubao-seedance-2.0", ModelKind::Video),
        ("seed-tts-2.0", ModelKind::Speech),
    ] {
        validate_model(name, kind)
            .unwrap_or_else(|e| panic!("expected {name:?} ({kind:?}) to be accepted: {e}"));
    }
}

#[test]
fn wrong_kind_is_rejected() {
    let error = validate_model("doubao-seedream-5.0-lite", ModelKind::Text).unwrap_err();
    assert_eq!(
        error.to_string(),
        "unsupported Text model/resource id: doubao-seedream-5.0-lite"
    );
}

#[test]
fn unknown_model_is_rejected() {
    // A name that exists in no catalog at all.
    let error = validate_model("does-not-exist-xyz", ModelKind::Text).unwrap_err();
    assert_eq!(
        error.to_string(),
        "unsupported Text model/resource id: does-not-exist-xyz"
    );
}

// ---------------------------------------------------------------------------
// chat_body
// ---------------------------------------------------------------------------

#[test]
fn anthropic_chat_body_uses_messages_shape() {
    let body = chat_body(
        Protocol::Anthropic,
        "doubao-seed-2.0-code",
        Some("hello"),
        None,
        Some("be concise"),
        32,
    )
    .unwrap();
    assert_eq!(body["model"], "doubao-seed-2.0-code");
    assert_eq!(body["system"], "be concise");
    assert_eq!(body["max_tokens"], 32);
    assert_eq!(body["messages"][0]["role"], "user");
    // Harden: pin the message content too, not just the role.
    assert_eq!(body["messages"][0]["content"], "hello");
    // Harden: pin the exact top-level shape.
    let mut keys: Vec<&str> = body.as_object().unwrap().keys().map(|s| s.as_str()).collect();
    keys.sort();
    assert_eq!(keys, vec!["max_tokens", "messages", "model", "system"]);
}

#[test]
fn anthropic_chat_body_omits_system_when_none() {
    let body = chat_body(
        Protocol::Anthropic,
        "doubao-seed-2.0-code",
        Some("hello"),
        None,
        None,
        16,
    )
    .unwrap();
    assert!(body.get("system").is_none());
    let mut keys: Vec<&str> = body.as_object().unwrap().keys().map(|s| s.as_str()).collect();
    keys.sort();
    assert_eq!(keys, vec!["max_tokens", "messages", "model"]);
}

#[test]
fn openai_chat_body_prepends_system_to_default_message() {
    // Default messages path (message=Some, messages_json=None) yields a single
    // user message; with system=Some the OpenAI branch prepends a system turn.
    let body = chat_body(
        Protocol::Openai,
        "doubao-seed-2.0-code",
        Some("hello"),
        None,
        Some("sys-prompt"),
        16,
    )
    .unwrap();
    assert_eq!(body["model"], "doubao-seed-2.0-code");
    assert_eq!(body["max_tokens"], 16);
    assert_eq!(body["messages"].as_array().unwrap().len(), 2);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "sys-prompt");
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["messages"][1]["content"], "hello");
}

#[test]
fn openai_chat_body_prepends_system_to_messages_json_array() {
    let messages_json = r#"[{"role":"user","content":"hi"}]"#;
    let body = chat_body(
        Protocol::Openai,
        "doubao-seed-2.0-code",
        None,
        Some(messages_json),
        Some("sys"),
        8,
    )
    .unwrap();
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "sys");
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["messages"][1]["content"], "hi");
}

#[test]
fn openai_chat_body_without_system_leaves_messages_unchanged() {
    let messages_json = r#"[{"role":"user","content":"hi"}]"#;
    let body = chat_body(
        Protocol::Openai,
        "doubao-seed-2.0-code",
        None,
        Some(messages_json),
        None,
        8,
    )
    .unwrap();
    assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["messages"][0]["content"], "hi");
    // OpenAI body never carries a top-level "system" key.
    assert!(body.get("system").is_none());
}

#[test]
fn openai_chat_body_rejects_non_array_messages_json_when_system_present() {
    // messages_json decodes to an object; with system=Some the prepend guard
    // must bail rather than silently swallow the value.
    let error = chat_body(
        Protocol::Openai,
        "doubao-seed-2.0-code",
        None,
        Some(r#"{"not":"an array"}"#),
        Some("sys"),
        8,
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "messages_json must decode to an array"
    );
}

// ---------------------------------------------------------------------------
// AppConfig::merged
// ---------------------------------------------------------------------------

#[test]
fn config_overrides_file_with_cli_values() {
    let cli = Cli::parse_from([
        "ark-cli",
        "--api-key",
        "from-cli",
        "--model",
        "doubao-seed-2.0-code",
        "list",
    ]);
    let config = AppConfig {
        api_key: Some("from-file".to_string()),
        model: Some("deepseek-v4-flash".to_string()),
        ..Default::default()
    }
    .merged(&cli);
    assert_eq!(config.api_key.as_deref(), Some("from-cli"));
    assert_eq!(config.model.as_deref(), Some("doubao-seed-2.0-code"));
}

#[test]
fn merged_retains_config_values_when_cli_fields_absent() {
    // The merge must only override on the CLI-present direction; absent CLI
    // fields leave every config value untouched. We construct `Cli` directly
    // (rather than `parse_from`) so clap's `env` fallback can't inject values
    // from the ambient ARK_* environment and muddy the retain-direction check.
    let cli = Cli {
        config: None,
        api_key: None,
        base_url: None,
        model: None,
        resource_id: None,
        protocol: None,
        dryrun: false,
        json: false,
        verbose: false,
        command: Command::List { kind: None },
    };
    let config = AppConfig {
        api_key: Some("k".to_string()),
        base_url: Some("https://from-config".to_string()),
        model: Some("doubao-seed-2.0-code".to_string()),
        resource_id: Some("seed-tts-2.0".to_string()),
        protocol: Some(Protocol::Anthropic),
    }
    .merged(&cli);
    assert_eq!(config.api_key.as_deref(), Some("k"));
    assert_eq!(config.base_url.as_deref(), Some("https://from-config"));
    assert_eq!(config.model.as_deref(), Some("doubao-seed-2.0-code"));
    assert_eq!(config.resource_id.as_deref(), Some("seed-tts-2.0"));
    assert_eq!(config.protocol, Some(Protocol::Anthropic));
}

#[test]
fn merged_overrides_base_url_resource_id_and_protocol() {
    let cli = Cli::parse_from([
        "ark-cli",
        "--base-url",
        "https://cli-url",
        "--resource-id",
        "seed-tts-2.0",
        "--protocol",
        "anthropic",
        "list",
    ]);
    let config = AppConfig {
        base_url: Some("https://config-url".to_string()),
        resource_id: Some("volc.seedasr.sauc.duration".to_string()),
        protocol: Some(Protocol::Openai),
        ..Default::default()
    }
    .merged(&cli);
    assert_eq!(config.base_url.as_deref(), Some("https://cli-url"));
    assert_eq!(config.resource_id.as_deref(), Some("seed-tts-2.0"));
    assert_eq!(config.protocol, Some(Protocol::Anthropic));
}

// ---------------------------------------------------------------------------
// append_query_params
// ---------------------------------------------------------------------------

#[test]
fn append_query_params_empty_returns_url_unchanged() {
    let url = "https://example.com/api".to_string();
    assert_eq!(append_query_params(url.clone(), &[]), url);
}

#[test]
fn append_query_params_single_param() {
    let params = [("page_num".to_string(), "1".to_string())];
    assert_eq!(
        append_query_params("https://example.com/api".to_string(), &params),
        "https://example.com/api?page_num=1"
    );
}

#[test]
fn append_query_params_multiple_params_joined_with_amp() {
    let params = [
        ("page_num".to_string(), "1".to_string()),
        ("page_size".to_string(), "20".to_string()),
        ("filter.status".to_string(), "succeeded".to_string()),
    ];
    assert_eq!(
        append_query_params("https://example.com/api".to_string(), &params),
        "https://example.com/api?page_num=1&page_size=20&filter.status=succeeded"
    );
}

#[test]
fn append_query_params_does_not_url_encode_values() {
    // The implementation joins with raw `format!("{key}={value}")`; values
    // containing reserved characters are passed through verbatim. Pin this so
    // a future switch to percent-encoding is a deliberate, reviewed change.
    let params = [("q".to_string(), "a&b=c".to_string())];
    assert_eq!(
        append_query_params("https://example.com/api".to_string(), &params),
        "https://example.com/api?q=a&b=c"
    );
}

// ---------------------------------------------------------------------------
// read_value_arg / read_json_arg
// ---------------------------------------------------------------------------

#[test]
fn read_value_arg_returns_literal_unchanged() {
    assert_eq!(read_value_arg("plain literal").unwrap(), "plain literal");
}

#[test]
fn read_value_arg_reads_at_file_path() {
    let path = temp_file("value-arg.txt", "file contents here");
    let arg = format!("@{}", path.display());
    assert_eq!(read_value_arg(&arg).unwrap(), "file contents here");
    let _ = fs::remove_file(&path);
}

#[test]
fn read_value_arg_missing_file_errors() {
    let error = read_value_arg("@/definitely/not/a/real/path/xyz").unwrap_err();
    assert!(
        error.to_string().contains("failed to read"),
        "unexpected error: {error}"
    );
}

#[test]
fn read_json_arg_parses_literal_json() {
    let value = read_json_arg(r#"{"a":1,"b":[2,3]}"#).unwrap();
    assert_eq!(value["a"], 1);
    assert_eq!(value["b"][0], 2);
    assert_eq!(value["b"][1], 3);
}

#[test]
fn read_json_arg_reads_and_parses_at_file() {
    let path = temp_file("json-arg.json", r#"{"k":"v"}"#);
    let arg = format!("@{}", path.display());
    let value = read_json_arg(&arg).unwrap();
    assert_eq!(value["k"], "v");
    let _ = fs::remove_file(&path);
}

#[test]
fn read_json_arg_invalid_json_errors() {
    let error = read_json_arg("not json at all").unwrap_err();
    assert!(
        error.to_string().contains("failed to parse JSON body"),
        "unexpected error: {error}"
    );
}

// ---------------------------------------------------------------------------
// load_config
// ---------------------------------------------------------------------------

#[test]
fn load_config_none_returns_default() {
    let config = load_config(None).unwrap();
    assert!(config.api_key.is_none());
    assert!(config.base_url.is_none());
    assert!(config.model.is_none());
    assert!(config.resource_id.is_none());
    assert!(config.protocol.is_none());
}

#[test]
fn load_config_json_file_parses_fields() {
    let path = temp_file(
        "config.json",
        r#"{"api_key":"from-json","model":"doubao-seed-2.0-code"}"#,
    );
    let config = load_config(Some(&path)).unwrap();
    assert_eq!(config.api_key.as_deref(), Some("from-json"));
    assert_eq!(config.model.as_deref(), Some("doubao-seed-2.0-code"));
    let _ = fs::remove_file(&path);
}

#[test]
fn load_config_toml_file_parses_fields() {
    let path = temp_file(
        "config.toml",
        "api_key = \"from-toml\"\nmodel = \"glm-5.2\"\n",
    );
    let config = load_config(Some(&path)).unwrap();
    assert_eq!(config.api_key.as_deref(), Some("from-toml"));
    assert_eq!(config.model.as_deref(), Some("glm-5.2"));
    let _ = fs::remove_file(&path);
}

#[test]
fn load_config_missing_file_errors() {
    let path = PathBuf::from("/definitely/not/a/real/config.xyz");
    let error = load_config(Some(&path)).unwrap_err();
    assert!(
        error.to_string().contains("failed to read config"),
        "unexpected error: {error}"
    );
}

// ---------------------------------------------------------------------------
// resolve_model / resolve_resource_id
// ---------------------------------------------------------------------------

#[test]
fn resolve_model_errors_when_neither_cli_nor_config_provide_it() {
    let config = AppConfig::default();
    let error = resolve_model(None, &config, ModelKind::Text).unwrap_err();
    assert_eq!(
        error.to_string(),
        "missing model; pass --model, ARK_MODEL, or config model"
    );
}

#[test]
fn resolve_model_falls_back_to_config_model() {
    let config = AppConfig {
        model: Some("doubao-seed-2.0-code".to_string()),
        ..Default::default()
    };
    assert_eq!(resolve_model(None, &config, ModelKind::Text).unwrap(), "doubao-seed-2.0-code");
}

#[test]
fn resolve_model_cli_value_takes_precedence_over_config() {
    let config = AppConfig {
        model: Some("deepseek-v4-flash".to_string()),
        ..Default::default()
    };
    assert_eq!(
        resolve_model(Some("doubao-seed-2.0-code"), &config, ModelKind::Text).unwrap(),
        "doubao-seed-2.0-code"
    );
}

#[test]
fn resolve_model_rejects_invalid_model() {
    let config = AppConfig::default();
    let error = resolve_model(Some("no-such-model"), &config, ModelKind::Text).unwrap_err();
    assert_eq!(
        error.to_string(),
        "unsupported Text model/resource id: no-such-model"
    );
}

#[test]
fn resolve_resource_id_defaults_to_seed_tts_when_unspecified() {
    let config = AppConfig::default();
    assert_eq!(resolve_resource_id(None, &config).unwrap(), "seed-tts-2.0");
}

#[test]
fn resolve_resource_id_uses_config_value_when_cli_absent() {
    let config = AppConfig {
        resource_id: Some("volc.seedasr.sauc.duration".to_string()),
        ..Default::default()
    };
    assert_eq!(
        resolve_resource_id(None, &config).unwrap(),
        "volc.seedasr.sauc.duration"
    );
}

#[test]
fn resolve_resource_id_cli_value_takes_precedence_over_config() {
    let config = AppConfig {
        resource_id: Some("volc.seedasr.sauc.duration".to_string()),
        ..Default::default()
    };
    assert_eq!(
        resolve_resource_id(Some("seed-tts-2.0"), &config).unwrap(),
        "seed-tts-2.0"
    );
}

#[test]
fn resolve_resource_id_rejects_invalid_id() {
    let config = AppConfig::default();
    let error = resolve_resource_id(Some("bad-tts-id"), &config).unwrap_err();
    assert_eq!(
        error.to_string(),
        "unsupported Speech model/resource id: bad-tts-id"
    );
}
