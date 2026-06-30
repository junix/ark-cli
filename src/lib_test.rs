use super::*;

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
fn video_task_endpoint_can_include_id() {
    let url = endpoint_url(OPENAI_BASE_URL, Endpoint::VideoTasks, Some("task-123"));
    assert_eq!(
        url,
        "https://ark.cn-beijing.volces.com/api/plan/v3/contents/generations/tasks/task-123"
    );
}

#[test]
fn auto_model_is_rejected() {
    let error = validate_model("Auto", ModelKind::Text).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Auto mode is not supported by these Ark Plan endpoints; choose a concrete model"
    );
}

#[test]
fn listed_text_model_is_accepted() {
    validate_model("doubao-seed-2.0-code", ModelKind::Text).unwrap();
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
}

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
