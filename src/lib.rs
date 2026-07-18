use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use reqwest::blocking::{Client, RequestBuilder};
use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const ANTHROPIC_BASE_URL: &str = "https://ark.cn-beijing.volces.com/api/plan";
const OPENAI_BASE_URL: &str = "https://ark.cn-beijing.volces.com/api/plan/v3";
const TTS_HTTP_URL: &str = "https://openspeech.bytedance.com/api/v3/plan/tts/unidirectional";
const TTS_BIDIRECTIONAL_WS_URL: &str = "wss://openspeech.bytedance.com/api/v3/plan/tts/bidirection";
const TTS_UNIDIRECTIONAL_WS_URL: &str =
    "wss://openspeech.bytedance.com/api/v3/plan/tts/unidirectional/stream";

#[derive(Debug, Clone, Parser)]
#[command(
    name = "ark-cli",
    version,
    about = "Volcengine Ark Agent/Coding Plan CLI",
    after_help = concat!("Source: ", env!("PROJECT_SOURCE_PATH"))
)]
pub struct Cli {
    /// TOML or JSON config file. Values can also come from ARK_* env vars.
    #[arg(long, global = true, env = "ARK_CONFIG")]
    config: Option<PathBuf>,

    #[arg(long, global = true, env = "ARK_API_KEY")]
    api_key: Option<String>,

    #[arg(long, global = true, env = "ARK_BASE_URL")]
    base_url: Option<String>,

    #[arg(long, global = true, env = "ARK_MODEL")]
    model: Option<String>,

    #[arg(long, global = true, env = "ARK_RESOURCE_ID")]
    resource_id: Option<String>,

    #[arg(long, global = true, env = "ARK_PROTOCOL")]
    protocol: Option<Protocol>,

    /// Print the request that would be sent instead of calling the network.
    #[arg(long, global = true)]
    dryrun: bool,

    /// Emit structured JSON where supported.
    #[arg(long, global = true)]
    json: bool,

    /// Print request details to stderr.
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Subcommand)]
enum Command {
    /// List supported model names and speech resource ids.
    List {
        #[arg(long, value_enum)]
        kind: Option<ModelKind>,
    },
    /// Print a configured endpoint URL.
    Endpoint {
        #[arg(value_enum)]
        endpoint: Endpoint,
        #[arg(long)]
        task_id: Option<String>,
    },
    /// Print shell environment for Anthropic/OpenAI compatible tools.
    Env {
        #[arg(long, value_enum)]
        tool: Tool,
        #[arg(long)]
        model: Option<String>,
    },
    /// Print Claude Code settings.json content.
    ClaudeSettings {
        #[arg(long)]
        model: Option<String>,
    },
    /// Send a text chat request through the Anthropic or OpenAI compatible API.
    Chat {
        #[arg(long)]
        message: Option<String>,
        /// JSON value for OpenAI messages array or Anthropic messages array.
        #[arg(long)]
        messages_json: Option<String>,
        #[arg(long)]
        system: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value_t = 1024)]
        max_tokens: u32,
    },
    /// Send an OpenAI-compatible embeddings request.
    Embeddings {
        #[arg(long)]
        input: String,
        #[arg(long)]
        model: Option<String>,
    },
    /// Send an image generation request.
    Image {
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        size: Option<String>,
    },
    /// Create/query/list/cancel video generation tasks.
    Video {
        #[command(subcommand)]
        command: VideoCommand,
    },
    /// Send a speech HTTP TTS request.
    Tts {
        #[arg(long)]
        text: Option<String>,
        /// Raw JSON body, @file, or - for stdin.
        #[arg(long)]
        body_json: Option<String>,
        #[arg(long)]
        resource_id: Option<String>,
        #[arg(long)]
        voice: Option<String>,
    },
    /// Send a generic request to a known Ark endpoint.
    Request {
        #[arg(value_enum)]
        endpoint: Endpoint,
        #[arg(long, default_value = "POST")]
        method: String,
        /// Raw JSON body, @file, or - for stdin.
        #[arg(long)]
        body_json: Option<String>,
        #[arg(long)]
        task_id: Option<String>,
    },
}

#[derive(Debug, Clone, Subcommand)]
enum VideoCommand {
    Create {
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long)]
        model: Option<String>,
        /// Raw JSON body, @file, or - for stdin.
        #[arg(long)]
        body_json: Option<String>,
    },
    Get {
        task_id: String,
    },
    List {
        #[arg(long)]
        page_num: Option<u32>,
        #[arg(long)]
        page_size: Option<u32>,
        #[arg(long)]
        status: Option<String>,
        #[arg(long)]
        task_id: Vec<String>,
        #[arg(long)]
        model: Option<String>,
    },
    Cancel {
        task_id: String,
    },
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum Tool {
    Anthropic,
    Openai,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Protocol {
    Anthropic,
    Openai,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    Text,
    Embedding,
    Image,
    Video,
    Speech,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
enum Endpoint {
    AnthropicMessages,
    OpenaiChat,
    Embeddings,
    Images,
    VideoTasks,
    TtsHttp,
    TtsBidirectionalWs,
    TtsUnidirectionalWs,
}

#[derive(Debug, Clone, Serialize)]
struct CatalogEntry {
    name: &'static str,
    kind: ModelKind,
    note: &'static str,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct AppConfig {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    resource_id: Option<String>,
    protocol: Option<Protocol>,
}

impl AppConfig {
    fn merged(mut self, cli: &Cli) -> Self {
        if cli.api_key.is_some() {
            self.api_key = cli.api_key.clone();
        }
        if cli.base_url.is_some() {
            self.base_url = cli.base_url.clone();
        }
        if cli.model.is_some() {
            self.model = cli.model.clone();
        }
        if cli.resource_id.is_some() {
            self.resource_id = cli.resource_id.clone();
        }
        if cli.protocol.is_some() {
            self.protocol = cli.protocol;
        }
        self
    }

    fn protocol_or(&self, default: Protocol) -> Protocol {
        self.protocol.unwrap_or(default)
    }

    fn base_url_for(&self, protocol: Protocol) -> &str {
        self.base_url
            .as_deref()
            .unwrap_or_else(|| default_base_url(protocol))
    }

    fn api_key(&self) -> Result<&str> {
        self.api_key
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                anyhow!("missing API key; set --api-key, ARK_API_KEY, or config api_key")
            })
    }
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    let config = load_config(cli.config.as_deref())?.merged(&cli);
    run_with(cli, config)
}

fn run_with(cli: Cli, config: AppConfig) -> Result<()> {
    match &cli.command {
        Command::List { kind } => print_catalog(*kind, cli.json),
        Command::Endpoint { endpoint, task_id } => {
            let protocol = config.protocol_or(protocol_for_endpoint(*endpoint));
            println!(
                "{}",
                endpoint_url(config.base_url_for(protocol), *endpoint, task_id.as_deref())
            );
            Ok(())
        }
        Command::Env { tool, model } => {
            let model = resolve_model(model.as_deref(), &config, ModelKind::Text)?;
            print_env(*tool, &model, &config)
        }
        Command::ClaudeSettings { model } => {
            let model = resolve_model(model.as_deref(), &config, ModelKind::Text)?;
            print_claude_settings(&model, &config)
        }
        Command::Chat {
            message,
            messages_json,
            system,
            model,
            max_tokens,
        } => {
            let protocol = config.protocol_or(Protocol::Openai);
            let model = resolve_model(model.as_deref(), &config, ModelKind::Text)?;
            let body = chat_body(
                protocol,
                &model,
                message.as_deref(),
                messages_json.as_deref(),
                system.as_deref(),
                *max_tokens,
            )?;
            let endpoint = match protocol {
                Protocol::Anthropic => Endpoint::AnthropicMessages,
                Protocol::Openai => Endpoint::OpenaiChat,
            };
            send_json(&cli, &config, protocol, endpoint, None, Method::POST, body)
        }
        Command::Embeddings { input, model } => {
            let model = resolve_model(model.as_deref(), &config, ModelKind::Embedding)?;
            let body = json!({ "model": model, "input": read_value_arg(input)? });
            send_json(
                &cli,
                &config,
                Protocol::Openai,
                Endpoint::Embeddings,
                None,
                Method::POST,
                body,
            )
        }
        Command::Image {
            prompt,
            model,
            size,
        } => {
            let model = resolve_model(model.as_deref(), &config, ModelKind::Image)?;
            let mut body = json!({ "model": model, "prompt": read_value_arg(prompt)? });
            if let Some(size) = size {
                body["size"] = Value::String(size.clone());
            }
            send_json(
                &cli,
                &config,
                Protocol::Openai,
                Endpoint::Images,
                None,
                Method::POST,
                body,
            )
        }
        Command::Video { command } => handle_video(&cli, &config, command),
        Command::Tts {
            text,
            body_json,
            resource_id,
            voice,
        } => {
            let resource_id = resolve_resource_id(resource_id.as_deref(), &config)?;
            let mut body = if let Some(body_json) = body_json {
                read_json_arg(body_json)?
            } else {
                json!({
                    "resource_id": resource_id,
                    "text": read_value_arg(text.as_deref().unwrap_or("-"))?,
                })
            };
            if let Some(voice) = voice {
                body["voice"] = Value::String(voice.clone());
            }
            send_json(
                &cli,
                &config,
                Protocol::Openai,
                Endpoint::TtsHttp,
                None,
                Method::POST,
                body,
            )
        }
        Command::Request {
            endpoint,
            method,
            body_json,
            task_id,
        } => {
            let method = method.parse::<Method>().context("invalid HTTP method")?;
            let protocol = config.protocol_or(protocol_for_endpoint(*endpoint));
            let body = match body_json {
                Some(value) => read_json_arg(value)?,
                None => Value::Object(Default::default()),
            };
            send_json(
                &cli,
                &config,
                protocol,
                *endpoint,
                task_id.as_deref(),
                method,
                body,
            )
        }
    }
}

fn handle_video(cli: &Cli, config: &AppConfig, command: &VideoCommand) -> Result<()> {
    match command {
        VideoCommand::Create {
            prompt,
            model,
            body_json,
        } => {
            let body = if let Some(body_json) = body_json {
                read_json_arg(body_json)?
            } else {
                let model = resolve_model(model.as_deref(), config, ModelKind::Video)?;
                json!({
                    "model": model,
                    "prompt": read_value_arg(prompt.as_deref().unwrap_or("-"))?,
                })
            };
            send_json(
                cli,
                config,
                Protocol::Openai,
                Endpoint::VideoTasks,
                None,
                Method::POST,
                body,
            )
        }
        VideoCommand::Get { task_id } => send_json(
            cli,
            config,
            Protocol::Openai,
            Endpoint::VideoTasks,
            Some(task_id),
            Method::GET,
            Value::Object(Default::default()),
        ),
        VideoCommand::List {
            page_num,
            page_size,
            status,
            task_id,
            model,
        } => {
            let url = endpoint_url(
                config.base_url_for(Protocol::Openai),
                Endpoint::VideoTasks,
                None,
            );
            let mut query = Vec::new();
            if let Some(page_num) = page_num {
                query.push(("page_num".to_string(), page_num.to_string()));
            }
            if let Some(page_size) = page_size {
                query.push(("page_size".to_string(), page_size.to_string()));
            }
            if let Some(status) = status {
                query.push(("filter.status".to_string(), status.clone()));
            }
            if !task_id.is_empty() {
                query.push(("filter.task_ids".to_string(), task_id.join(",")));
            }
            if let Some(model) = model {
                validate_model(model, ModelKind::Video)?;
                query.push(("filter.model".to_string(), model.clone()));
            }
            let url = append_query_params(url, &query);
            send_url(
                cli,
                config,
                Protocol::Openai,
                Method::GET,
                url,
                Value::Object(Default::default()),
            )
        }
        VideoCommand::Cancel { task_id } => send_json(
            cli,
            config,
            Protocol::Openai,
            Endpoint::VideoTasks,
            Some(task_id),
            Method::DELETE,
            Value::Object(Default::default()),
        ),
    }
}

fn send_json(
    cli: &Cli,
    config: &AppConfig,
    protocol: Protocol,
    endpoint: Endpoint,
    task_id: Option<&str>,
    method: Method,
    body: Value,
) -> Result<()> {
    let url = endpoint_url(config.base_url_for(protocol), endpoint, task_id);
    send_url(cli, config, protocol, method, url, body)
}

fn send_url(
    cli: &Cli,
    config: &AppConfig,
    protocol: Protocol,
    method: Method,
    url: String,
    body: Value,
) -> Result<()> {
    if cli.dryrun {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "method": method.as_str(),
                "url": url,
                "protocol": protocol,
                "body": body,
            }))?
        );
        return Ok(());
    }

    let api_key = config.api_key()?;
    if cli.verbose {
        eprintln!("{} {}", method, url);
    }
    let client = Client::new();
    let request = client.request(method.clone(), &url);
    let request = apply_auth(request, protocol, api_key);
    let request = if method == Method::GET || method == Method::DELETE {
        request
    } else {
        request.json(&body)
    };
    let response = request.send().context("request failed")?;
    let status = response.status();
    let text = response.text().context("failed to read response body")?;
    if !status.is_success() {
        bail!("request returned {status}: {text}");
    }
    println!("{text}");
    Ok(())
}

fn apply_auth(request: RequestBuilder, protocol: Protocol, api_key: &str) -> RequestBuilder {
    match protocol {
        Protocol::Anthropic => request
            .header("x-api-key", api_key)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("anthropic-version", "2023-06-01"),
        Protocol::Openai => request.header("Authorization", format!("Bearer {api_key}")),
    }
}

fn load_config(path: Option<&Path>) -> Result<AppConfig> {
    let Some(path) = path else {
        return Ok(AppConfig::default());
    };
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read config {}", path.display()))?;
    if path.extension().is_some_and(|ext| ext == "json") {
        serde_json::from_str(&raw).context("failed to parse JSON config")
    } else {
        toml::from_str(&raw).context("failed to parse TOML config")
    }
}

fn print_catalog(kind: Option<ModelKind>, as_json: bool) -> Result<()> {
    let entries = catalog()
        .into_iter()
        .filter(|entry| kind.is_none_or(|kind| entry.kind == kind))
        .collect::<Vec<_>>();
    if as_json {
        println!("{}", serde_json::to_string_pretty(&entries)?);
    } else {
        for entry in entries {
            println!(
                "{:<32} {:<10} {}",
                entry.name,
                format!("{:?}", entry.kind),
                entry.note
            );
        }
    }
    Ok(())
}

fn print_env(tool: Tool, model: &str, _config: &AppConfig) -> Result<()> {
    match tool {
        Tool::Anthropic => {
            println!("export ANTHROPIC_BASE_URL='{}'", ANTHROPIC_BASE_URL);
            println!("export ANTHROPIC_AUTH_TOKEN='<ARK_API_KEY>'");
            println!("export ANTHROPIC_MODEL='{model}'");
            println!("export ANTHROPIC_DEFAULT_HAIKU_MODEL='{model}'");
            println!("export ANTHROPIC_DEFAULT_SONNET_MODEL='{model}'");
            println!("export ANTHROPIC_DEFAULT_OPUS_MODEL='{model}'");
            println!("export CLAUDE_CODE_SUBAGENT_MODEL='{model}'");
        }
        Tool::Openai => {
            println!("export OPENAI_BASE_URL='{}'", OPENAI_BASE_URL);
            println!("export OPENAI_API_KEY='<ARK_API_KEY>'");
            println!("export OPENAI_MODEL='{model}'");
        }
    }
    Ok(())
}

fn print_claude_settings(model: &str, _config: &AppConfig) -> Result<()> {
    let settings = json!({
        "env": {
            "ANTHROPIC_AUTH_TOKEN": "<ARK_API_KEY>",
            "ANTHROPIC_BASE_URL": ANTHROPIC_BASE_URL,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": model
        }
    });
    println!("{}", serde_json::to_string_pretty(&settings)?);
    Ok(())
}

fn chat_body(
    protocol: Protocol,
    model: &str,
    message: Option<&str>,
    messages_json: Option<&str>,
    system: Option<&str>,
    max_tokens: u32,
) -> Result<Value> {
    let messages = if let Some(messages_json) = messages_json {
        read_json_arg(messages_json)?
    } else {
        json!([{ "role": "user", "content": read_value_arg(message.unwrap_or("-"))? }])
    };
    match protocol {
        Protocol::Anthropic => {
            let mut body = json!({
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            });
            if let Some(system) = system {
                body["system"] = Value::String(system.to_string());
            }
            Ok(body)
        }
        Protocol::Openai => {
            let mut messages = messages;
            if let Some(system) = system {
                let mut array = match messages {
                    Value::Array(array) => array,
                    _ => bail!("messages_json must decode to an array"),
                };
                array.insert(0, json!({ "role": "system", "content": system }));
                messages = Value::Array(array);
            }
            Ok(json!({
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            }))
        }
    }
}

fn resolve_model(cli_model: Option<&str>, config: &AppConfig, kind: ModelKind) -> Result<String> {
    let model = cli_model
        .or(config.model.as_deref())
        .ok_or_else(|| anyhow!("missing model; pass --model, ARK_MODEL, or config model"))?;
    validate_model(model, kind)?;
    Ok(model.to_string())
}

fn resolve_resource_id(cli_resource_id: Option<&str>, config: &AppConfig) -> Result<String> {
    let resource_id = cli_resource_id
        .or(config.resource_id.as_deref())
        .unwrap_or("seed-tts-2.0");
    validate_model(resource_id, ModelKind::Speech)?;
    Ok(resource_id.to_string())
}

fn validate_model(name: &str, kind: ModelKind) -> Result<()> {
    if name.eq_ignore_ascii_case("auto") {
        bail!("Auto mode is not supported by these Ark Plan endpoints; choose a concrete model");
    }
    let found = catalog()
        .into_iter()
        .any(|entry| entry.kind == kind && entry.name == name);
    if !found {
        bail!("unsupported {kind:?} model/resource id: {name}");
    }
    Ok(())
}

fn endpoint_url(base_url: &str, endpoint: Endpoint, task_id: Option<&str>) -> String {
    match endpoint {
        Endpoint::TtsHttp => TTS_HTTP_URL.to_string(),
        Endpoint::TtsBidirectionalWs => TTS_BIDIRECTIONAL_WS_URL.to_string(),
        Endpoint::TtsUnidirectionalWs => TTS_UNIDIRECTIONAL_WS_URL.to_string(),
        _ => {
            let base = base_url.trim_end_matches('/');
            let path = match endpoint {
                Endpoint::AnthropicMessages => "/v1/messages",
                Endpoint::OpenaiChat => "/chat/completions",
                Endpoint::Embeddings => "/embeddings",
                Endpoint::Images => "/images/generations",
                Endpoint::VideoTasks => "/contents/generations/tasks",
                Endpoint::TtsHttp
                | Endpoint::TtsBidirectionalWs
                | Endpoint::TtsUnidirectionalWs => unreachable!(),
            };
            let mut url = format!("{base}{path}");
            if let Some(task_id) = task_id {
                url.push('/');
                url.push_str(task_id);
            }
            url
        }
    }
}

fn append_query_params(url: String, params: &[(String, String)]) -> String {
    if params.is_empty() {
        return url;
    }
    let query = params
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("&");
    let mut url = url;
    url.push('?');
    url.push_str(&query);
    url
}

fn protocol_for_endpoint(endpoint: Endpoint) -> Protocol {
    match endpoint {
        Endpoint::AnthropicMessages => Protocol::Anthropic,
        Endpoint::OpenaiChat
        | Endpoint::Embeddings
        | Endpoint::Images
        | Endpoint::VideoTasks
        | Endpoint::TtsHttp
        | Endpoint::TtsBidirectionalWs
        | Endpoint::TtsUnidirectionalWs => Protocol::Openai,
    }
}

fn default_base_url(protocol: Protocol) -> &'static str {
    match protocol {
        Protocol::Anthropic => ANTHROPIC_BASE_URL,
        Protocol::Openai => OPENAI_BASE_URL,
    }
}

fn read_json_arg(value: &str) -> Result<Value> {
    let raw = read_value_arg(value)?;
    serde_json::from_str(&raw).context("failed to parse JSON body")
}

fn read_value_arg(value: &str) -> Result<String> {
    if value == "-" {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .context("failed to read stdin")?;
        Ok(input)
    } else if let Some(path) = value.strip_prefix('@') {
        fs::read_to_string(path).with_context(|| format!("failed to read {path}"))
    } else {
        Ok(value.to_string())
    }
}

fn catalog() -> Vec<CatalogEntry> {
    vec![
        CatalogEntry {
            name: "doubao-seed-2.0-code",
            kind: ModelKind::Text,
            note: "text generation",
        },
        CatalogEntry {
            name: "doubao-seed-2.0-pro",
            kind: ModelKind::Text,
            note: "text generation",
        },
        CatalogEntry {
            name: "doubao-seed-2.0-lite",
            kind: ModelKind::Text,
            note: "text generation",
        },
        CatalogEntry {
            name: "doubao-seed-2.0-mini",
            kind: ModelKind::Text,
            note: "text generation",
        },
        CatalogEntry {
            name: "glm-5.2",
            kind: ModelKind::Text,
            note: "text generation",
        },
        CatalogEntry {
            name: "kimi-k2.7-code",
            kind: ModelKind::Text,
            note: "text generation",
        },
        CatalogEntry {
            name: "deepseek-v4-pro",
            kind: ModelKind::Text,
            note: "text generation",
        },
        CatalogEntry {
            name: "deepseek-v4-flash",
            kind: ModelKind::Text,
            note: "text generation",
        },
        CatalogEntry {
            name: "minimax-m3",
            kind: ModelKind::Text,
            note: "text generation",
        },
        CatalogEntry {
            name: "minimax-m2.7",
            kind: ModelKind::Text,
            note: "text generation",
        },
        CatalogEntry {
            name: "glm-5.1",
            kind: ModelKind::Text,
            note: "text generation",
        },
        CatalogEntry {
            name: "kimi-k2.6",
            kind: ModelKind::Text,
            note: "text generation",
        },
        CatalogEntry {
            name: "deepseek-v3.2",
            kind: ModelKind::Text,
            note: "text generation",
        },
        CatalogEntry {
            name: "doubao-embedding-vision",
            kind: ModelKind::Embedding,
            note: "embedding; no Auto/console switch",
        },
        CatalogEntry {
            name: "doubao-seedream-5.0-lite",
            kind: ModelKind::Image,
            note: "image generation",
        },
        CatalogEntry {
            name: "doubao-seedance-2.0",
            kind: ModelKind::Video,
            note: "video generation",
        },
        CatalogEntry {
            name: "doubao-seedance-2.0-fast",
            kind: ModelKind::Video,
            note: "video generation",
        },
        CatalogEntry {
            name: "doubao-seedance-2.0-mini",
            kind: ModelKind::Video,
            note: "video generation",
        },
        CatalogEntry {
            name: "doubao-seedance-1.5-pro",
            kind: ModelKind::Video,
            note: "video generation",
        },
        CatalogEntry {
            name: "seed-tts-2.0",
            kind: ModelKind::Speech,
            note: "TTS Resource-Id",
        },
        CatalogEntry {
            name: "volc.seedasr.sauc.duration",
            kind: ModelKind::Speech,
            note: "ASR Resource-Id",
        },
    ]
}

#[cfg(test)]
#[path = "lib_test.rs"]
mod tests;
