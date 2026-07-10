# ark-cli - Volcengine Ark Agent/Coding Plan helper

set shell := ["bash", "-euo", "pipefail", "-c"]
os_suffix := if os() == "macos" { "macos" } else { "linux" }
arch_suffix := if arch() == "aarch64" { "arm64" } else { "x86" }
install_bin := env("SYNC_BIN_DIR", home_directory() / "sync" / (os_suffix + "-" + arch_suffix + "-bin"))
target_dir := env("CARGO_TARGET_DIR", justfile_directory() / "target")

default: build

build:
    cargo build --release

test:
    cargo test

install: build
    mkdir -p "{{ install_bin }}"
    cp "{{ target_dir }}/release/ark-cli" "{{ install_bin }}/ark-cli"
    xattr -c "{{ install_bin }}/ark-cli" 2>/dev/null || true
    codesign -f -s - "{{ install_bin }}/ark-cli" 2>/dev/null || true
    echo "Installed ark-cli to {{ install_bin }}/ark-cli"
