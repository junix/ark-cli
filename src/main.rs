fn main() {
    if let Err(error) = ark_cli::run() {
        eprintln!("ark-cli: {error:#}");
        std::process::exit(1);
    }
}
