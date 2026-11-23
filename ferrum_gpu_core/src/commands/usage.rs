use super::*;

/// Usage text.
pub const USAGE: &str = "usage: ferrum-gpu <command> [flags]\n\
commands:\n  \
info          [--model F.gguf]\n  \
train         --data FILE --out F.gguf [--arch gpt2|llama] [--preset tiny|small|medium] [--steps N | --epochs N | --tokens N] [--lr X] [--batch N] [--context N] [--seed N] [--quant f32|f16|int8|int4|fp4|q2]\n                gpt2 only: --embed --heads --blocks --hidden --vocab --stride --weight_decay --cosine --warmup --clip --no-qat --tie --dropout RATE --vocab 0 (character-level bytes) --tokenizer F --checkpoint F --checkpoint-every N --resume F --val X --patience N --max-seconds X --metrics-csv F --verbose --sample\n  \
finetune-gguf --model F.gguf --data FILE --out G.gguf [same training flags]\n  \
run           --data FILE --model F.gguf --prompt TEXT [train and generate flags]   (trains F.gguf on FILE if it does not exist, then generates; a model that exists is loaded as is)\n  \
generate      --model F.gguf --prompt TEXT [--max-new N] [--temperature X] [--top-k N] [--seed N]\n                gpt2 files also: --top-p X --repetition-penalty X --stop S (repeatable) --stream --ids \"1 2 3\" (aliases --temp --gen-seed --max-tokens --chars)\n  \
run-gguf      same flags as generate; also prints the model's shape first\n  \
prepare       --input FILE|DIR (repeatable) --out DIR [--nfkc] [--lowercase] [--strip-gutenberg] [--dedup] [--min-chars N] [--max-chars N] [--val-frac X] [--test-frac X] [--seed N] [--doc-sep S] [--lossy]   (writes train.txt val.txt test.txt manifest.json)\n  \
tokenize      --data FILE|DIR --tokenizer TOK.gguf --out TOKENS.gguf [--doc-sep S]   (a token file trains with `train --data TOKENS.gguf --tokenizer TOK.gguf`, windows read from disk)\n  \
export-ollama --model F.gguf --name NAME [--bin P] [--modelfile-only OUT]   (GPT-2 files; needs the `ollama` tool)\n  \
export-lmstudio --model F.gguf [--repo publisher/name] [--bin P]   (GPT-2 files; needs the `lms` tool)\n  \
export-gguf   --model F.gguf --out G.gguf [--quant f32|f16|int8|int4|fp4|q2]   (re-write a model in another precision; reads Q4_1/Q5_*/Q8_1/K-quant files too)\n  \
eval          --model F.gguf --data FILE [--ctx N] [--quant f32|f16|int8|int4|fp4|q2] [--json F]   (alias: eval-gguf)\n  \
lab           [--gpu-adapter NAME] [--size small|medium] [--reps N] [--parity-only] [--json]   (GPU against the CPU reference, speed per precision, adapter facts; --json is the file to send back)\n\
device flags (info, train, finetune-gguf, generate, run-gguf): --device cpu|gpu|auto  --gpu-adapter NAME  --gpu-mem-mb N  --gpu-weights f32|f16|q8_0|q4_0|fp4|fp8|fp8e5m2|q2_k (how a GPU decoder stores weights; converted at load; fp8 is never written to a file)  --threads N";

pub(super) const DEVICE: [Flag; 5] = [
    val("--gpu-weights"),
    val("--device"),
    val("--gpu-adapter"),
    val("--threads"),
    val("--gpu-mem-mb"),
];
pub(super) const TRAIN: [Flag; 7] = [
    val("--data"),
    val("--out"),
    val("--steps"),
    val("--lr"),
    val("--batch"),
    val("--context"),
    val("--seed"),
];
pub(super) const GEN: [Flag; 5] = [
    val("--model"),
    val("--prompt"),
    val("--max-new"),
    val("--temperature"),
    val("--top-k"),
];

pub(super) fn table(parts: &[&[Flag]]) -> Vec<Flag> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

/// Adapter enumeration (EGL/Vulkan drivers) is not safe from several threads at once, and every
/// command lists adapters; tests run many commands in parallel, so they take turns.
#[cfg(test)]
pub(super) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Run `args` (without the program name), writing the report to `out`.
pub fn run(args: &[String], out: &mut dyn Write) -> Result<(), String> {
    #[cfg(test)]
    let _turn = crate::sync::lock(&ONE_AT_A_TIME);
    dispatch(args, out)
}
