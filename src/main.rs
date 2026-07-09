use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

#[derive(Debug)]
struct GgufHeader {
    magic: String,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = Path::new(
        "/usr/share/ollama/.ollama/models/blobs/sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a",
    );

    println!("Opening model file: {:?}", file_path);
    let file = File::open(file_path)?;

    let mmap = unsafe { Mmap::map(&file)? };
    println!(
        "File mmapped successfully. Total size: {} bytes",
        mmap.len()
    );

    if mmap.len() < 24 {
        panic!("File is too small to be a valid GGUF file.");
    }

    let mut offset = 0;

    let magic_bytes = &mmap[offset..offset + 4];
    let magic = String::from_utf8_lossy(magic_bytes).into_owned();
    offset += 4;

    let version = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
    offset += 4;

    let tensor_count = u64::from_le_bytes(mmap[offset..offset + 8].try_into()?);
    offset += 8;

    let metadata_kv_count = u64::from_le_bytes(mmap[offset..offset + 8].try_into()?);
    offset += 8;

    let header = GgufHeader {
        magic,
        version,
        tensor_count,
        metadata_kv_count,
    };

    println!("\n=== GGUF Header Analysis ===");
    println!("Magic Number    : {}", header.magic);
    println!("GGUF Version    : {}", header.version);
    println!(
        "Tensor Count    : {} (この数だけの巨大な行列が眠っています)",
        header.tensor_count
    );
    println!(
        "Metadata KV Count: {} (モデル名や設定パラメーターの数)",
        header.metadata_kv_count
    );
    println!("============================\n");

    if header.magic != "GGUF" {
        println!(
            "[警告] マジックナンバーが 'GGUF' ではありません！正しいGGUFファイルか確認してください。"
        );
    } else {
        println!("[成功] 正真正銘のGGUFファイルです。低レイヤハックの準備は完了しました。");
        println!(
            "次のオフセット位置 ({} バイト目) からメタデータとテンソルの解析が始まります。",
            offset
        );
    }

    Ok(())
}
