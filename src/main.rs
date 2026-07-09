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

// GGUFの値の型を定義する列挙型（低レイヤの仕様に基づく）
#[derive(Debug, PartialEq)]
#[repr(u32)]
enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

// バイトスライスからGGUF形式の文字列を読み出すヘルパー関数
fn read_string(mmap: &[u8], offset: &mut usize) -> Result<String, Box<dyn std::error::Error>> {
    // 1. 最初に8バイトの長さ(u64)を取得
    let len = u64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?) as usize;
    *offset += 8;

    // 2. その長さ分のバイトを文字列に変換
    let bytes = &mmap[*offset..*offset + len];
    let s = String::from_utf8_lossy(bytes).into_owned();
    *offset += len;

    Ok(s)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = Path::new(
        "/usr/share/ollama/.ollama/models/blobs/sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a",
    );
    let file = File::open(file_path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    let mut offset = 0;

    // ヘッダーのパース
    let magic = String::from_utf8_lossy(&mmap[offset..offset + 4]).into_owned();
    offset += 4;
    let version = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
    offset += 4;
    let tensor_count = u64::from_le_bytes(mmap[offset..offset + 8].try_into()?);
    offset += 8;
    let metadata_kv_count = u64::from_le_bytes(mmap[offset..offset + 8].try_into()?);
    offset += 8;

    println!("=== GGUF KV Metadata Analysis ===");

    // 55個のメタデータをパースしていくループ
    // (今回は複雑化を避けるため、String型と数値型のKVだけを綺麗に表示してみます)
    for i in 0..metadata_kv_count {
        if offset >= mmap.len() {
            break;
        }

        // キー名を読み込む
        let key = read_string(&mmap, &mut offset)?;

        // 値の型(u32)を読み込む
        let type_id = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
        offset += 4;

        // 型に応じて中身をパース
        match type_id {
            6 => {
                // Float32
                let value_float = f32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
                offset += 4;
                println!("    {}: {}", key, value_float);
            }
            8 => {
                // String型
                let value_str = read_string(&mmap, &mut offset)?;
                // 特に面白い重要なメタデータだけピックアップして表示、それ以外はシンプルに
                if key.contains("name") || key.contains("architecture") {
                    println!("[★] {}: {}", key, value_str);
                } else {
                    println!("    {}: {}", key, value_str);
                }
            }
            4 | 5 => {
                // Uint32 / Int32型
                let value_val = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
                offset += 4;
                println!("    {}: {}", key, value_val);
            }
            7 => {
                // Bool型
                let value_bool = mmap[offset] != 0;
                offset += 1;
                println!("    {}: {}", key, value_bool);
            }
            9 => {
                // Array型
                let _any_type = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
                offset += 4;
                let array_len = u64::from_le_bytes(mmap[offset..offset + 8].try_into()?) as usize;
                offset += 8;

                println!("    {}: [Array ({} elements)]", key, array_len);
                if i > 25 {
                    // 少し表示制限を緩めました
                    println!("... (以降の複雑な配列KVはパース省略) ...");
                    break;
                }
            }
            _ => {
                println!("    {}: [未知または未実装の型 ID:{}]", key, type_id);
                // 安全のため、未対応の型が来たら一旦ループを抜ける
                break;
            }
        }
    }

    Ok(())
}
