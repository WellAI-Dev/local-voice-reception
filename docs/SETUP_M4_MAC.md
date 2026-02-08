# M4 Max MacBook Pro セットアップガイド

## 概要

このガイドでは、M4 Max MacBook Pro（Apple Silicon / ARM64）環境での
Local Voice Reception AI のセットアップ手順を説明します。

## 前提条件

| 項目 | 要件 |
|------|------|
| OS | macOS Sonoma 15.0以上 |
| チップ | Apple M4 Max |
| メモリ | 32GB以上推奨（64GB理想） |
| ストレージ | 50GB以上の空き容量 |
| Python | 3.12（mise で自動管理） |
| Homebrew | インストール済み |

## 1. ツーリングのセットアップ

### 1.1 mise のインストール

```bash
# mise をインストール（タスクランナー + ランタイム管理）
curl https://mise.run | sh

# シェルに追加（~/.zshrc に追記）
echo 'eval "$(mise activate zsh)"' >> ~/.zshrc
source ~/.zshrc

# 確認
mise --version
```

### 1.2 Homebrew パッケージのインストール

```bash
# オーディオ関連
brew install portaudio
brew install ffmpeg

# 開発ツール
brew install git-lfs

# ローカルLLM
brew install ollama
```

## 2. プロジェクトセットアップ

### 2.1 リポジトリのクローン

```bash
git clone https://github.com/WellAI-Dev/local-voice-reception.git
cd local-voice-reception
```

### 2.2 Python + 依存関係のインストール

```bash
# mise が .mise.toml を読み、Python 3.12 を自動インストール
mise install

# 依存関係のインストール（uv sync + vosk optional）
mise run install

# 開発用依存関係も含める場合
mise run install-dev
```

> **Note**: `uv` は mise 経由で自動的に利用されます。手動インストール不要です。
> Vosk は macOS ARM64 ではホイールが提供されていないため、インストールに失敗しても動作に影響しません（optional dependency）。

### 2.3 PyAudio のトラブルシューティング

PyAudio のインストールでエラーが発生する場合:

```bash
# portaudio のパスを指定してインストール
CFLAGS="-I$(brew --prefix portaudio)/include" \
LDFLAGS="-L$(brew --prefix portaudio)/lib" \
uv pip install pyaudio
```

## 3. モデルのダウンロード

### 3.1 Vosk モデル（音声認識）

```bash
# モデルディレクトリの作成
mkdir -p models/vosk

# 軽量版（テスト用）
cd models/vosk
curl -LO https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip
unzip vosk-model-small-ja-0.22.zip
rm vosk-model-small-ja-0.22.zip

# 高精度版（本番用）
curl -LO https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip
unzip vosk-model-ja-0.22.zip
rm vosk-model-ja-0.22.zip

cd ../..
```

### 3.2 Qwen3-TTS モデル（音声合成）

Qwen3-TTS はモード設定に応じて Hugging Face から自動ダウンロードされます。
手動でダウンロードする場合:

```bash
# Hugging Face CLI のインストール
uv pip install -U "huggingface_hub[cli]"

# Tokenizer モデル
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --local-dir models/qwen-tts/Qwen3-TTS-Tokenizer-12Hz

# CustomVoice モデル（プリセット音声用）
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --local-dir models/qwen-tts/Qwen3-TTS-12Hz-1.7B-CustomVoice

# Base モデル（音声クローン用）
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --local-dir models/qwen-tts/Qwen3-TTS-12Hz-1.7B-Base
```

### 3.3 Ollama のセットアップ（ローカルLLM）

```bash
# Ollama サービスの起動
ollama serve &

# 推奨モデルのダウンロード
ollama pull qwen2.5:7b        # バランス型
ollama pull gemma2:2b         # 軽量版（テスト用）

# モデルの確認
ollama list
```

## 4. M4 Max 固有の最適化

### 4.1 Metal Performance Shaders (MPS) の設定

本プロジェクトでは `config/settings.yaml` の `tts.device: "auto"` により
MPS を自動検出します。手動で確認するには:

```python
import torch

print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

### 4.2 Qwen3-TTS の MPS 対応

- `flash-attn` は Apple Silicon では利用不可。代わりに `sdpa` を自動使用
- dtype は `float32` が使用されます（MPS は bfloat16 非対応）
- これらの設定はすべて `QwenTTS` クラス内で自動処理されます

### 4.3 メモリ最適化

```bash
# 環境変数の設定（.env ファイルまたはシェル設定）
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

## 5. 動作確認

### 5.1 テストの実行

```bash
# 全テスト実行
mise run test

# カバレッジ付き
mise run test-cov
```

### 5.2 アプリケーションの起動

```bash
# アプリケーション起動
mise run run
```

ブラウザで http://localhost:7860 を開き、以下を確認:

1. **会話タブ**: Push-to-Talk ボタンが表示される
2. **STT辞書タブ**: 辞書の確認・編集ができる
3. **ナレッジ管理タブ**: ナレッジベースの管理ができる
4. **音声クローン設定タブ**: 声の録音・登録ができる

### 5.3 Ollama の動作確認

```bash
# CLI で直接テスト
ollama run qwen2.5:7b "こんにちは、簡単な自己紹介をしてください。"
```

## 6. トラブルシューティング

### 6.1 よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| `MPS backend out of memory` | VRAM が不足 | `torch.mps.empty_cache()` を呼び出す |
| `No module named 'vosk'` | vosk がインストールされていない | macOS ARM64 では optional（STT 無しで動作可） |
| `PortAudio not found` | portaudio がない | `brew install portaudio` |
| `Ollama connection refused` | Ollama が起動していない | `ollama serve` で起動 |
| `mise: command not found` | mise がパスにない | `eval "$(mise activate zsh)"` を実行 |

### 6.2 M4 Max 特有の問題

**flash-attn エラー**:
Apple Silicon では flash-attn が利用不可ですが、`QwenTTS` クラスが自動的に `sdpa` にフォールバックします。

**MPS メモリエラー**:
```bash
# メモリ制限を無効化
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

## 7. 推奨設定まとめ

`config/settings.yaml` のデフォルト設定は M4 Max 向けに最適化されています:

```yaml
tts:
  device: "auto"          # MPS を自動検出
  mode: "custom_voice"    # プリセット話者で開始
  preload: true           # 起動時にモデルをプリロード
  custom_voice:
    speaker: "ono_anna"
    language: "Japanese"

llm:
  ollama:
    model: "qwen2.5:7b"   # バランス型

stt:
  model_path: "models/vosk/vosk-model-small-ja-0.22"
```

## 次のステップ

セットアップが完了したら、以下のドキュメントを参照してください:

1. [ARCHITECTURE.md](./ARCHITECTURE.md) - システムアーキテクチャの詳細
2. [REQUIREMENTS.md](./REQUIREMENTS.md) - 技術要件の詳細
3. [../README.md](../README.md) - アプリケーションの起動方法
