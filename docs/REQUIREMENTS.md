# 技術要件

## 概要

本ドキュメントでは、Local Voice Reception AIの技術要件と依存関係を定義します。

## ハードウェア要件

### 最小構成

| 項目 | 要件 |
|------|------|
| CPU/チップ | Apple M1 / Intel Core i7 以上 |
| メモリ | 16GB |
| ストレージ | 30GB SSD |
| オーディオ | マイク入力、スピーカー出力 |

### 推奨構成（本プロジェクト）

| 項目 | 要件 |
|------|------|
| チップ | Apple M4 Max |
| メモリ | 64GB (Unified Memory) |
| ストレージ | 50GB以上 |
| オーディオ | 高品質マイク（USB推奨） |

### ストレージ内訳

| 項目 | サイズ |
|------|--------|
| Voskモデル（日本語高精度） | 1GB |
| Qwen3-TTSモデル（1.7B） | ~3.5GB |
| LLMモデル（Qwen2.5 7B） | ~4.4GB |
| ChromaDBベクトルストア | ~500MB（ナレッジ量による） |
| その他（キャッシュ等） | ~2GB |
| **合計** | **~12GB** |

## ソフトウェア要件

### OS

| OS | バージョン | 対応状況 |
|----|-----------|---------|
| macOS (Apple Silicon) | 14.0+ | ✅ 推奨 |
| macOS (Intel) | 13.0+ | ⚠️ 動作可能 |
| Ubuntu | 22.04+ | ⚠️ 要検証 |
| Windows | 11 | ⚠️ 要検証 |

### Python

- **バージョン**: 3.11以上
- **パッケージマネージャ**: pip / poetry

## 依存パッケージ

### コア依存関係

```
# requirements.txt

# === Core ===
numpy>=1.24.0,<2.0.0
scipy>=1.11.0

# === Audio Processing ===
sounddevice>=0.4.6
soundfile>=0.12.1
pyaudio>=0.2.14

# === Speech Recognition (STT) ===
vosk>=0.3.45

# === Text-to-Speech (TTS) ===
qwen-tts>=0.1.0
torch>=2.1.0
torchaudio>=2.1.0
transformers>=4.36.0

# === LLM & RAG ===
langchain>=0.1.0
langchain-community>=0.0.20
chromadb>=0.4.22
sentence-transformers>=2.2.2
ollama>=0.1.6

# === Embedding Models ===
# intfloat/multilingual-e5-small (HuggingFace経由)

# === Web UI ===
gradio>=4.19.0

# === Utilities ===
pyyaml>=6.0.1
python-dotenv>=1.0.0
tqdm>=4.66.0
rich>=13.0.0
```

### 開発依存関係

```
# requirements-dev.txt

pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
isort>=5.12.0
mypy>=1.5.0
ruff>=0.1.0
```

## 外部サービス・ツール

### 必須

| ツール | 用途 | インストール |
|--------|------|-------------|
| Ollama | ローカルLLM実行 | `brew install ollama` |
| Homebrew | パッケージ管理 | 公式サイト参照 |
| Git LFS | 大容量ファイル管理 | `brew install git-lfs` |

### オプション

| ツール | 用途 | インストール |
|--------|------|-------------|
| Docker | コンテナ化 | Docker Desktop |
| FFmpeg | 音声変換 | `brew install ffmpeg` |

## モデル仕様

### Vosk 日本語モデル

| モデル | サイズ | 精度 | 用途 |
|--------|--------|------|------|
| vosk-model-small-ja-0.22 | 48MB | 中 | テスト・軽量環境 |
| vosk-model-ja-0.22 | 1GB | 高 | 本番環境 |

**ダウンロード元**: https://alphacephei.com/vosk/models

### Qwen3-TTS モデル

| モデル | パラメータ | 用途 |
|--------|-----------|------|
| Qwen3-TTS-12Hz-0.6B-Base | 0.6B | 軽量・高速 |
| Qwen3-TTS-12Hz-1.7B-Base | 1.7B | 高品質・音声クローン |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 1.7B | プリセット音声 |

**ダウンロード元**: https://huggingface.co/Qwen

### LLMモデル（Ollama経由）

| モデル | サイズ | 特徴 |
|--------|--------|------|
| qwen2.5:7b | 4.4GB | バランス型、日本語良好 |
| qwen2.5:3b | 2.0GB | 軽量版 |
| gemma2:9b | 5.5GB | 高品質 |
| gemma2:2b | 1.6GB | 超軽量 |
| deepseek-r1:7b | 4.7GB | 推論特化 |

## API仕様

### 内部API

```python
# STT Module API
class STTModule:
    def __init__(self, model_path: str) -> None: ...
    def recognize(self, audio_data: bytes) -> str: ...
    def stream_recognize(self, audio_stream) -> Generator[str, None, None]: ...

# TTS Module API
class TTSModule:
    def __init__(self, model_name: str, voice_config: dict) -> None: ...
    def synthesize(self, text: str) -> np.ndarray: ...
    def synthesize_stream(self, text: str) -> Generator[np.ndarray, None, None]: ...

# RAG Pipeline API
class RAGPipeline:
    def __init__(self, knowledge_dir: str) -> None: ...
    def retrieve(self, query: str) -> List[Document]: ...
    def build_context(self, query: str, documents: List[Document]) -> str: ...

# LLM Interface API
class LLMInterface:
    def __init__(self, model_name: str) -> None: ...
    def generate(self, prompt: str, context: str) -> str: ...
    def generate_stream(self, prompt: str, context: str) -> Generator[str, None, None]: ...
```

### 設定ファイル形式

```yaml
# config/settings.yaml

app:
  name: "Local Voice Reception AI"
  version: "0.1.0"
  debug: false

stt:
  model_path: "models/vosk/vosk-model-ja-0.22"
  sample_rate: 16000
  chunk_size: 8000

tts:
  model_name: "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
  device: "mps"  # mps / cuda / cpu
  dtype: "float16"
  speaker: "Ono Anna"
  ref_audio: "data/voice_samples/company_voice.wav"
  ref_text: "お電話ありがとうございます。コア株式会社でございます。"

rag:
  knowledge_dir: "data/knowledge"
  embedding_model: "intfloat/multilingual-e5-small"
  chunk_size: 500
  chunk_overlap: 50
  top_k: 3

llm:
  provider: "ollama"
  model: "qwen2.5:7b"
  temperature: 0.7
  max_tokens: 512

ui:
  host: "127.0.0.1"
  port: 7860
  share: false
```

## パフォーマンス要件

### レイテンシ目標

| 処理 | 目標 | 備考 |
|------|------|------|
| 音声認識 (STT) | <500ms | ストリーミング時 |
| ベクトル検索 (RAG) | <100ms | top_k=3 |
| LLM推論 | <3s | 初回トークン |
| 音声合成 (TTS) | <200ms | 初回音声パケット |
| **エンドツーエンド** | **<5s** | 質問→回答完了 |

### 同時接続

- **現行設計**: 1（シングルユーザー）
- **将来拡張**: N（マルチユーザー対応予定）

## セキュリティ要件

1. **データの外部送信禁止**: すべての処理がローカルで完結
2. **機密データの保護**: ナレッジベースへのアクセス制御（将来実装）
3. **音声データの非永続化**: デフォルトでは音声を保存しない
4. **ログの匿名化**: 個人を特定可能な情報のマスキング

## テスト要件

### 単体テスト

```bash
pytest tests/unit/ -v
```

### 統合テスト

```bash
pytest tests/integration/ -v
```

### E2Eテスト

```bash
pytest tests/e2e/ -v
```

### カバレッジ目標

- 単体テスト: 80%以上
- 統合テスト: 主要フローをカバー

## 監視・ログ

### ログ形式

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### メトリクス（将来実装）

- 応答時間
- 認識精度
- エラー率
- メモリ使用量
