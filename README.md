# Local Voice Reception AI

ローカル環境で動作する音声AI受付システム。WellAIのサマリをナレッジベースとして活用し、完全オフラインで電話受付業務を支援します。

## プロジェクト概要

### 目的
- **ローカル実行**: クラウドAPIを使用せず、月額費用を固定化
- **音声入出力**: 自然な日本語での対話
- **RAG統合**: WellAIサマリをナレッジとして回答生成
- **音声クローン**: 自社の声をリアルタイムでクローン再現
- **STT辞書**: 音声認識の誤変換を自動補正

### 主要技術スタック

| コンポーネント | 技術 | 役割 |
|--------------|------|------|
| 音声認識 (STT) | Vosk | オフライン音声→テキスト変換 |
| 音声合成 (TTS) | Qwen3-TTS (dual-mode) | テキスト→音声変換 |
| LLM | Qwen2.5/Gemma/etc | 回答生成（Ollama経由） |
| RAG | LangChain + ChromaDB | ナレッジ検索・文脈付与 |
| UI | Gradio | Web UI（会話・辞書管理・ナレッジ管理・音声登録） |
| ツーリング | mise + uv | タスクランナー + パッケージ管理 |

## ディレクトリ構造

```
local-voice-reception/
├── README.md
├── pyproject.toml              # Python 依存関係（uv で管理）
├── uv.lock                     # ロックファイル
├── .mise.toml                  # mise タスク定義
├── Dockerfile
├── docker-compose.yml
├── docs/                       # ドキュメント
│   ├── ARCHITECTURE.md         # システムアーキテクチャ
│   ├── SETUP_M4_MAC.md         # M4 Mac セットアップガイド
│   ├── REQUIREMENTS.md         # 技術要件
│   ├── TTS_COMPARISON.md       # TTS 比較調査（アーカイブ）
│   └── UI_IMPROVEMENT_PROPOSAL.md  # UI 改善提案（一部実装済み）
├── src/                        # ソースコード
│   ├── app.py                  # メインアプリケーション（Gradio UI）
│   ├── stt/                    # 音声認識モジュール (Vosk)
│   │   ├── vosk_stt.py         # Vosk STT ラッパー
│   │   └── dictionary.py       # STT 辞書（誤認識補正）
│   ├── tts/                    # 音声合成モジュール (Qwen3-TTS)
│   │   └── qwen_tts.py         # dual-mode TTS（custom_voice / voice_clone）
│   ├── llm/                    # LLM インターフェース
│   │   ├── ollama_client.py    # Ollama クライアント
│   │   └── knowledge_manager.py # ナレッジベース管理（RAG）
│   └── utils/                  # ユーティリティ
├── models/                     # モデルファイル
├── data/
│   ├── knowledge/              # ナレッジベース（Markdown）
│   └── voice_samples/          # 音声クローン用サンプル
├── config/
│   ├── settings.yaml           # アプリ設定
│   ├── pronunciation_dict.yaml # 発音辞書（TTS 前処理用）
│   └── stt_dictionary.yaml     # STT 辞書（認識後処理用）
└── tests/                      # テストコード
```

## 動作環境

| 環境 | デバイス | 用途 |
|-----|---------|-----|
| ローカル開発 | Apple Silicon (M4 Max) + MPS | 高速開発・テスト |
| Docker CPU | Any | ポータブル、低速 |
| Docker CUDA | NVIDIA GPU | 本番デプロイ |

- **Python**: 3.12（mise で自動管理）
- **メモリ**: 32GB以上推奨
- **ストレージ**: 20GB以上（モデル格納用）

## クイックスタート

### 前提条件

```bash
# mise のインストール（未導入の場合）
curl https://mise.run | sh

# Homebrew パッケージ
brew install portaudio ffmpeg ollama
```

### Option A: ローカル開発（Apple Silicon推奨）

```bash
# 1. リポジトリをクローン
git clone https://github.com/WellAI-Dev/local-voice-reception.git
cd local-voice-reception

# 2. 依存関係のインストール（mise + uv）
mise install        # Python 3.12 をセットアップ
mise run install    # uv sync + vosk (optional)

# 3. Ollama モデル取得
ollama pull qwen2.5:7b

# 4. アプリケーション起動
mise run run
```

ブラウザで http://localhost:7860 を開いてください。

### Option B: Docker（本番デプロイ）

```bash
# CPU版（ポータブル）
docker compose --profile cpu up -d

# CUDA版（NVIDIA GPU）
docker compose --profile cuda up -d

# Ollama も一緒に起動
docker compose up ollama -d
docker exec ollama ollama pull qwen2.5:7b
```

## mise タスク一覧

| コマンド | 説明 |
|---------|------|
| `mise run install` | 依存関係インストール（uv sync） |
| `mise run install-dev` | 開発用依存関係込みでインストール |
| `mise run run` | アプリケーション起動 |
| `mise run test` | 全テスト実行 |
| `mise run test-cov` | カバレッジ付きテスト |
| `mise run lint` | Ruff でリント |
| `mise run format` | Ruff でフォーマット |
| `mise run clean` | ビルドキャッシュ削除 |

## TTS デュアルモード

Qwen3-TTS は2つのモードをサポートし、ランタイムで切り替え可能です:

| モード | モデル | 用途 |
|--------|--------|------|
| `custom_voice` | CustomVoice (1.7B) | プリセット話者（ono_anna 等）で生成 |
| `voice_clone` | Base (1.7B) | 録音した声をクローンして生成 |

UI の「音声クローン設定」タブから声を録音すると、自動的に `voice_clone` モードに切り替わります。

## UI 機能

Gradio Web UI では以下の機能を提供:

- **会話タブ**: Push-to-Talk での音声対話、Chatbot 形式の会話履歴
- **STT辞書タブ**: Vosk 誤認識の補正ルール管理（exact / regex）
- **ナレッジ管理タブ**: RAG 用ナレッジベースの追加・削除・再構築
- **音声クローン設定タブ**: 声の録音・登録、モード切り替え
- **設定タブ**: TTS/LLM パラメータの調整

## システム構成図

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   マイク    │────>│  Vosk STT   │────>│  STT辞書     │
│   入力      │     │             │     │  (補正処理)  │
└─────────────┘     └─────────────┘     └──────┬───────┘
                                               │
                                               v
┌──────────────────────────────────────────────────────────┐
│                    RAG パイプライン                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐     │
│  │ WellAI   │-->│ ChromaDB │-->│   ローカルLLM    │     │
│  │ ナレッジ │   │  検索    │   │   (Qwen2.5)      │     │
│  └──────────┘   └──────────┘   └────────┬─────────┘     │
└─────────────────────────────────────────┬────────────────┘
                                          │
                                          v
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ スピーカー  │<────│  Qwen3-TTS  │<────│  回答文     │
│   出力      │     │  (dual-mode)│     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

## デバイス別設定

| デバイス | dtype | Attention | 備考 |
|---------|-------|-----------|------|
| Apple Silicon (MPS) | float32 | sdpa | FlashAttention非対応 |
| NVIDIA CUDA | bfloat16 | flash_attention_2 | 最高速 |
| CPU | float32 | sdpa | 低速だがポータブル |

設定は `config/settings.yaml` の `tts.device: "auto"` で自動検出されます。

## 設定ファイル

### config/settings.yaml

主要な設定項目:

```yaml
# STT設定
stt:
  model_path: "models/vosk/vosk-model-small-ja-0.22"

# TTS設定（デュアルモード）
tts:
  device: "auto"
  mode: "custom_voice"  # custom_voice / voice_clone
  custom_voice:
    speaker: "ono_anna"
    language: "Japanese"
  voice_clone:
    ref_audio: "data/voice_samples/company_voice.wav"
    ref_text: "..."

# LLM設定
llm:
  ollama:
    model: "qwen2.5:7b"
```

### config/stt_dictionary.yaml

STT辞書で音声認識の誤変換を補正:

```yaml
corrections:
  - wrong: "うぇるあい"
    correct: "WellAI"
    note: "会社名"

patterns:
  - pattern: "おでんわ"
    replacement: "お電話"
```

### config/pronunciation_dict.yaml

発音辞書で固有名詞の読みを制御:

```yaml
terms:
  - original: "WellAI"
    reading: "ウェルアイ"
```

## テスト

```bash
# 全テスト実行
mise run test

# カバレッジ付き
mise run test-cov

# 特定テスト
uv run pytest tests/test_app.py -v
```

## ライセンス

- Vosk: Apache 2.0
- Qwen3-TTS: Apache 2.0
- 本プロジェクト: MIT

## 参考リンク

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Vosk Speech Recognition](https://alphacephei.com/vosk/)
- [Ollama](https://ollama.ai/)
- [Gradio](https://gradio.app/)
- [mise](https://mise.jdx.dev/)
- [uv](https://docs.astral.sh/uv/)

---

**WellAI** - AI for Well-being
