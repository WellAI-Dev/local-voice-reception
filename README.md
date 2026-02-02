# Local Voice Reception AI

ローカル環境で動作する音声AI受付システム。WellAIのサマリをナレッジベースとして活用し、完全オフラインで電話受付業務を支援します。

## 🎯 プロジェクト概要

### 目的
- **ローカル実行**: クラウドAPIを使用せず、月額費用を固定化
- **音声入出力**: 自然な日本語での対話
- **RAG統合**: WellAIサマリをナレッジとして回答生成
- **カスタム発音**: 自社名・固有名詞のイントネーション制御

### 主要技術スタック
| コンポーネント | 技術 | 役割 |
|--------------|------|------|
| 音声認識 (STT) | Vosk | オフライン音声→テキスト変換 |
| 音声合成 (TTS) | Qwen3-TTS | テキスト→音声変換（97ms低レイテンシ） |
| LLM | Qwen2.5/Gemma/etc | 回答生成（Ollama経由） |
| RAG | LangChain + ChromaDB | ナレッジ検索・文脈付与 |
| UI | Gradio | シンプルなWeb UI |

## 📁 ディレクトリ構造

```
local-voice-reception/
├── README.md                   # このファイル
├── Dockerfile                  # 本番用Dockerイメージ
├── docker-compose.yml          # Docker Compose設定
├── requirements.txt            # Python依存関係（ローカル）
├── requirements-docker.txt     # Python依存関係（Docker）
├── docs/                       # ドキュメント
│   ├── ARCHITECTURE.md         # システムアーキテクチャ
│   ├── SETUP_M4_MAC.md         # M4 Mac セットアップガイド
│   └── REQUIREMENTS.md         # 技術要件
├── scripts/                    # ユーティリティスクリプト
│   ├── setup_local.sh          # ローカル環境セットアップ
│   └── download_models.py      # モデルダウンロード
├── src/                        # ソースコード
│   ├── stt/                    # 音声認識モジュール (Vosk)
│   ├── tts/                    # 音声合成モジュール (Qwen3-TTS)
│   ├── llm/                    # LLMインターフェース (Ollama)
│   ├── utils/                  # ユーティリティ
│   └── app.py                  # メインアプリケーション
├── models/                     # モデルファイル（Git LFS）
├── data/
│   ├── knowledge/              # ナレッジベース（Markdown）
│   └── voice_samples/          # 音声クローン用サンプル
└── config/
    ├── pronunciation_dict.yaml # 発音辞書
    └── settings.yaml           # アプリ設定
```

## 🖥️ 動作環境

| 環境 | デバイス | 用途 |
|-----|---------|-----|
| ローカル開発 | Apple Silicon (M4 Max) + MPS | 高速開発・テスト |
| Docker CPU | Any | ポータブル、低速 |
| Docker CUDA | NVIDIA GPU | 本番デプロイ |

- **Python**: 3.11+
- **メモリ**: 32GB以上推奨
- **ストレージ**: 20GB以上（モデル格納用）

## 🚀 クイックスタート

### Option A: ローカル開発（Apple Silicon推奨）

```bash
# 1. セットアップスクリプト実行
cd local-voice-reception
./scripts/setup_local.sh

# 2. モデルダウンロード
source .venv/bin/activate
python scripts/download_models.py

# 3. Ollamaモデル取得
ollama pull qwen2.5:7b

# 4. アプリケーション起動
python src/app.py
```

ブラウザで http://localhost:7860 を開いてください。

### Option B: Docker（本番デプロイ）

```bash
# CPU版（ポータブル）
docker compose --profile cpu up -d

# CUDA版（NVIDIA GPU）
docker compose --profile cuda up -d

# Ollamaも一緒に起動
docker compose up ollama -d
docker exec ollama ollama pull qwen2.5:7b
```

## 📊 システム構成図

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   マイク    │────▶│    Vosk     │────▶│  テキスト   │
│   入力      │     │    STT      │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────┐
│                    RAG パイプライン                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐    │
│  │ WellAI   │──▶│ ChromaDB │──▶│   ローカルLLM    │    │
│  │ ナレッジ │   │  検索    │   │   (Qwen2.5)      │    │
│  └──────────┘   └──────────┘   └────────┬─────────┘    │
└────────────────────────────────────────┬────────────────┘
                                         │
                                         ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ スピーカー  │◀────│  Qwen3-TTS  │◀────│  回答文     │
│   出力      │     │    TTS      │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

## ⚙️ デバイス別設定

| デバイス | dtype | Attention | 備考 |
|---------|-------|-----------|------|
| Apple Silicon (MPS) | float32 | sdpa | FlashAttention非対応 |
| NVIDIA CUDA | bfloat16 | flash_attention_2 | 最高速 |
| CPU | float32 | sdpa | 低速だがポータブル |

設定は `config/settings.yaml` の `tts.device: "auto"` で自動検出されます。

## 🔧 設定ファイル

### config/settings.yaml

主要な設定項目:

```yaml
# STT設定
stt:
  model_path: "models/vosk/vosk-model-small-ja-0.22"

# TTS設定
tts:
  model_name: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  device: "auto"  # mps / cuda / cpu

# LLM設定
llm:
  ollama:
    model: "qwen2.5:7b"
```

### config/pronunciation_dict.yaml

発音辞書で固有名詞の読みを制御:

```yaml
terms:
  - original: "Cor.Inc"
    reading: "コア インク"
```

## 📝 ライセンス

- Vosk: Apache 2.0
- Qwen3-TTS: Apache 2.0
- 本プロジェクト: MIT

## 🔗 参考リンク

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Vosk Speech Recognition](https://alphacephei.com/vosk/)
- [Ollama](https://ollama.ai/)
- [Gradio](https://gradio.app/)

---

**Cor.Inc** - Communication beyond words
