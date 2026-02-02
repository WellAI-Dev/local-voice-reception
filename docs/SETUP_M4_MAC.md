# M4 Max MacBook Pro セットアップガイド

## 概要

このガイドでは、M4 Max MacBook Pro（Apple Silicon / ARM64）環境での
Local Voice Reception AIのセットアップ手順を説明します。

## 前提条件

| 項目 | 要件 |
|------|------|
| OS | macOS Sonoma 15.0以上 |
| チップ | Apple M4 Max |
| メモリ | 32GB以上推奨（64GB理想） |
| ストレージ | 50GB以上の空き容量 |
| Python | 3.11以上 |
| Homebrew | インストール済み |

## 1. 基本環境のセットアップ

### 1.1 Homebrewパッケージのインストール

```bash
# オーディオ関連
brew install portaudio
brew install ffmpeg
brew install sox

# 開発ツール
brew install cmake
brew install git-lfs
```

### 1.2 Python環境の構築

```bash
# pyenv経由でPythonをインストール（推奨）
brew install pyenv
pyenv install 3.11.8
pyenv local 3.11.8

# または、システムPythonを使用
python3 --version  # 3.11以上を確認
```

### 1.3 プロジェクト仮想環境の作成

```bash
cd /path/to/local-voice-reception

# 仮想環境の作成
python -m venv .venv

# 仮想環境の有効化
source .venv/bin/activate

# pipのアップグレード
pip install --upgrade pip
```

## 2. 依存パッケージのインストール

### 2.1 requirements.txtの作成

```bash
cat > requirements.txt << 'EOF'
# Core
numpy>=1.24.0
scipy>=1.11.0

# Audio Processing
sounddevice>=0.4.6
soundfile>=0.12.1
pyaudio>=0.2.14

# Speech Recognition (STT)
vosk>=0.3.45

# Text-to-Speech (TTS) - Qwen3-TTS
qwen-tts>=0.1.0
torch>=2.1.0
torchaudio>=2.1.0

# LLM & RAG
langchain>=0.1.0
langchain-community>=0.0.20
chromadb>=0.4.22
sentence-transformers>=2.2.2
ollama>=0.1.6

# Web UI
gradio>=4.19.0

# Utilities
pyyaml>=6.0.1
python-dotenv>=1.0.0
tqdm>=4.66.0
EOF
```

### 2.2 依存パッケージのインストール

```bash
# 基本パッケージ
pip install -r requirements.txt

# PyTorch (Apple Silicon最適化版)
pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 注意: M4 MaxではMPS (Metal Performance Shaders) を使用
```

### 2.3 PyAudioのトラブルシューティング

PyAudioのインストールでエラーが発生する場合:

```bash
# portaudioのパスを指定してインストール
CFLAGS="-I$(brew --prefix portaudio)/include" \
LDFLAGS="-L$(brew --prefix portaudio)/lib" \
pip install pyaudio
```

## 3. モデルのダウンロード

### 3.1 Voskモデル（音声認識）

```bash
# モデルディレクトリの作成
mkdir -p models/vosk

# 日本語モデル（高精度版）のダウンロード
cd models/vosk
curl -LO https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip
unzip vosk-model-ja-0.22.zip
rm vosk-model-ja-0.22.zip

# 軽量版（テスト用）
curl -LO https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip
unzip vosk-model-small-ja-0.22.zip
rm vosk-model-small-ja-0.22.zip

cd ../..
```

### 3.2 Qwen3-TTSモデル（音声合成）

```bash
# Hugging Face CLIのインストール
pip install -U "huggingface_hub[cli]"

# モデルのダウンロード
mkdir -p models/qwen-tts

# Tokenizerモデル
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --local-dir models/qwen-tts/Qwen3-TTS-Tokenizer-12Hz

# Base モデル（音声クローン用）
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --local-dir models/qwen-tts/Qwen3-TTS-12Hz-1.7B-Base

# CustomVoice モデル（プリセット音声用）
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --local-dir models/qwen-tts/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

### 3.3 Ollamaのセットアップ（ローカルLLM）

```bash
# Ollamaのインストール
brew install ollama

# Ollamaサービスの起動
ollama serve &

# 推奨モデルのダウンロード
ollama pull qwen2.5:7b        # バランス型
ollama pull gemma2:2b         # 軽量版（テスト用）

# モデルの確認
ollama list
```

## 4. M4 Max固有の最適化

### 4.1 Metal Performance Shaders (MPS) の設定

```python
# PyTorchでMPSを使用する設定
import torch

# MPSが利用可能か確認
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# デバイスの設定
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

### 4.2 Qwen3-TTS のMPS対応

```python
from qwen_tts import Qwen3TTSModel

# M4 Max用の設定
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="mps",           # Apple Silicon用
    dtype=torch.float16,        # 半精度（メモリ節約）
    # attn_implementation="sdpa"  # flash-attnの代わり
)
```

**注意**: `flash-attn` はApple Siliconでは利用不可。代わりに `sdpa` を使用。

### 4.3 メモリ最適化

```python
import os

# 環境変数の設定（.envファイルまたはシェル設定）
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # メモリ制限を無効化

# メモリキャッシュのクリア
torch.mps.empty_cache()
```

## 5. 動作確認

### 5.1 Voskの動作確認

```python
# test_vosk.py
import vosk
import json

model = vosk.Model("models/vosk/vosk-model-ja-0.22")
recognizer = vosk.KaldiRecognizer(model, 16000)

print("Voskモデルの読み込み成功！")
```

```bash
python test_vosk.py
```

### 5.2 Qwen3-TTSの動作確認

```python
# test_qwen_tts.py
import torch
from qwen_tts import Qwen3TTSModel
import soundfile as sf

# モデルの読み込み
model = Qwen3TTSModel.from_pretrained(
    "models/qwen-tts/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="mps",
    dtype=torch.float16,
)

# 音声生成
wavs, sr = model.generate_custom_voice(
    text="こんにちは、コア株式会社でございます。",
    language="Japanese",
    speaker="Ono Anna",
)

# ファイルに保存
sf.write("test_output.wav", wavs[0], sr)
print("音声生成成功！test_output.wavを確認してください。")
```

```bash
python test_qwen_tts.py
afplay test_output.wav  # macOSで再生
```

### 5.3 Ollamaの動作確認

```python
# test_ollama.py
import ollama

response = ollama.generate(
    model="qwen2.5:7b",
    prompt="こんにちは、簡単な自己紹介をしてください。"
)

print(response['response'])
```

```bash
python test_ollama.py
```

## 6. トラブルシューティング

### 6.1 よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| `MPS backend out of memory` | VRAMが不足 | `torch.mps.empty_cache()` を呼び出す |
| `No module named 'vosk'` | voskがインストールされていない | `pip install vosk` |
| `PortAudio not found` | portaudioがない | `brew install portaudio` |
| `Ollama connection refused` | Ollamaが起動していない | `ollama serve` で起動 |

### 6.2 M4 Max特有の問題

**flash-attnエラー**:
```
error: flash-attn is not available on Apple Silicon
```

解決策: Qwen3-TTSの起動時に `--no-flash-attn` オプションを使用:
```bash
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --device mps --no-flash-attn
```

**MPSメモリエラー**:
```python
# より小さいモデルを使用
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",  # 0.6Bモデル
    device_map="mps",
    dtype=torch.float16,
)
```

## 7. 推奨設定まとめ

```yaml
# config/m4_max_settings.yaml
hardware:
  device: "mps"
  dtype: "float16"
  memory_optimization: true

models:
  stt:
    name: "vosk-model-ja-0.22"
    sample_rate: 16000
  
  tts:
    name: "Qwen3-TTS-12Hz-1.7B-Base"
    flash_attn: false
    speaker: "Ono Anna"
  
  llm:
    provider: "ollama"
    model: "qwen2.5:7b"
    temperature: 0.7
```

## 次のステップ

セットアップが完了したら、以下のドキュメントを参照してください:

1. [ARCHITECTURE.md](./ARCHITECTURE.md) - システムアーキテクチャの詳細
2. [REQUIREMENTS.md](./REQUIREMENTS.md) - 技術要件の詳細
3. [../README.md](../README.md) - アプリケーションの起動方法
