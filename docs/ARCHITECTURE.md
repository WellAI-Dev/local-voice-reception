# システムアーキテクチャ

## 概要

本システムは、完全ローカル環境で動作する音声AI受付システムです。
クラウドAPIを一切使用せず、すべての処理をローカルで実行することで、月額費用の固定化とプライバシー保護を実現します。

## アーキテクチャ図

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Local Voice Reception AI                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Presentation Layer (UI)                       │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │                 Gradio Web Interface                      │   │   │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │   │
│  │  │  │ 会話     │ │ STT辞書  │ │ ナレッジ │ │ 音声     │   │   │   │
│  │  │  │ タブ     │ │ タブ     │ │ 管理タブ │ │ クローン │   │   │   │
│  │  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    v                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Application Layer                             │   │
│  │                                                                   │   │
│  │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │   │
│  │  │  VoskSTT     │     │ OllamaClient │     │  QwenTTS     │     │   │
│  │  │  + STT辞書   │────>│ + RAG        │────>│  (dual-mode) │     │   │
│  │  │              │     │              │     │              │     │   │
│  │  └──────────────┘     └──────┬───────┘     └──────────────┘     │   │
│  │                              │                                   │   │
│  │                              v                                   │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │                 KnowledgeManager (RAG)                     │  │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │  │   │
│  │  │  │ Document     │  │ ChromaDB     │  │ Context      │    │  │   │
│  │  │  │ Loader       │──│ Vector Store │──│ Builder      │    │  │   │
│  │  │  │              │  │              │  │              │    │  │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘    │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    v                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Data Layer                                    │   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │   │
│  │  │ Knowledge    │  │ Vector       │  │ Voice        │           │   │
│  │  │ Base (MD)    │  │ Store        │  │ Samples      │           │   │
│  │  │              │  │ (ChromaDB)   │  │              │           │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │   │
│  │  │ Pronunciation│  │ STT          │  │ Config       │           │   │
│  │  │ Dictionary   │  │ Dictionary   │  │ Settings     │           │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## コンポーネント詳細

### 1. VoskSTT（音声認識）

**ファイル**: `src/stt/vosk_stt.py`

```python
class VoskSTT:
    def __init__(self, model_path: str, sample_rate: int = 16000):
        """Vosk モデルを読み込み、KaldiRecognizer を初期化"""

    def recognize(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """音声データをテキストに変換"""
```

**特徴**:
- 完全オフライン動作
- 日本語モデル: `vosk-model-small-ja-0.22`（テスト用）/ `vosk-model-ja-0.22`（本番用）
- macOS ARM64 では optional dependency（wheel 未提供のため）

### 2. STTDictionary（認識後補正）

**ファイル**: `src/stt/dictionary.py`

```python
class STTDictionary:
    def __init__(self, dict_path: str = "config/stt_dictionary.yaml"):
        """YAML 辞書ファイルを読み込み"""

    def correct(self, text: str) -> str:
        """exact 置換 → regex パターン の順で補正を適用"""

    def add_correction(self, wrong: str, correct: str, note: str = "") -> None:
        """補正エントリを追加（既存の場合は更新）"""

    def add_pattern(self, pattern: str, replacement: str) -> None:
        """正規表現パターンを追加（バリデーション付き）"""

    def replace_all(self, corrections: list[dict], patterns: list[dict]) -> None:
        """UI からの一括更新用"""

    def save(self) -> None:
        """辞書を YAML ファイルに保存"""
```

**特徴**:
- Vosk の誤認識を後処理で補正
- exact 置換（最長マッチ優先）と regex パターンの2段階
- UI から直接編集可能（interactive DataFrame）
- スレッドセーフ（`threading.Lock`）

### 3. QwenTTS（音声合成）

**ファイル**: `src/tts/qwen_tts.py`

```python
class QwenTTS:
    def __init__(self, model_name: str = None, device: str = "auto",
                 mode: str = "custom_voice", ...):
        """デュアルモード TTS の初期化"""

    def synthesize(self, text: str, speaker: str, language: str,
                   instruct: str = None) -> tuple[np.ndarray, int]:
        """custom_voice モードでの音声合成"""

    def synthesize_with_clone(self, text: str, language: str) -> tuple[np.ndarray, int]:
        """voice_clone モードでの音声合成（キャッシュ済みプロンプト使用）"""

    def switch_mode(self, new_mode: str) -> None:
        """ランタイムでのモード切り替え（モデル入れ替え）"""

    def prepare_clone(self, ref_audio: str, ref_text: str, language: str) -> None:
        """音声クローン用プロンプトを事前生成してキャッシュ"""

    def update_reference_audio(self, ref_audio_path: str, ref_text: str,
                               language: str) -> None:
        """参照音声を更新してプロンプトを再生成"""

    def preload(self) -> None:
        """モデルの事前ロード（初回遅延の回避）"""
```

**デュアルモード**:

| モード | モデル | 用途 |
|--------|--------|------|
| `custom_voice` | `Qwen3-TTS-12Hz-1.7B-CustomVoice` | プリセット話者（ono_anna 等） |
| `voice_clone` | `Qwen3-TTS-12Hz-1.7B-Base` | 録音した声のクローン |

**スレッドセーフティ**:
- `_model_lock` (Lock): モデルのロード/アンロード
- `_prompt_lock` (RLock): voice_clone_prompt キャッシュのアクセス

### 4. OllamaClient（LLM）

**ファイル**: `src/llm/ollama_client.py`

```python
class OllamaClient:
    def __init__(self, base_url: str, model: str, system_prompt: str, ...):
        """Ollama クライアントの初期化"""

    def generate(self, user_message: str, context: str = "") -> str:
        """コンテキスト付きで回答を生成"""

    def check_connection(self) -> bool:
        """Ollama サーバーへの接続確認"""

    def clear_history(self) -> None:
        """会話履歴をクリア"""
```

**推奨モデル（M4 Max 用）**:

| モデル | サイズ | 特徴 |
|--------|--------|------|
| qwen2.5:7b | 4.4GB | バランス型、日本語良好 |
| gemma2:9b | 5.5GB | 高品質、やや重い |
| deepseek-r1:7b | 4.7GB | 推論特化 |

### 5. KnowledgeManager（RAG）

**ファイル**: `src/llm/knowledge_manager.py`

```python
class KnowledgeManager:
    def __init__(self, knowledge_dir: str, vectorstore_dir: str,
                 embedding_model: str, ...):
        """RAG パイプラインの初期化"""

    def add_document(self, file_path: str) -> None:
        """Markdown ドキュメントをベクトルストアに追加"""

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """類似ドキュメントを検索"""

    def rebuild_index(self) -> None:
        """ベクトルストアを全再構築"""
```

**ナレッジ形式**:
- Markdown ファイル（WellAI サマリ）
- チャンクサイズ: 500文字（設定可能）
- Embedding: `intfloat/multilingual-e5-small`

### 6. VoiceReceptionApp（メインアプリケーション）

**ファイル**: `src/app.py`

```python
class VoiceReceptionApp:
    def __init__(self, config: dict):
        """設定を読み込み、各コンポーネントの参照を保持"""

    def initialize(self) -> None:
        """STT, LLM, TTS, KnowledgeManager を初期化"""

    def _synthesize_speech(self, text: str) -> tuple[np.ndarray, int]:
        """self.tts.mode に応じて適切な TTS メソッドを呼び出し"""

    def get_conversation_display(self) -> str:
        """会話ログのテキスト表現を返す"""

    def clear_conversation(self) -> None:
        """会話履歴をクリア"""
```

**UI 構成** (`create_ui()` 関数):
- **会話タブ**: Push-to-Talk、Chatbot、音声出力
- **STT辞書タブ**: corrections/patterns の DataFrame 編集 + 保存
- **ナレッジ管理タブ**: ドキュメントの追加・削除・再構築
- **音声クローン設定タブ**: 音声録音 → `switch_mode("voice_clone")` → `update_reference_audio()`
- **設定タブ**: TTS/LLM パラメータの調整

## データフロー

### 1. 音声入力→テキスト変換

```
[マイク] → [音声データ (16kHz)] → [VoskSTT] → [STTDictionary] → [補正済みテキスト]
```

### 2. テキスト→回答生成

```
[ユーザー質問]
       │
       v
[KnowledgeManager.search()] → [関連ドキュメント取得]
                                       │
                                       v
                           [コンテキスト構築]
                                       │
                                       v
                           [OllamaClient.generate()]
                                       │
                                       v
                           [回答テキスト]
```

### 3. 回答テキスト→音声出力

```
[回答テキスト]
       │
       v
[発音辞書による前処理 (_preprocess_text)]
       │
       v
[QwenTTS (mode に応じて分岐)]
  ├── custom_voice: synthesize(text, speaker, language)
  └── voice_clone:  synthesize_with_clone(text, language)
       │
       v
[スピーカー出力]
```

### 4. 音声クローン登録フロー

```
[UI: 音声録音 + テキスト入力]
       │
       v
[WAV ファイル保存 (data/voice_samples/)]
       │
       v
[QwenTTS.switch_mode("voice_clone")]  ← モデルを Base に切り替え
       │
       v
[QwenTTS.update_reference_audio()]    ← プロンプトキャッシュ生成
       │
       v
[config["tts"]["mode"] = "voice_clone"]
```

## セキュリティ考慮事項

1. **データの外部送信なし**: すべての処理がローカルで完結
2. **STT辞書の regex バリデーション**: パターン長制限（200文字）で ReDoS を防止
3. **音声データの非保存**: 一時処理のみ（音声クローン用サンプルは明示的に保存）

## スケーラビリティ

### 単一マシン構成（現行）
- M4 Max MacBook Pro
- 同時接続: 1

### 将来の拡張（検討中）
- 複数GPU構成
- Kubernetes + vLLM
- 同時接続: N
