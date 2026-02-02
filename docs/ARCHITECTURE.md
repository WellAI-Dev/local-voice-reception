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
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │                  Gradio Web Interface                    │    │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │    │   │
│  │  │  │ 音声入力     │  │ 会話ログ     │  │ ステータス   │   │    │   │
│  │  │  │ ボタン       │  │ 表示エリア   │  │ 表示         │   │    │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘   │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Application Layer                             │   │
│  │                                                                   │   │
│  │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │   │
│  │  │   STT        │     │   対話       │     │   TTS        │     │   │
│  │  │   Module     │────▶│   Manager    │────▶│   Module     │     │   │
│  │  │   (Vosk)     │     │              │     │  (Qwen3-TTS) │     │   │
│  │  └──────────────┘     └──────┬───────┘     └──────────────┘     │   │
│  │                              │                                   │   │
│  │                              ▼                                   │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │                   RAG Pipeline                             │  │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │  │   │
│  │  │  │ Query        │  │ Vector       │  │ Context      │    │  │   │
│  │  │  │ Processor    │──│ Search       │──│ Builder      │    │  │   │
│  │  │  │              │  │ (ChromaDB)   │  │              │    │  │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘    │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  │                              │                                   │   │
│  │                              ▼                                   │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │                   LLM Interface                            │  │   │
│  │  │  ┌──────────────────────────────────────────────────────┐ │  │   │
│  │  │  │              Ollama / vLLM                            │ │  │   │
│  │  │  │         (Qwen3 / Gemma / DeepSeek-R1)                │ │  │   │
│  │  │  └──────────────────────────────────────────────────────┘ │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Data Layer                                    │   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │   │
│  │  │ Knowledge    │  │ Vector       │  │ Voice        │           │   │
│  │  │ Base (MD)    │  │ Store        │  │ Samples      │           │   │
│  │  │              │  │ (ChromaDB)   │  │              │           │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐                             │   │
│  │  │ Pronunciation│  │ Config       │                             │   │
│  │  │ Dictionary   │  │ Settings     │                             │   │
│  │  └──────────────┘  └──────────────┘                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## コンポーネント詳細

### 1. STT Module (音声認識)

**技術**: Vosk

```python
# 基本構成
class STTModule:
    def __init__(self, model_path: str):
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
    
    def recognize(self, audio_data: bytes) -> str:
        """音声データをテキストに変換"""
        pass
    
    def stream_recognize(self, audio_stream) -> Generator[str, None, None]:
        """ストリーミング音声認識"""
        pass
```

**特徴**:
- 完全オフライン動作
- ストリーミング対応（リアルタイム認識）
- 日本語モデル: `vosk-model-ja-0.22`（1GB、高精度）

### 2. TTS Module (音声合成)

**技術**: Qwen3-TTS

```python
# 基本構成
class TTSModule:
    def __init__(self, model_name: str, voice_config: dict):
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="mps",  # Apple Silicon
            dtype=torch.float16
        )
        self.voice_config = voice_config
        self.pronunciation_dict = load_pronunciation_dict()
    
    def synthesize(self, text: str) -> np.ndarray:
        """テキストを音声に変換"""
        # 発音辞書による前処理
        processed_text = self._preprocess_text(text)
        # 音声合成
        wavs, sr = self.model.generate_voice_clone(
            text=processed_text,
            ref_audio=self.voice_config["ref_audio"],
            ref_text=self.voice_config["ref_text"]
        )
        return wavs[0]
```

**特徴**:
- 97ms低レイテンシ（リアルタイム対話可能）
- 音声クローン機能（3秒サンプルから複製）
- 日本語ネイティブ話者プリセット（Ono Anna）

### 3. RAG Pipeline

**技術**: LangChain + ChromaDB

```python
# 基本構成
class RAGPipeline:
    def __init__(self, knowledge_dir: str):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small"
        )
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
    
    def retrieve(self, query: str) -> List[Document]:
        """関連ドキュメントを検索"""
        return self.retriever.get_relevant_documents(query)
    
    def build_context(self, query: str, documents: List[Document]) -> str:
        """コンテキストを構築"""
        pass
```

**ナレッジ形式**:
- Markdownファイル（WellAIサマリ）
- チャンクサイズ: 500文字
- オーバーラップ: 50文字

### 4. LLM Interface

**技術**: Ollama (ローカルLLM実行)

```python
# 基本構成
class LLMInterface:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        self.model_name = model_name
        self.client = ollama.Client()
    
    def generate(self, prompt: str, context: str) -> str:
        """回答を生成"""
        full_prompt = self._build_prompt(prompt, context)
        response = self.client.generate(
            model=self.model_name,
            prompt=full_prompt
        )
        return response['response']
```

**推奨モデル（M4 Max用）**:
| モデル | サイズ | 特徴 |
|--------|--------|------|
| qwen2.5:7b | 4.4GB | バランス型、日本語良好 |
| gemma2:9b | 5.5GB | 高品質、やや重い |
| deepseek-r1:7b | 4.7GB | 推論特化 |

## データフロー

### 1. 音声入力→テキスト変換

```
[マイク] → [音声データ (16kHz)] → [Vosk] → [テキスト]
                                      │
                            ┌─────────┴─────────┐
                            │ 部分認識結果      │
                            │ (リアルタイム表示) │
                            └───────────────────┘
```

### 2. テキスト→回答生成

```
[ユーザー質問]
       │
       ▼
[Embedding変換] → [ベクトル検索] → [関連ドキュメント取得]
                                          │
                                          ▼
                              [コンテキスト構築]
                                          │
                                          ▼
                              [LLMプロンプト生成]
                                          │
                                          ▼
                              [回答生成 (Ollama)]
```

### 3. 回答テキスト→音声出力

```
[回答テキスト]
       │
       ▼
[発音辞書による前処理]
       │
       ▼
[Qwen3-TTS 音声合成]
       │
       ▼
[音声クローン適用]
       │
       ▼
[スピーカー出力]
```

## 発音辞書システム

### 構造

```yaml
# config/pronunciation_dict.yaml
terms:
  - original: "Cor.Inc"
    reading: "コア インク"
    priority: high
  
  - original: "TapForge"
    reading: "タップフォージ"
    priority: high
  
  - original: "WellAI"
    reading: "ウェルエーアイ"
    priority: high

patterns:
  # 正規表現パターン
  - pattern: "([0-9]+)円"
    replacement: "\\1えん"
```

### 処理フロー

```python
def preprocess_text(text: str, dict_config: dict) -> str:
    """発音辞書による前処理"""
    # 1. 高優先度の固有名詞を置換
    for term in dict_config["terms"]:
        if term["priority"] == "high":
            text = text.replace(term["original"], term["reading"])
    
    # 2. パターンマッチング
    for pattern in dict_config["patterns"]:
        text = re.sub(pattern["pattern"], pattern["replacement"], text)
    
    return text
```

## セキュリティ考慮事項

1. **データの外部送信なし**: すべての処理がローカルで完結
2. **ナレッジベースの保護**: アクセス制御（将来実装）
3. **音声データの非保存**: 一時処理のみ（オプションで録音可能）

## スケーラビリティ

### 単一マシン構成（現行）
- M4 Max MacBook Pro
- 同時接続: 1

### 将来の拡張（検討中）
- 複数GPU構成
- Kubernetes + vLLM
- 同時接続: N
