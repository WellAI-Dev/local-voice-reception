# UI改善提案書

> **[一部実装済み]** このドキュメントの提案のうち、以下が実装されています:
> - Chatbot コンポーネント導入（優先度1）
> - ステータスインジケーター / バッジ表示（優先度2）
> - STT辞書管理タブ、ナレッジ管理タブ、音声クローン設定タブの追加（追加機能）
>
> 未実装: gr.State() リファクタリング、ネイティブオーディオストリーミング

> 調査日: 2026-02-02
> ベース: Gradio 6.0 公式ドキュメント (context7)

## 現状分析

### 現在の実装
- カスタムJavaScript による Push-to-Talk
- `gr.Textbox` で会話ログ表示
- 手動の base64 エンコード/デコード
- カスタムCSS で要素を非表示

### 問題点
1. **カスタムJS依存**: Gradio のネイティブ機能を活用していない
2. **会話UI**: Chatbot コンポーネントを使っていない
3. **状態管理**: `gr.State()` を適切に使っていない
4. **ローディング表示**: 処理中の視覚的フィードバックが弱い
5. **レスポンシブ**: モバイル対応が不十分

---

## 改善提案

### 1. Gradio ネイティブのオーディオストリーミング活用

**現状:**
```python
# カスタムJS で WebRTC 録音 → base64 → hidden input
PTT_JS = """... 300行のJavaScript ..."""
```

**改善案:**
```python
with gr.Blocks() as demo:
    audio_input = gr.Audio(
        sources=["microphone"],
        type="numpy",
        streaming=True,  # ストリーミング有効
        label="音声入力"
    )
    audio_state = gr.State(None)

    # ストリーミング中の処理
    audio_input.stream(
        fn=process_audio_chunk,
        inputs=[audio_input, audio_state],
        outputs=[audio_state, transcription_text],
        stream_every=0.5,  # 500msごとに処理
        time_limit=30,     # 最大30秒
    )

    # 録音停止時の処理
    audio_input.stop_recording(
        fn=process_final_audio,
        inputs=[audio_state],
        outputs=[audio_output, response_text]
    )
```

**メリット:**
- JavaScript コード削減 (300行 → 0行)
- ブラウザ互換性が Gradio 側で保証
- WebRTC の複雑さを隠蔽

**デメリット:**
- 「押している間だけ録音」の PTT 動作は Gradio ネイティブでは難しい
- ストリーミングモードは連続録音向け

**結論:** PTT が必須なら現在のカスタム JS を維持。ただし、以下の代替案を検討:

---

### 2. 会話UIを Chatbot コンポーネントに変更

**現状:**
```python
conversation_display = gr.Textbox(
    label="📋 会話ログ",
    lines=8,
    interactive=False,
)
```

**改善案:**
```python
chatbot = gr.Chatbot(
    label="会話",
    type="messages",  # OpenAI形式
    height=400,
    avatar_images=(
        "https://em-content.zobj.net/source/apple/391/bust-in-silhouette_1f464.png",  # user
        "https://em-content.zobj.net/source/apple/391/robot_1f916.png",  # assistant
    ),
)

# 更新時
def update_chat(user_text, ai_response, history):
    history = history or []
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": ai_response})
    return history
```

**メリット:**
- プロフェッショナルな見た目
- アバター、タイムスタンプ対応
- スクロール自動追従
- Markdown サポート

---

### 3. gr.State() による適切な状態管理

**現状:**
```python
class VoiceReceptionApp:
    def __init__(self):
        self.conversation_log: list = []  # インスタンス変数
```

**改善案:**
```python
with gr.Blocks() as demo:
    # Gradio の State で状態管理
    conversation_state = gr.State(value=[])
    audio_buffer_state = gr.State(value=None)

    def process_audio(audio, conv_state):
        # 状態を更新して返す
        new_conv = conv_state + [{"user": text, "ai": response}]
        return audio_output, new_conv

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, conversation_state],
        outputs=[audio_output, conversation_state],
    )
```

**メリット:**
- 複数ユーザー対応（各ユーザーが独立した状態を持つ）
- リフレッシュで状態リセット
- Gradio の最適化が効く

---

### 4. ローディング状態の改善

**現状:**
```python
yield None, recognized_text, "回答を生成中...", self.get_conversation_display()
```

**改善案:**
```python
with gr.Blocks() as demo:
    with gr.Row():
        # ステータスインジケーター
        status_indicator = gr.HTML(
            value='<div class="status idle">待機中</div>',
            elem_id="status-indicator"
        )

    # カスタムCSS
    custom_css = """
    .status {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }
    .status.idle { background: #e5e7eb; color: #374151; }
    .status.recording { background: #fecaca; color: #dc2626; animation: pulse 1s infinite; }
    .status.processing { background: #dbeafe; color: #2563eb; }
    .status.speaking { background: #d1fae5; color: #059669; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    """
```

**ステータス遷移:**
```
待機中 → 録音中 → 音声認識中 → 回答生成中 → 音声合成中 → 再生中 → 待機中
```

---

### 5. レスポンシブ対応

**現状:**
```python
with gr.Row():
    with gr.Column(scale=1):
        # 入力
    with gr.Column(scale=1):
        # 出力
```

**改善案:**
```python
custom_css = """
/* モバイル対応 */
@media (max-width: 768px) {
    .gradio-container {
        padding: 8px !important;
    }

    #ptt-btn {
        padding: 32px !important;
        font-size: 24px !important;
    }

    .gradio-row {
        flex-direction: column !important;
    }

    .gradio-column {
        width: 100% !important;
        max-width: 100% !important;
    }
}

/* タッチデバイス最適化 */
@media (hover: none) {
    #ptt-btn {
        padding: 40px !important;
    }
}
"""
```

---

### 6. エラーハンドリング UI

**現状:**
```python
yield None, "", f"⚠️ 音声認識エラー: {e}", ""
```

**改善案:**
```python
def show_error(message: str, error_type: str = "warning") -> str:
    """Generate error HTML with appropriate styling."""
    icons = {
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️",
    }
    colors = {
        "error": "#dc2626",
        "warning": "#d97706",
        "info": "#2563eb",
    }
    return f"""
    <div style="
        background: {colors[error_type]}10;
        border-left: 4px solid {colors[error_type]};
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    ">
        <strong>{icons[error_type]} {error_type.upper()}</strong><br>
        {message}
    </div>
    """

# 使用例
error_display = gr.HTML(elem_id="error-display")
yield gr.update(value=show_error("マイクが検出されませんでした", "warning"))
```

---

## 推奨実装順序

| 優先度 | 改善項目 | 工数 | 効果 |
|--------|---------|------|------|
| 1 | Chatbot コンポーネント導入 | 小 | 高 |
| 2 | ローディング状態改善 | 小 | 高 |
| 3 | レスポンシブCSS追加 | 小 | 中 |
| 4 | エラーハンドリングUI | 小 | 中 |
| 5 | gr.State() リファクタリング | 中 | 中 |
| 6 | ネイティブオーディオ検討 | 大 | 低* |

*PTT 要件がある限り、ネイティブオーディオへの移行効果は限定的

---

## 完全な改善後コード例

```python
import gradio as gr
import numpy as np
from typing import Optional, Tuple, Generator

# カスタムCSS
CUSTOM_CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}

.status-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 16px;
    font-size: 14px;
    font-weight: 600;
}
.status-idle { background: #e5e7eb; color: #374151; }
.status-recording { background: #fecaca; color: #dc2626; }
.status-processing { background: #dbeafe; color: #2563eb; }
.status-speaking { background: #d1fae5; color: #059669; }

@media (max-width: 768px) {
    #ptt-btn { padding: 32px !important; font-size: 22px !important; }
    .gradio-row { flex-direction: column !important; }
}
"""

def create_improved_ui(app):
    with gr.Blocks(css=CUSTOM_CSS, js=PTT_JS) as demo:
        # ヘッダー
        gr.Markdown("# 🎙️ 音声AI受付システム")

        # ステータスバー
        status_html = gr.HTML(
            value='<span class="status-badge status-idle">待機中</span>'
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                # PTT ボタン
                gr.HTML(PTT_HTML)

                # 非表示入力
                audio_base64 = gr.Textbox(elem_id="ptt-audio-data", visible=False)
                submit_btn = gr.Button("送信", elem_id="ptt-submit-btn", visible=False)

                # フォールバック
                with gr.Accordion("📁 ファイルアップロード", open=False):
                    audio_file = gr.Audio(sources=["upload"], type="numpy")
                    upload_btn = gr.Button("アップロード", size="sm")

            with gr.Column(scale=1):
                # 出力オーディオ
                audio_output = gr.Audio(
                    label="🔊 AI応答",
                    autoplay=True,
                    show_download_button=True,
                )

        # 認識結果
        with gr.Row():
            user_text = gr.Textbox(label="📝 あなたの発言", interactive=False)
            ai_text = gr.Textbox(label="💬 AIの回答", interactive=False)

        # 会話履歴（Chatbot コンポーネント）
        chatbot = gr.Chatbot(
            label="会話履歴",
            type="messages",
            height=300,
        )

        # 状態管理
        conversation_state = gr.State(value=[])

        # クリアボタン
        clear_btn = gr.Button("🗑️ 会話をクリア", size="sm")

        # イベントハンドラ
        def process_and_update_chat(audio_b64, history):
            for result in app.process_base64_audio(audio_b64):
                audio_out, user, ai, _ = result
                yield audio_out, user, ai, history

            # 最終結果で履歴更新
            if user and ai:
                history = history or []
                history.append({"role": "user", "content": user})
                history.append({"role": "assistant", "content": ai})
            yield audio_out, user, ai, history

        submit_btn.click(
            fn=process_and_update_chat,
            inputs=[audio_base64, conversation_state],
            outputs=[audio_output, user_text, ai_text, chatbot],
        )

        clear_btn.click(
            fn=lambda: (None, "", "", []),
            outputs=[audio_output, user_text, ai_text, chatbot],
        )

    return demo
```

---

## 参考リンク

- [Gradio Streaming Inputs Guide](https://github.com/gradio-app/gradio/blob/gradio@6.0.1/guides/04_additional-features/03_streaming-inputs.md)
- [Gradio Custom CSS/JS Guide](https://github.com/gradio-app/gradio/blob/gradio@6.0.1/guides/03_building-with-blocks/07_custom-CSS-and-JS.md)
- [Gradio Conversational Chatbot](https://github.com/gradio-app/gradio/blob/gradio@6.0.1/guides/07_streaming/04_conversational-chatbot.md)
- [Gradio Real-time ASR Example](https://github.com/gradio-app/gradio/blob/gradio@6.0.1/demo/stream_asr/run.ipynb)
