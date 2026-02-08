#!/usr/bin/env python3
"""
Local Voice Reception AI - Main Application.
Gradio-based web interface for voice-to-voice conversation with streaming support.
"""

import base64
import io
import logging
import os
import sys
from pathlib import Path
from typing import Generator, Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stt import VoskSTT
from src.llm import OllamaClient
from src.tts import QwenTTS
from src.utils.device import get_device_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_config() -> dict:
    """Load configuration from YAML."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class VoiceReceptionApp:
    """
    Main application class for Voice Reception AI.
    Orchestrates STT → LLM → TTS pipeline with streaming support.
    """

    def __init__(self, config: dict):
        """Initialize the application with configuration."""
        self.config = config
        self.stt: Optional[VoskSTT] = None
        self.llm: Optional[OllamaClient] = None
        self.tts: Optional[QwenTTS] = None
        self.conversation_log: list = []
        self._tts_sample_rate = 24000

    def initialize(self) -> Tuple[bool, str]:
        """Initialize all components."""
        errors = []

        # Initialize STT
        try:
            stt_config = self.config.get("stt", {})
            model_path = PROJECT_ROOT / stt_config.get(
                "model_path", "models/vosk/vosk-model-small-ja-0.22"
            )
            if model_path.exists():
                self.stt = VoskSTT(
                    model_path=str(model_path),
                    sample_rate=stt_config.get("sample_rate", 16000),
                )
                logger.info("STT initialized")
            else:
                errors.append(f"Vosk model not found: {model_path}")
        except Exception as e:
            errors.append(f"STT init error: {e}")

        # Initialize LLM
        try:
            llm_config = self.config.get("llm", {})
            ollama_config = llm_config.get("ollama", {})
            self.llm = OllamaClient(
                model=ollama_config.get("model", "qwen2.5:7b"),
                base_url=ollama_config.get("base_url", "http://localhost:11434"),
                system_prompt=llm_config.get("system_prompt"),
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 512),
            )
            if not self.llm.check_connection():
                errors.append("Ollama server not running")
            else:
                logger.info("LLM initialized")
        except Exception as e:
            errors.append(f"LLM init error: {e}")

        # Initialize TTS
        try:
            tts_config = self.config.get("tts", {})
            tts_mode = tts_config.get("mode", "custom_voice")

            self.tts = QwenTTS(
                model_name=tts_config.get("model_name"),
                mode=tts_mode,
                device=tts_config.get("device", "auto"),
                pronunciation_dict_path=str(
                    PROJECT_ROOT / tts_config.get(
                        "pronunciation_dict", "config/pronunciation_dict.yaml"
                    )
                ),
            )

            if tts_config.get("preload", False):
                logger.info("Preloading TTS model...")
                self.tts.preload()
                logger.info("TTS model preloaded")
            else:
                logger.info("TTS initialized (model will load on first use)")

            # Pre-compute voice clone prompt if mode is voice_clone
            if tts_mode == "voice_clone":
                clone_config = tts_config.get("voice_clone", {})
                ref_audio = clone_config.get("ref_audio", "")
                ref_text = clone_config.get("ref_text", "")
                ref_audio_path = str(PROJECT_ROOT / ref_audio) if ref_audio else ""

                if ref_audio_path and Path(ref_audio_path).exists():
                    logger.info("Pre-computing voice clone prompt...")
                    self.tts.prepare_clone(
                        ref_audio_path=ref_audio_path,
                        ref_text=ref_text,
                        language="Japanese",
                    )
                    logger.info("Voice clone prompt ready")
                else:
                    logger.warning(
                        f"Voice clone ref audio not found: {ref_audio_path}. "
                        "Record your voice via the UI or place a WAV file there."
                    )

        except Exception as e:
            errors.append(f"TTS init error: {e}")

        if errors:
            return False, "\n".join(errors)
        return True, "All components initialized successfully"

    def _synthesize_speech(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech using the configured TTS mode.
        Routes to voice_clone or custom_voice based on config.
        """
        tts_config = self.config.get("tts", {})
        tts_mode = tts_config.get("mode", "custom_voice")
        custom_config = tts_config.get("custom_voice", {})
        language = custom_config.get("language", "Japanese")

        # Thread-safe check of voice clone prompt cache
        if tts_mode == "voice_clone":
            with self.tts._prompt_lock:
                has_prompt = self.tts._voice_clone_prompt is not None

            if has_prompt:
                return self.tts.synthesize_with_clone(
                    text=text,
                    language=language,
                )

            # Voice clone requested but no prompt cached - fallback with warning
            logger.warning("Voice clone prompt not cached, falling back to custom_voice")

            # Base model may not support generate_custom_voice,
            # so only fallback if the TTS mode allows it
            if self.tts.mode == "voice_clone":
                logger.warning(
                    "Base model may not support custom_voice synthesis. "
                    "Record your voice via the UI or provide ref_audio in config."
                )

        # custom_voice mode (or fallback)
        speaker = custom_config.get("speaker", "ono_anna")
        return self.tts.synthesize(
            text=text,
            speaker=speaker,
            language=language,
        )

    def process_audio_streaming(
        self,
        audio: Optional[Tuple[int, np.ndarray]],
    ) -> Generator[Tuple[Optional[Tuple[int, np.ndarray]], str, str, str], None, None]:
        """
        Process audio with streaming LLM text display, but single TTS call for quality.
        """
        if audio is None:
            yield None, "", "音声が入力されていません。", ""
            return

        sample_rate, audio_data = audio

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Convert to int16 if needed
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = (audio_data * 32767).astype(np.int16)

        # Resample if needed (Vosk expects 16kHz)
        if sample_rate != 16000:
            import scipy.signal
            audio_data = scipy.signal.resample(
                audio_data, int(len(audio_data) * 16000 / sample_rate)
            ).astype(np.int16)

        # Step 1: Speech-to-Text
        if self.stt is None:
            yield None, "", "⚠️ STTが初期化されていません", ""
            return

        try:
            recognized_text = self.stt.recognize(audio_data)
            if not recognized_text:
                yield None, "", "音声を認識できませんでした。もう一度お話しください。", ""
                return
            logger.info(f"Recognized: {recognized_text}")
        except Exception as e:
            logger.error(f"STT error: {e}")
            yield None, "", f"⚠️ 音声認識エラー: {e}", ""
            return

        yield None, recognized_text, "回答を生成中...", self.get_conversation_display()

        if self.llm is None:
            yield None, recognized_text, "⚠️ LLMが初期化されていません", self.get_conversation_display()
            return

        # Step 2: Stream LLM response
        full_response = ""
        try:
            for token in self.llm.generate_stream(recognized_text):
                full_response += token
                yield None, recognized_text, full_response + "▌", self.get_conversation_display()

            logger.info(f"Response: {full_response}")

        except Exception as e:
            logger.error(f"LLM error: {e}")
            yield None, recognized_text, f"⚠️ 回答生成エラー: {e}", self.get_conversation_display()
            return

        # Step 3: Generate TTS for complete response
        yield None, recognized_text, full_response + "\n\n🔊 音声を生成中...", self.get_conversation_display()

        audio_output = None
        if self.tts and full_response.strip():
            try:
                audio_data_out, sr = self._synthesize_speech(full_response)
                audio_output = (sr, audio_data_out)
                logger.info(f"TTS generated: {len(audio_data_out)} samples at {sr}Hz")
            except Exception as e:
                logger.error(f"TTS error: {e}")
                full_response += f"\n\n⚠️ 音声合成エラー: {e}"

        self.conversation_log.append(
            {"user": recognized_text, "assistant": full_response}
        )

        yield audio_output, recognized_text, full_response, self.get_conversation_display()

    def process_base64_audio(
        self,
        audio_base64: str,
    ) -> Generator[Tuple[Optional[Tuple[int, np.ndarray]], str, str, str], None, None]:
        """Process base64 encoded audio from PTT (WebM format)."""
        if not audio_base64:
            yield None, "", "音声が入力されていません。", ""
            return

        try:
            from pydub import AudioSegment

            # Decode base64
            audio_bytes = base64.b64decode(audio_base64)
            logger.info(f"Base64 decoded: {len(audio_bytes)} bytes")

            # Convert WebM to WAV using pydub (requires ffmpeg)
            audio_buffer = io.BytesIO(audio_bytes)
            audio_segment = AudioSegment.from_file(audio_buffer, format="webm")

            logger.info(f"Audio loaded: {len(audio_segment)}ms, {audio_segment.channels}ch, {audio_segment.frame_rate}Hz, {audio_segment.sample_width * 8}bit")

            # Convert to mono and set sample rate to 16kHz for Vosk
            audio_segment = audio_segment.set_channels(1).set_frame_rate(16000).set_sample_width(2)  # 16-bit

            # Get raw audio data as numpy array (int16)
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
            sample_rate = audio_segment.frame_rate

            # Check audio level
            max_amplitude = np.max(np.abs(samples))
            logger.info(f"PTT audio: {len(samples)} samples, {sample_rate}Hz, dtype={samples.dtype}, max_amp={max_amplitude}")

            if max_amplitude < 100:
                logger.warning("Audio level is very low - might be silent")

            yield from self.process_audio_streaming((sample_rate, samples))

        except Exception as e:
            logger.error(f"Base64 audio decode error: {e}")
            import traceback
            traceback.print_exc()
            yield None, "", f"⚠️ 音声デコードエラー: {e}", ""

    def clear_conversation(self) -> Tuple[str, list]:
        """Clear conversation history."""
        self.conversation_log = []
        if self.llm:
            self.llm.clear_history()
        return "", []

    def get_conversation_display(self) -> str:
        """Get formatted conversation log for display."""
        lines = []
        for turn in self.conversation_log:
            lines.append(f"👤 {turn['user']}")
            lines.append(f"🤖 {turn['assistant']}")
            lines.append("")
        return "\n".join(lines)


# Push-to-Talk HTML (no script - script will be injected via js parameter)
PTT_HTML = """
<div id="ptt-container" style="margin: 10px 0; user-select: none;">
    <button id="ptt-btn" type="button"
            style="width: 100%; padding: 24px; font-size: 20px; font-weight: bold;
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   color: white; border: none; border-radius: 12px; cursor: pointer;
                   transition: all 0.2s ease; box-shadow: 0 4px 15px rgba(102,126,234,0.4);
                   -webkit-user-select: none; user-select: none;">
        🎤 押して話す
    </button>
    <div id="ptt-status" style="text-align: center; margin-top: 10px; color: #666; font-size: 14px;">
        ボタンを押している間、録音されます
    </div>
</div>
"""

# Push-to-Talk JavaScript (injected via js parameter)
PTT_JS = """
function initPTT() {
    console.log('Initializing PTT...');

    const btn = document.getElementById('ptt-btn');
    const status = document.getElementById('ptt-status');

    if (!btn) {
        console.log('PTT button not found, retrying in 500ms...');
        setTimeout(initPTT, 500);
        return;
    }

    let mediaRecorder = null;
    let audioChunks = [];
    let stream = null;
    let isRecording = false;

    async function startRecording(e) {
        e.preventDefault();
        e.stopPropagation();

        if (isRecording) return;

        console.log('Starting recording...');

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                console.log('Recording stopped, processing...');
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64data = reader.result.split(',')[1];
                    console.log('Audio encoded, length:', base64data.length);

                    // Find Gradio's textbox - try multiple selectors
                    let hiddenInput = document.querySelector('#ptt-audio-data textarea');
                    if (!hiddenInput) {
                        hiddenInput = document.querySelector('#ptt-audio-data input[type="text"]');
                    }
                    if (!hiddenInput) {
                        // Gradio 6.x structure: elem_id is on the wrapper, textarea is inside
                        const wrapper = document.getElementById('ptt-audio-data');
                        if (wrapper) {
                            hiddenInput = wrapper.querySelector('textarea, input');
                        }
                    }

                    console.log('Found hidden input:', !!hiddenInput);

                    if (hiddenInput) {
                        // Update value using native setter to trigger Gradio's change detection
                        const proto = hiddenInput.tagName === 'TEXTAREA'
                            ? window.HTMLTextAreaElement.prototype
                            : window.HTMLInputElement.prototype;
                        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(proto, 'value').set;
                        nativeInputValueSetter.call(hiddenInput, base64data);
                        hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));

                        // Click submit button
                        setTimeout(() => {
                            let submitBtn = document.querySelector('#ptt-submit-btn button');
                            if (!submitBtn) {
                                submitBtn = document.querySelector('#ptt-submit-btn');
                            }
                            if (submitBtn) {
                                console.log('Clicking submit button...');
                                submitBtn.click();
                            } else {
                                console.error('Submit button not found');
                            }
                        }, 150);
                    } else {
                        console.error('Hidden input not found. DOM structure:', document.getElementById('ptt-audio-data')?.innerHTML);
                    }
                };
                reader.readAsDataURL(audioBlob);

                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
            };

            mediaRecorder.start(100);
            isRecording = true;

            btn.textContent = '🔴 録音中... (離すと送信)';
            btn.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
            btn.style.boxShadow = '0 4px 15px rgba(239,68,68,0.5)';
            status.textContent = '録音中...話し終わったらボタンを離してください';
            status.style.color = '#ef4444';

            console.log('Recording started');

        } catch (err) {
            console.error('Microphone access error:', err);
            status.textContent = '⚠️ マイクへのアクセスが拒否されました: ' + err.message;
            status.style.color = '#ef4444';
        }
    }

    function stopRecording(e) {
        e.preventDefault();
        e.stopPropagation();

        if (!isRecording || !mediaRecorder) return;

        console.log('Stopping recording...');

        mediaRecorder.stop();
        isRecording = false;

        btn.textContent = '🎤 押して話す';
        btn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
        btn.style.boxShadow = '0 4px 15px rgba(102,126,234,0.4)';
        status.textContent = '処理中...';
        status.style.color = '#666';
    }

    // Mouse events
    btn.addEventListener('mousedown', startRecording);
    btn.addEventListener('mouseup', stopRecording);
    btn.addEventListener('mouseleave', (e) => {
        if (isRecording) stopRecording(e);
    });

    // Touch events
    btn.addEventListener('touchstart', startRecording, { passive: false });
    btn.addEventListener('touchend', stopRecording, { passive: false });
    btn.addEventListener('touchcancel', stopRecording, { passive: false });

    // Prevent context menu
    btn.addEventListener('contextmenu', (e) => e.preventDefault());

    console.log('PTT initialized successfully!');
    status.textContent = 'ボタンを押している間、録音されます ✓';
}

// Start initialization
setTimeout(initPTT, 1000);
"""


def create_ui(app: VoiceReceptionApp) -> Tuple[gr.Blocks, dict]:
    """Create Gradio UI with Push-to-Talk and streaming support."""
    ui_config = app.config.get("ui", {})
    theme_name = ui_config.get("theme", "soft")
    theme = getattr(gr.themes, theme_name.capitalize(), gr.themes.Soft)()

    # Custom CSS with responsive design and status styling
    custom_css = """
    /* Container */
    .gradio-container {
        max-width: 960px !important;
        margin: 0 auto !important;
    }

    /* Hidden elements */
    .hidden-input {
        position: absolute !important;
        left: -9999px !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .status-idle { background: #e5e7eb; color: #374151; }
    .status-recording { background: #fecaca; color: #dc2626; animation: pulse 1s infinite; }
    .status-processing { background: #dbeafe; color: #2563eb; }
    .status-speaking { background: #d1fae5; color: #059669; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* Error display */
    .error-box {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .warning-box {
        background: #fffbeb;
        border-left: 4px solid #d97706;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }

    /* Chatbot styling */
    .chatbot-container {
        border-radius: 12px !important;
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 8px !important;
        }
        #ptt-btn {
            padding: 28px !important;
            font-size: 22px !important;
        }
        .gradio-row {
            flex-direction: column !important;
        }
        .gradio-column {
            width: 100% !important;
            max-width: 100% !important;
        }
    }

    /* Touch device optimization */
    @media (hover: none) {
        #ptt-btn {
            padding: 36px !important;
        }
    }
    """

    with gr.Blocks(
        title=ui_config.get("title", "音声AI受付システム"),
    ) as demo:
        # Header
        gr.Markdown("# 🎙️ 音声AI受付システム")

        # Status indicator
        status_html = gr.HTML(
            value='<div style="text-align: center; margin: 8px 0;"><span class="status-badge status-idle">⏸️ 待機中</span></div>',
            elem_id="status-display",
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                # Instructions
                gr.Markdown(
                    "**ボタンを押している間**だけ録音されます",
                    elem_classes=["instruction-text"],
                )

                # Push-to-Talk HTML component
                gr.HTML(PTT_HTML)

                # Hidden textbox for base64 audio
                audio_base64_input = gr.Textbox(
                    label="",
                    elem_id="ptt-audio-data",
                    elem_classes=["hidden-input"],
                    container=False,
                )

                # Hidden submit button
                ptt_submit_btn = gr.Button(
                    "送信",
                    elem_id="ptt-submit-btn",
                    elem_classes=["hidden-input"],
                )

                # Fallback: File upload
                with gr.Accordion("📁 ファイルをアップロード", open=False):
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="numpy",
                        label="音声入力",
                    )
                    manual_submit_btn = gr.Button("🚀 送信", variant="secondary", size="sm")

            with gr.Column(scale=1):
                # Output audio
                audio_output = gr.Audio(
                    label="🔊 AI応答",
                    type="numpy",
                    autoplay=True,
                )

                # Current turn display
                with gr.Group():
                    recognized_text = gr.Textbox(
                        label="📝 あなたの発言",
                        lines=2,
                        interactive=False,
                        
                    )
                    response_text = gr.Textbox(
                        label="💬 AIの回答",
                        lines=3,
                        interactive=False,
                        
                    )

        # Conversation history using Chatbot component
        chatbot = gr.Chatbot(
            label="📋 会話履歴",
            height=280,
            elem_classes=["chatbot-container"],
        )

        # Action buttons
        with gr.Row():
            clear_btn = gr.Button("🗑️ 会話をクリア", size="sm", variant="secondary")

        # Voice clone recording section
        with gr.Accordion("🎙️ 音声クローン設定", open=False):
            gr.Markdown(
                "**自分の声で応答させたい場合**\n\n"
                "下記のテキストを読み上げて録音してください。"
                "摩擦音・長音・特殊音節を含む文で音質が向上します。"
            )
            ref_text_input = gr.Textbox(
                label="読み上げテキスト",
                value="今日はとても晴れた日で、風は少し冷たく感じます。朝はパンとコーヒーを用意して、ゆっくりニュースを読みました。",
                lines=2,
            )
            ref_audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="リファレンス音声 (3秒以上推奨)",
            )
            register_voice_btn = gr.Button(
                "🔊 この声で登録",
                variant="primary",
                size="sm",
            )
            clone_status = gr.Textbox(
                label="登録ステータス",
                interactive=False,
                value="未登録",
            )

            def register_voice(audio_tuple, ref_text):
                if audio_tuple is None:
                    return "音声が入力されていません。マイクで録音するかファイルをアップロードしてください。"

                if not ref_text.strip():
                    return "読み上げテキストを入力してください。"

                sample_rate_in, audio_data_in = audio_tuple

                # Convert to mono if stereo
                if len(audio_data_in.shape) > 1:
                    audio_data_in = audio_data_in.mean(axis=1)

                # Convert to float32 for saving
                if audio_data_in.dtype == np.int16:
                    audio_data_in = audio_data_in.astype(np.float32) / 32767.0

                # Validate audio duration
                duration = len(audio_data_in) / sample_rate_in
                if duration < 1.0:
                    return f"音声が短すぎます（{duration:.1f}秒）。最低1秒以上、推奨3秒以上で録音してください。"

                # Save reference audio
                voice_dir = PROJECT_ROOT / "data" / "voice_samples"
                voice_dir.mkdir(parents=True, exist_ok=True)
                ref_path = voice_dir / "company_voice.wav"

                sf.write(str(ref_path), audio_data_in, sample_rate_in)
                logger.info(f"Reference audio saved: {ref_path} ({len(audio_data_in)} samples, {sample_rate_in}Hz)")

                # Update voice clone prompt if TTS is initialized
                if app.tts is not None:
                    try:
                        app.tts.update_reference_audio(
                            ref_audio_path=str(ref_path),
                            ref_text=ref_text,
                            language="Japanese",
                        )
                        return f"登録完了! 音声クローンプロンプトを更新しました。({len(audio_data_in) / sample_rate_in:.1f}秒)"
                    except Exception as e:
                        logger.error(f"Voice clone update error: {e}")
                        return (
                            f"音声は保存しましたが、プロンプト更新に失敗しました。\n"
                            f"エラー: {e}\n\n"
                            f"対処法: アプリを再起動するか、音声を再録音してください。"
                        )

                return (
                    f"音声を保存しました: {ref_path}\n"
                    f"TTSが初期化されていないため、アプリを再起動してください。"
                )

            register_voice_btn.click(
                fn=register_voice,
                inputs=[ref_audio_input, ref_text_input],
                outputs=[clone_status],
            )

        # System info accordion
        with gr.Accordion("🔧 システム情報", open=False):
            status_text = gr.Textbox(
                label="初期化ステータス",
                value="初期化中...",
                interactive=False,
            )
            device_info = gr.JSON(
                label="デバイス情報",
                value=get_device_info(),
            )

        # Conversation state
        chat_history = gr.State(value=[])

        # Status HTML helpers
        def make_status(status: str) -> str:
            statuses = {
                "idle": ("⏸️ 待機中", "status-idle"),
                "recording": ("🔴 録音中", "status-recording"),
                "stt": ("🎯 音声認識中...", "status-processing"),
                "llm": ("🤔 回答生成中...", "status-processing"),
                "tts": ("🔊 音声合成中...", "status-processing"),
                "speaking": ("🔈 再生中", "status-speaking"),
            }
            text, css_class = statuses.get(status, ("⏸️ 待機中", "status-idle"))
            return f'<div style="text-align: center; margin: 8px 0;"><span class="status-badge {css_class}">{text}</span></div>'

        # Wrapper for PTT with status updates and chatbot
        def process_ptt_with_status(audio_b64, history):
            if not audio_b64:
                yield None, "", "", history, make_status("idle"), history
                return

            history = history or []

            # STT phase
            yield None, "", "", history, make_status("stt"), history

            user_text = ""
            ai_text = ""
            audio_out = None

            for result in app.process_base64_audio(audio_b64):
                audio_out, user_text, ai_text, _ = result

                if user_text and "生成中" in ai_text:
                    yield audio_out, user_text, ai_text, history, make_status("llm"), history
                elif user_text and "音声を生成中" in ai_text:
                    yield audio_out, user_text, ai_text, history, make_status("tts"), history
                else:
                    yield audio_out, user_text, ai_text, history, make_status("llm"), history

            # Update chat history
            if user_text and ai_text and "⚠️" not in ai_text:
                history = history + [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": ai_text},
                ]

            yield audio_out, user_text, ai_text, history, make_status("idle"), history

        # Wrapper for manual upload
        def process_upload_with_status(audio, history):
            if audio is None:
                yield None, "", "", history, make_status("idle"), history
                return

            history = history or []
            yield None, "", "", history, make_status("stt"), history

            user_text = ""
            ai_text = ""
            audio_out = None

            for result in app.process_audio_streaming(audio):
                audio_out, user_text, ai_text, _ = result

                if user_text and "生成中" in ai_text:
                    yield audio_out, user_text, ai_text, history, make_status("llm"), history
                elif user_text and "音声を生成中" in ai_text:
                    yield audio_out, user_text, ai_text, history, make_status("tts"), history
                else:
                    yield audio_out, user_text, ai_text, history, make_status("llm"), history

            if user_text and ai_text and "⚠️" not in ai_text:
                history = history + [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": ai_text},
                ]

            yield audio_out, user_text, ai_text, history, make_status("idle"), history

        # Event handlers
        ptt_submit_btn.click(
            fn=process_ptt_with_status,
            inputs=[audio_base64_input, chat_history],
            outputs=[audio_output, recognized_text, response_text, chatbot, status_html, chat_history],
        )

        manual_submit_btn.click(
            fn=process_upload_with_status,
            inputs=[audio_input, chat_history],
            outputs=[audio_output, recognized_text, response_text, chatbot, status_html, chat_history],
        )

        def clear_all():
            app.clear_conversation()
            return None, None, "", "", [], "", make_status("idle"), []

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[audio_input, audio_output, recognized_text, response_text, chatbot, audio_base64_input, status_html, chat_history],
        )

        def on_load():
            success, message = app.initialize()
            return f"{'✅' if success else '❌'} {message}"

        demo.load(fn=on_load, outputs=[status_text])

    return demo, theme, custom_css


def main():
    """Main entry point."""
    config = load_config()
    app = VoiceReceptionApp(config)
    demo, theme, custom_css = create_ui(app)

    ui_config = config.get("ui", {})
    host = os.environ.get("GRADIO_SERVER_NAME", ui_config.get("host", "127.0.0.1"))
    port = int(os.environ.get("GRADIO_SERVER_PORT", ui_config.get("port", 7860)))
    share = ui_config.get("share", False)

    logger.info(f"Starting server at http://{host}:{port}")

    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=theme,
        css=custom_css,
        js=PTT_JS,
    )


if __name__ == "__main__":
    main()
