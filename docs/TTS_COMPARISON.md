# ローカルTTS選択肢比較

> 調査日: 2026-02-02
> 目的: リアルタイム音声応答システムに適したローカルTTSの選定

## 現状の問題

**Qwen3-TTS 1.7B** の課題:
- 生成時間: **30秒以上**（Apple Silicon MPS）
- ストリーミングAPI: **ローカル版では未サポート**
- 「97ms低レイテンシ」は **Alibaba Cloud API専用**

## ローカルTTS比較表

| TTS | パラメータ | 日本語 | レイテンシ | 品質 | ライセンス |
|-----|-----------|--------|-----------|------|-----------|
| **VOICEVOX** | N/A (ONNX) | **26+ キャラ** | **1-3秒** | **高** | 非商用/商用別 |
| **Kokoro** | 82M | 5音声 (C評価) | **数百ms** | 中 | Apache 2.0 |
| **Piper** | 小 (ONNX) | あり | **高速** | 中 | MIT |
| **StyleTTS2** | 中 | 要カスタム | 中 | 高 | MIT |
| **Qwen3-TTS** | 1.7B/0.6B | 高品質 | 15-30秒 | **最高** | Apache 2.0 |

---

## 1. VOICEVOX（推奨）

### 概要
- 日本製の高品質日本語TTS
- **26種類以上のキャラクターボイス**
- 完全ローカル動作
- Apple Silicon対応 (voicevox-cli)

### インストール

```bash
# voicevox_core (Python)
pip install voicevox_core

# Apple Silicon向けCLI
# https://github.com/usabarashi/voicevox-cli
```

### 使用例

```python
from pathlib import Path
from voicevox_core import VoicevoxCore

core = VoicevoxCore(open_jtalk_dict_dir=Path("./open_jtalk_dic_utf_8-1.11"))

# スピーカーID: 2 = 四国めたん（ノーマル）
speaker_id = 2

if not core.is_model_loaded(speaker_id):
    core.load_model(speaker_id)

wave_bytes = core.tts("お電話ありがとうございます", speaker_id)
with open("output.wav", "wb") as f:
    f.write(wave_bytes)
```

### 主なキャラクター（スピーカーID）
| ID | キャラクター | 声質 |
|----|-------------|------|
| 0 | 四国めたん（あまあま） | 女性・甘い |
| 2 | 四国めたん（ノーマル） | 女性・標準 |
| 3 | ずんだもん | 女性・かわいい |
| 8 | 春日部つむぎ | 女性・明るい |
| 13 | 青山龍星 | 男性 |

### メリット
- 日本語に特化した高品質
- 多様なキャラクターボイス
- 活発なコミュニティ
- 商用利用可（一部制限あり）

### デメリット
- 他言語サポートなし
- ライセンスが複雑（キャラクターごと）

### リンク
- [公式サイト](https://voicevox.hiroshiba.jp/)
- [voicevox_core GitHub](https://github.com/VOICEVOX/voicevox_core)
- [voicevox-cli (Apple Silicon)](https://github.com/usabarashi/voicevox-cli)

---

## 2. Kokoro-82M

### 概要
- **82Mパラメータ**（超軽量）
- 8言語・54音声サポート
- Apache 2.0ライセンス
- ストリーミング対応（CLI）

### インストール

```bash
pip install kokoro>=0.9.4 soundfile
apt-get install espeak-ng  # Linux
brew install espeak-ng     # macOS
```

### 使用例

```python
from kokoro import KPipeline
import soundfile as sf

# 日本語: lang_code='j'
pipeline = KPipeline(lang_code='j')

# 日本語音声: jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro, jm_kumo
generator = pipeline("お電話ありがとうございます", voice='jf_alpha')

for i, (gs, ps, audio) in enumerate(generator):
    sf.write(f'output_{i}.wav', audio, 24000)
```

### 日本語音声一覧
| 名前 | 性別 | 品質評価 |
|------|------|---------|
| jf_alpha | 女性 | C+ |
| jf_gongitsune | 女性 | C |
| jf_nezumi | 女性 | C- |
| jf_tebukuro | 女性 | C |
| jm_kumo | 男性 | C- |

### メリット
- 超軽量（82Mパラメータ）
- Apache 2.0（商用自由）
- ストリーミング出力対応
- RealtimeTTS統合可能

### デメリット
- **日本語品質が低い（C評価）**
- トレーニングデータ不足

### リンク
- [HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)
- [GitHub](https://github.com/hexgrad/kokoro)

---

## 3. Piper TTS

### 概要
- Raspberry Pi対応の軽量TTS
- ONNX推論
- **piper-tts-plus**: 日本語強化版

### インストール

```bash
# 基本版
pip install piper-tts

# 日本語強化版
pip install piper-tts-plus

# GPU対応
pip install "piper-tts-plus[gpu]"
```

### 日本語機能（piper-tts-plus）
- OpenJTalk辞書自動ダウンロード
- 技術用語辞書（Docker→ドッカー、GitHub→ギットハブ等）
- オフラインモード対応

### メリット
- 超高速（Raspberry Pi動作可）
- 軽量ONNXモデル
- MIT ライセンス

### デメリット
- 日本語音声モデルの選択肢が限られる
- 音質はやや機械的

### リンク
- [GitHub](https://github.com/rhasspy/piper)
- [piper-tts-plus PyPI](https://pypi.org/project/piper-tts-plus/)
- [音声サンプル](https://rhasspy.github.io/piper-samples/)

---

## 4. StyleTTS2

### 概要
- 人間レベルの音質を目指すTTS
- スタイル拡散モデル
- 音声クローニング対応

### 日本語対応状況
- **Text Aligner**: 日本語対応済み（JVSコーパスで事前学習）
- **PL-BERT**: 英語のみ → **日本語用に再学習が必要**

### メリット
- 高品質・表現力豊か
- 音声クローニング

### デメリット
- 日本語は要カスタムトレーニング
- セットアップが複雑
- GPUが古いとノイズ発生

### リンク
- [GitHub](https://github.com/yl4579/StyleTTS2)
- [PyPI](https://pypi.org/project/styletts2/)

---

## 推奨構成

### ユースケース別推奨

| ユースケース | 推奨TTS | 理由 |
|-------------|---------|------|
| **日本語コールセンター** | VOICEVOX | 最高品質の日本語、キャラ選択可 |
| **超低レイテンシ優先** | Kokoro | 82Mで高速、品質は妥協 |
| **組み込み・省リソース** | Piper | Raspberry Pi対応 |
| **品質最優先（遅延許容）** | Qwen3-TTS 1.7B | 最高品質だが30秒以上 |

### 本プロジェクト推奨

```
VOICEVOX を採用

理由:
1. 日本語品質が最高クラス
2. 1-3秒のレイテンシ（許容範囲）
3. 26種類のキャラクターから選択可能
4. ローカル完結
5. 活発な日本語コミュニティ
```

---

## 実装計画（VOICEVOX採用時）

### 必要な作業

1. **voicevox_core インストール**
   ```bash
   pip install voicevox_core
   ```

2. **Open JTalk辞書ダウンロード**
   - https://jaist.dl.sourceforge.net/project/open-jtalk/Dictionary/

3. **src/tts/voicevox_tts.py 作成**
   - VoskSTTと同様のインターフェース
   - synthesize(text) -> (audio_data, sample_rate)

4. **設定ファイル更新**
   ```yaml
   tts:
     engine: "voicevox"  # qwen / voicevox / kokoro
     voicevox:
       speaker_id: 2  # 四国めたん（ノーマル）
   ```

5. **app.py 統合**
   - TTSエンジン切り替えロジック追加

---

## 参考リンク

- [VOICEVOX 公式](https://voicevox.hiroshiba.jp/)
- [Kokoro-82M HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)
- [Piper GitHub](https://github.com/rhasspy/piper)
- [StyleTTS2 GitHub](https://github.com/yl4579/StyleTTS2)
- [RealtimeTTS GitHub](https://github.com/KoljaB/RealtimeTTS)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
