# Language-guided tidy-up robot (MuJoCo + local VLM/LLM)

言葉で指示すると、ロボットアーム（Franka Panda）が俯瞰カメラで机を見て、
色つきブロックを指示どおりにトレイへ片付ける。
AI はローカルで動く gemma3:4b（ollama）のみ。クラウド API は使わず、学習もしない（推論のみ）。

```
指示文 ─┐
        ▼
 [観測]  腕を畳んで俯瞰カメラで撮影 → VLM が「何色のブロックがあるか」を答える
        ▼
 [計画]  指示文＋現在の状態 → LLM が「どれを・どのトレイへ」を JSON で返す
        ▼
 [実行]  逆運動学（減衰最小二乗法）で pick & place。物理は MuJoCo
        ▼
 [検証]  もう一度観測して差分を求める。やることが無くなるまで繰り返す（外乱からも回復）
```

## 動かし方

```bash
bash scripts/setup.sh                      # .venv を作り、MuJoCo Menagerie から Panda を取得
ollama pull gemma3:4b-it-q4_K_M            # ollama が起動していること（ollama serve）
MUJOCO_GL=egl .venv/bin/python src/main.py --seed 1 \
    --instruction "暖色系のブロックを左のトレイに、寒色系のブロックを右のトレイに片付けて"
#   → out/run_seed1.mp4（1920x1080・24fps: 上に字幕の帯、下に ロボット視点／ISO／正面 の 3 画面）
#     と、観測・計画・各ステップの結果を記録した out/run_seed1.json
MUJOCO_GL=egl .venv/bin/python src/main.py --seed 4 --disturb \
    --instruction "空の色のブロックだけ左のトレイに入れて"      # 外乱（弾き出し）からの回復
MUJOCO_GL=egl .venv/bin/python src/main.py --layout single --seed 1   # 1 画面＋字幕重ね（1280x720、速い）
```

必要なもの: Python 3.10+、ollama、GPU（なくても動くが遅い）。
Linux ではオフスクリーン描画に `MUJOCO_GL=egl`（または `glfw`）を指定する。
描画は 3 画面で 1 フレーム約 0.5 秒（WSL2 実測）なので、50 秒の動画に 10 分ほどかかる。
ollama の接続先は環境変数 `OLLAMA_URL`（既定 `http://localhost:11434`）で変えられる。

## 構成

| ファイル | 役割 |
|---|---|
| `src/main.py` | 観測→計画→実行→検証のループ。`--disturb` で外乱デモ |
| `src/perception.py` | 俯瞰画像を VLM に見せて物体の色を列挙 |
| `src/planner.py` | 指示文と状態を LLM に渡して手順（JSON）を得る |
| `src/scene.py` | MjSpec で机・トレイ・ブロックを Panda のモデルに足す |
| `src/control.py` | 減衰最小二乗法の逆運動学と pick / place の動作列 |
| `src/video.py` | 俯瞰カメラと 3 画面のカメラ設定、字幕の帯、MP4 書き出し |
| `scripts/setup.sh` | 環境構築（.venv と Panda モデルの取得） |
| `assets/fonts/` | 字幕用フォント Noto Sans JP とそのライセンス |

## 使用したもの・ライセンス
- [MuJoCo](https://mujoco.org/) 3.x（Apache-2.0）、[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) の `franka_emika_panda`（Apache-2.0、`scripts/setup.sh` で取得）
- [ollama](https://ollama.com/) と `gemma3:4b-it-q4_K_M`（Gemma Terms of Use）
- 字幕フォント: Noto Sans JP（SIL OFL 1.1、`assets/fonts/OFL.txt`）
- Python: numpy, imageio(+ffmpeg), Pillow, requests

逆運動学は減衰最小二乗法（Wampler 1986; Buss 2009）を MuJoCo の順運動学とヤコビアンで自作実装した。
コーディングには AI アシスタントを使用した。設計判断・検証・考察は提出者による。
