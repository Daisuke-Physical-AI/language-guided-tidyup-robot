#!/usr/bin/env bash
# 環境構築（本PC / 他の Linux 共通）。システムには何も入れない。
#   1. プロジェクト内 .venv に MuJoCo ほかを入れる（uv があれば uv、無ければ python -m venv）
#   2. MuJoCo Menagerie から Franka Panda だけを取得する（third_party/、git 管理外）
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"

if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv --python "$(command -v python3)"
  else
    python3 -m venv .venv
  fi
fi
if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -r requirements.txt
else
  .venv/bin/pip install -r requirements.txt
fi

PANDA=third_party/mujoco_menagerie/franka_emika_panda
if [ ! -f "$PANDA/panda.xml" ]; then
  mkdir -p third_party
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
  git -C third_party/mujoco_menagerie sparse-checkout set franka_emika_panda
fi

.venv/bin/python -c "import mujoco; print('mujoco', mujoco.__version__)"
echo "OK. 実行例:  MUJOCO_GL=egl .venv/bin/python src/main.py --seed 0"
