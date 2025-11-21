#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="eye_iq_train"
WORKDIR="/code/sa2va_wzx/eye_emotion_iq"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux 会话 ${SESSION_NAME} 已存在，可运行：tmux attach -t ${SESSION_NAME}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}"

tmux send-keys -t "${SESSION_NAME}" "cd ${WORKDIR}" C-m
tmux send-keys -t "${SESSION_NAME}" "python3 -m venv .venv" C-m
tmux send-keys -t "${SESSION_NAME}" "source .venv/bin/activate" C-m
tmux send-keys -t "${SESSION_NAME}" "pip install -r requirements.txt" C-m
tmux send-keys -t "${SESSION_NAME}" "python src/train.py --metadata data/processed/metadata.csv --image-root data/processed/images --epochs 30 --batch-size 64" C-m

echo "已启动 tmux 会话 ${SESSION_NAME}，可使用：tmux attach -t ${SESSION_NAME}"



