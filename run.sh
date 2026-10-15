#!/usr/bin/env bash
# Serverni ishga tushirish:  bash run.sh
set -e
cd "$(dirname "$0")"

# fashn-vton o'rnatilgan virtual muhit (torch shu yerda)
VENV="${VTON_VENV:-$HOME/fashn-vton-1.5/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
  echo "XATO: venv topilmadi: $VENV"
  exit 1
fi
source "$VENV/bin/activate"

export VTON_WEIGHTS_DIR="${VTON_WEIGHTS_DIR:-$HOME/fashn-vton-1.5/weights}"
export HF_HOME="${HF_HOME:-$HOME/fashn-vton-1.5/hf_cache}"
PORT="${PORT:-8000}"

IP=$(hostname -I | awk '{print $1}')
echo "=============================================="
echo " Boshqa kompyuterdan oching:  http://$IP:$PORT"
echo "=============================================="

# --workers 1 MUHIM: har bir worker modelni GPU'ga alohida yuklaydi
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1
