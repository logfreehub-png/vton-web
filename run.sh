#!/usr/bin/env bash
# Serverni ishga tushirish:  bash run.sh
# Eslatma: model ComfyUI ichida ishlaydi (comfyui.service, 127.0.0.1:8188).
set -e
cd "$(dirname "$0")"

# vton-web uchun alohida yengil venv (fastapi, uvicorn, pillow)
VENV="${VTON_VENV:-$PWD/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -r requirements.txt
fi
source "$VENV/bin/activate"

PORT="${PORT:-8000}"

IP=$(hostname -I | awk '{print $1}')
echo "=============================================="
echo " Boshqa kompyuterdan oching:  http://$IP:$PORT"
echo "=============================================="

exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1
