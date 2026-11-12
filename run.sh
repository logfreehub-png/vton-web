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
# Docker konteynerlari (clothes-shop) host.docker.internal orqali ulanadi, shuning uchun 0.0.0.0.
# Tashqi internetdan himoya: VTON_API_KEY + firewall (8000-portni faqat ichki tarmoqqa oching).
HOST="${HOST:-0.0.0.0}"

if [ -z "${VTON_API_KEY:-}" ]; then
  echo "OGOHLANTIRISH: VTON_API_KEY berilmagan, /api/tryon kalitsiz ochiq!"
  echo "  export VTON_API_KEY=\$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
fi

IP=$(hostname -I | awk '{print $1}')
echo "=============================================="
echo " Boshqa kompyuterdan oching:  http://$IP:$PORT"
echo "=============================================="

exec uvicorn app.main:app --host "$HOST" --port "$PORT" --workers 1
