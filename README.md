# vton-web — FASHN VTON v1.5 uchun test interfeysi

Serverda ishlaydigan model'ni bir Wi-Fi tarmog'idagi boshqa kompyuter yoki telefondan brauzer orqali sinash uchun.

## Fayl arxitekturasi

```
vton-web/
├── app/                  ← BACKEND (Python, FastAPI)
│   ├── __init__.py       ← bo'sh fayl; "app" papkasini Python paketi qiladi
│   ├── config.py         ← sozlamalar: weights yo'li, max rasm hajmi va h.k.
│   ├── model.py          ← modelni GPU'ga 1 marta yuklaydi, so'rovlarni navbatga qo'yadi
│   └── main.py           ← API: /api/health, /api/tryon + interfeysni beradi
├── static/
│   └── index.html        ← FRONTEND: brauzerda ochiladigan sahifa (HTML+CSS+JS)
├── outputs/              ← natijalar avtomatik saqlanadi (GitHub'ga ketmaydi)
├── requirements.txt      ← qo'shimcha Python paketlar ro'yxati
├── run.sh                ← serverni bitta buyruq bilan ishga tushirish
├── .gitignore            ← GitHub'ga yuklanmaydigan fayllar (weights, outputs)
├── .gitattributes        ← Windows'da tahrirlansa ham .sh fayl buzilmasin
└── README.md
```

So'rov qanday yuradi:

```
Brauzer (index.html)
   │  POST /api/tryon  (odam rasmi + kiyim rasmi + sozlamalar)
   ▼
main.py   → rasmlarni tekshiradi, model.run() ni chaqiradi
   ▼
model.py  → navbat (lock) → TryOnPipeline → GPU (RTX 3090)
   ▼
main.py   → natijani base64 PNG qilib qaytaradi + outputs/ ga saqlaydi
   ▼
Brauzer   → rasmni ko'rsatadi, yuklab olish tugmasi
```

## Serverdagi papkalar joylashuvi

```
~/                        (serverdagi home papka)
├── fashn-vton-1.5/       ← model kodi (GitHub'dan), .venv/ va weights/ shu yerda
└── vton-web/             ← SIZNING kodingiz (shu repo)
```

`vton-web` model papkasining venv'idan foydalanadi, shuning uchun torch qayta yuklanmaydi.

## Serverda ishga tushirish

```bash
cd ~
git clone https://github.com/<USERNAME>/vton-web.git
cd vton-web
source ~/fashn-vton-1.5/.venv/bin/activate
pip install -r requirements.txt
bash run.sh
```

Terminalda `http://192.168.x.x:8000` manzili chiqadi. Shuni boshqa kompyuter brauzerida oching.

Firewall yoqilgan bo'lsa:
```bash
sudo ufw allow 8000/tcp
```

## Kodni yangilash

Kompyuterda: `git add . && git commit -m "..." && git push`
Serverda: `cd ~/vton-web && git pull`, keyin `Ctrl+C` va `bash run.sh`.

## Sozlamalar (ixtiyoriy)

| O'zgaruvchi | Standart | Nima |
|---|---|---|
| `VTON_WEIGHTS_DIR` | `~/fashn-vton-1.5/weights` | Model og'irliklari |
| `VTON_VENV` | `~/fashn-vton-1.5/.venv` | Python muhiti |
| `PORT` | `8000` | Port |
| `VTON_MAX_UPLOAD_MB` | `15` | Bitta rasm limiti |
| `VTON_SAVE_OUTPUTS` | `1` | Natijalarni saqlash (0 = yo'q) |

Masalan: `PORT=9000 bash run.sh`

## Eslatma

Bu test uchun: parol yo'q. Faqat lokal tarmoqda ishlating, routerda portni internetga ochmang.
