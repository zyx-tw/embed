<!-- 翻譯自 README.md -->

<div align="center">

<h1>embed</h1>

<p><strong>一個私人服務：把照片和文字變成搜尋向量、找人臉、讀出圖片裡的字。</strong></p>

[English](README.md) | **繁體中文**

</div>

nas 和 roll 送一張照片或一句查詢過來，拿回一組數字。語意用 Google 的 EmbeddingGemma 2，人臉用 InsightFace 的 buffalo_l，文字用 PP-OCR。服務本身不留任何資料。

## 功能

- **文字和照片轉向量**：768 個數字，放在同一個空間，打字就能找到相符的照片。
- **用 Intel 內顯**：視覺模型透過 OpenVINO 跑，在 N150 上約比 CPU 快 2.5 倍。
- **找人臉**：每張臉給框、分數和 512 個數字的向量。
- **讀圖片裡的字**：截圖、菜單、招牌上的中英文。
- **搜尋優先**：文字查詢走自己的通道，不會排在照片後面。

## 快速開始

```sh
sh deploy/install.sh
curl -s http://localhost:8000/
```

第一次啟動會下載模型（約 1.4 GB），並轉換視覺模型（約 1 分鐘）。

## 使用方式

1. 在主機上用 root 執行 `sh deploy/install.sh`，建好 Python 環境。
2. 裝好 land agent(app `embed`)，照 `deploy/env.example` 寫 `/srv/apps/embed/env`，再把 `deploy/hardening.conf` 放進 `/etc/systemd/system/carrel-app-embed.service.d/`。
3. 呼叫 `POST /embed`、`/faces` 或 `/ocr`，帶 `Authorization: Bearer <EMBED_TOKEN>`。

## 設定

| 鍵 | 設定內容 | 預設 |
|----|----------|------|
| `EMBED_TOKEN` | 每個 POST 都要帶的 bearer token | 必填 |
| `EMBED_MODELS` | 模型和 OpenVINO 快取的資料夾 | `/var/lib/embed` |
| `EMBED_THREADS` | 每個模型的 CPU 執行緒數 | `4` |
| `EMBED_PORT` | 監聽的 port | `8000` |
| `EMBED_VISION` | 設成 `cpu` 就不用內顯 | `auto` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP/HTTP JSON log 送去哪裡 | 關閉 |

## 運作方式

內顯需要 Intel 的 compute runtime（`intel-opencl-icd`、`libze-intel-gpu1`、IGC），要從 github.com/intel/compute-runtime 裝，Debian 13 沒有附。

N150 實測（20 張 896 px 照片）：照片 p50 5.2 秒（CPU 要 12.8 秒）、查詢 0.1 秒、人臉 2.0 秒、手機截圖 OCR 1.0 秒。
