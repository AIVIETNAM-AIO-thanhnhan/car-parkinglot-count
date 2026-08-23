#!/usr/bin/env bash
# Tải PKLot (~2GB) về Google Drive. Chỉ P2 chạy, một lần.
set -euo pipefail
DEST="${1:-/content/drive/MyDrive/pklot_project/raw}"
mkdir -p "$DEST" && cd "$DEST"

if [ -d PKLot ]; then echo "Đã có PKLot/ — bỏ qua."; exit 0; fi

echo "Đang tải PKLot (~2GB, 15-40 phút)..."
wget -c http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
tar -xzf PKLot.tar.gz && rm -f PKLot.tar.gz

echo "Kiểm tra:"
find PKLot -name '*.jpg' | wc -l | xargs echo "  ảnh:"
find PKLot -name '*.xml' | wc -l | xargs echo "  nhãn:"
