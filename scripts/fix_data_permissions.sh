#!/usr/bin/env bash
# 修复 data/ 目录权限（需 sudo 密码）
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Fixing ownership: ${ROOT_DIR}/data → $(id -un)"
sudo chown -R "$(id -un):$(id -gn)" "${ROOT_DIR}/data"
echo "Done. CSV/NCU 也可继续使用 build/data/ 作为回退路径。"
