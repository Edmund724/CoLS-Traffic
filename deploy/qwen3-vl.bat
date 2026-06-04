@echo off
chcp 65001 >nul
cd /d D:\llama.cpp\build\bin\Release

start /min llama-server.exe ^
  -m "D:\models\Qwen3-VL-30B-A3B-Instruct\Qwen3-VL-30B-A3B-Instruct-UD-Q4_K_XL.gguf" ^
  --mmproj "D:\models\Qwen3-VL-30B-A3B-Instruct\mmproj-BF16.gguf" ^
  -ngl 999 ^
  --jinja ^
  --top-p 0.8 ^
  --top-k 20 ^
  --temp 0.7 ^
  --min-p 0.0 ^
  --presence-penalty 1.5 ^
  -c 4096 ^
  --flash-attn on ^
  --host 0.0.0.0 ^
  --port 8080 


