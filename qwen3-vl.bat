@echo off
chcp 65001 >nul
cd /d D:\llama-cpp\llama.cpp\build\bin\Release

start /min llama-server.exe ^
  -m "D:\models\qwen\Qwen3-VL-30B-A3B-Instruct\Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf" ^
  --mmproj "D:\models\qwen\Qwen3-VL-30B-A3B-Instruct\mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf" ^
  -ngl 999 ^
  -c 4096 ^
  --flash-attn on ^
  --host 0.0.0.0 ^
  --port 8080 


