#!/bin/bash
# 正确启动 NAS 训练（nohup 守护进程，避免被意外杀死）
cd /home/user/yuyue/gra/detection_bdd100k
LOG_DIR=runs/dair_v2x/logs
mkdir -p "$LOG_DIR"

# 如果存在旧日志则备份
if [ -f "$LOG_DIR/nas_yolo.log" ]; then
    mv "$LOG_DIR/nas_yolo.log" "$LOG_DIR/nas_yolo_$(date +%m%d_%H%M%S).log"
fi

nohup python run_phase1_training.py \
    --mode nas \
    --epochs 300 \
    --device 0 \
    --batch 8 \
    --accum-steps 2 \
    --project runs/dair_v2x \
    --name nas_yolo \
    --resume runs/dair_v2x/nas_yolo/epoch5.pt \
    > "$LOG_DIR/nas_yolo.log" 2>&1 &

PID=$!
echo "NAS training started with PID: $PID"
echo $PID > "$LOG_DIR/nas_yolo.pid"
sleep 3
ps -p $PID -o pid,stat,pcpu,pmem,cmd | tail -1
