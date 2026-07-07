#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM32MP257 极致性能版本 YOLO 双类别检测 + 异步音频警报
[真·多进程 + Ping-Pong 共享内存 + 硬件解码 + 高速 Numpy NMS + 异步音频]
"""
import sys
import os
import time
import argparse
import subprocess
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import cv2

# ================= 模型及推断参数 =================
IN_W, IN_H = 640, 640
OUT_SCALE  = 0.003917705733329058  
OUT_ZP     = -127                 
NUM_BOX    = 8400
NUM_ATTR   = 6     # cx, cy, w, h, cls0_score, cls1_score              
CONF_THR   = 0.40  # 置信度阈值              
IOU_THR    = 0.45  # NMS 重叠阈值               

# 类别定义 (0: 倒地人类, 1: 正常人类)
CLASSES = {0: "Fallen", 1: "Normal"}
COLORS  = {0: (0, 0, 255), 1: (0, 255, 0)}

# ================= 核心算法函数 =================
def preprocess(bgr, canvas):
    sh, sw = bgr.shape[:2]
    r  = min(IN_W / sw, IN_H / sh)
    nw = int(round(sw * r))
    nh = int(round(sh * r))
    pad_x = (IN_W - nw) // 2
    pad_y = (IN_H - nh) // 2
    
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    canvas.fill(114)
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = rgb
    tensor = (canvas.astype(np.int16) - 128).astype(np.int8)
    return tensor[None, ...], pad_x, pad_y, r

def nms_numpy(boxes, scores, class_ids, iou_thr):
    if len(boxes) == 0: return [], [], []
    
    x1, y1 = boxes[:, 0], boxes[:, 1]
    w, h   = boxes[:, 2], boxes[:, 3]
    x2, y2 = x1 + w, y1 + h
    areas = w * h
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        ww = np.maximum(0.0, xx2 - xx1)
        hh = np.maximum(0.0, yy2 - yy1)
        inter = ww * hh
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
        
    return boxes[keep].tolist(), scores[keep].tolist(), class_ids[keep].tolist()

def postprocess(raw, pad_x, pad_y, ratio):
    arr = np.asarray(raw)
    
    if arr.dtype == np.int8 or arr.dtype == np.uint8:
        if arr.dtype == np.uint8: arr = arr.view(np.int8)
        arr = (arr.astype(np.float32) - OUT_ZP) * OUT_SCALE

    arr = np.squeeze(arr)
    if arr.shape[0] != NUM_ATTR and arr.shape[-1] == NUM_ATTR: arr = arr.T
    if arr.shape != (NUM_ATTR, NUM_BOX): arr = arr.reshape(NUM_ATTR, NUM_BOX)

    cx, cy, w, h = arr[0], arr[1], arr[2], arr[3]
    cls_scores = arr[4:6, :] 
    
    max_scores = np.max(cls_scores, axis=0) 
    max_idxs = np.argmax(cls_scores, axis=0) 

    mask = max_scores >= CONF_THR
    if not mask.any(): return [], [], []

    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    scores = max_scores[mask]
    class_ids = max_idxs[mask]

    if np.nanmax(np.abs(cx)) <= 2.0: 
        cx, cy, w, h = cx * IN_W, cy * IN_H, w * IN_W, h * IN_H

    x1 = (cx - w * 0.5 - pad_x) / ratio
    y1 = (cy - h * 0.5 - pad_y) / ratio
    ww = w / ratio
    hh = h / ratio

    boxes = np.stack([x1, y1, ww, hh], axis=1).astype(np.int32)
    return nms_numpy(boxes, scores, class_ids, IOU_THR)

# ================= 子进程：纯血 NPU 推理引擎 =================
def infer_worker(shm_name, width, height, model_path, free_queue, ready_queue, result_queue, stop_event):
    print("[NPU Engine] Booting isolated NPU inference core...")
    from stai_mpu import stai_mpu_network
    net = stai_mpu_network(model_path=model_path, use_hw_acceleration=True)
    
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    single_size = height * width * 3
    
    buffers = [
        np.ndarray((height, width, 3), dtype=np.uint8, buffer=existing_shm.buf[:single_size]),
        np.ndarray((height, width, 3), dtype=np.uint8, buffer=existing_shm.buf[single_size:])
    ]
    canvas = np.full((IN_H, IN_W, 3), 114, dtype=np.uint8)

    while not stop_event.is_set():
        try:
            idx = ready_queue.get(timeout=0.5)
        except:
            continue
            
        tensor, px, py, r = preprocess(buffers[idx], canvas)
        free_queue.put(idx) 
        
        t0 = time.perf_counter()
        net.set_input(0, tensor)
        net.run()
        raw = net.get_output(index=0)
        infer_ms = (time.perf_counter() - t0) * 1000
        
        boxes, scores, class_ids = postprocess(raw, px, py, r)
        
        while not result_queue.empty():
            try: result_queue.get_nowait()
            except: pass
        result_queue.put((boxes, scores, class_ids, infer_ms))

    existing_shm.close()

# ================= 主进程：无阻塞渲染与异步报警 =================
def trigger_alarm(audio_path):
    """
    非阻塞唤起 ALSA 音频播放 (适用于嵌入式 Linux)
    将输出重定向至 DEVNULL 防止污染终端
    """
    if os.path.exists(audio_path):
        subprocess.Popen(["aplay", audio_path], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
    else:
        print(f"[Warning] Audio file '{audio_path}' not found!")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to your .nb model")
    ap.add_argument("--dev", default="/dev/video7", help="V4L2 camera device path")
    ap.add_argument("--w", type=int, default=1280, help="Camera width")
    ap.add_argument("--h", type=int, default=720, help="Camera height")
    ap.add_argument("--fps", type=int, default=30, help="Camera framerate")
    ap.add_argument("--audio", type=str, default="alarm.wav", help="Path to alarm .wav file")
    args = ap.parse_args()

    single_size = args.h * args.w * 3
    shm_size = single_size * 2
    shm = None
    
    try:
        shm = shared_memory.SharedMemory(create=True, size=shm_size)
        buffers = [
            np.ndarray((args.h, args.w, 3), dtype=np.uint8, buffer=shm.buf[:single_size]),
            np.ndarray((args.h, args.w, 3), dtype=np.uint8, buffer=shm.buf[single_size:])
        ]
        
        free_queue = mp.Queue()
        ready_queue = mp.Queue()
        result_queue = mp.Queue()
        stop_event  = mp.Event()

        free_queue.put(0)
        free_queue.put(1)

        p_infer = mp.Process(target=infer_worker, 
                             args=(shm.name, args.w, args.h, args.model, free_queue, ready_queue, result_queue, stop_event))
        p_infer.start()

        gst_pipeline = (
            f"v4l2src device={args.dev} ! "
            f"image/jpeg, width={args.w}, height={args.h}, framerate={args.fps}/1 ! "
            f"jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true sync=false max-buffers=1"
        )
        print(f"[UI Pipeline] Booting Hardware Decoder:\n  {gst_pipeline}")

        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            print("[Error] Camera feed cannot be established.")
            sys.exit(1)

        win_name = "Real-time Fall Detection"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        frame_cnt = 0
        t_last = time.perf_counter()
        fps_display = 0.0
        
        current_boxes, current_scores, current_classes, current_infer_ms = [], [], [], 0.0

        # === 警报系统状态变量 ===
        last_alarm_time = 0.0
        ALARM_COOLDOWN = 3.0  # 冷却时间：每隔3秒最多响一次警报，可自行修改

        print("[UI Pipeline] Display loop started. Press 'q' to exit.")
        
        while True:
            ok, frame = cap.read()
            if not ok: continue

            try:
                write_idx = free_queue.get_nowait() 
                buffers[write_idx][:] = frame[:]
                ready_queue.put(write_idx) 
            except:
                pass 

            try:
                current_boxes, current_scores, current_classes, current_infer_ms = result_queue.get_nowait()
            except:
                pass 

            # 检测是否包含倒地的人 (类别 0)
            fall_detected = 0 in current_classes
            
            # 如果检测到倒地，并且冷却时间已到，则触发异步警报
            if fall_detected:
                now_time = time.time()
                if now_time - last_alarm_time > ALARM_COOLDOWN:
                    trigger_alarm(args.audio)
                    last_alarm_time = now_time

            for (x, y, w, h), s, cls_id in zip(current_boxes, current_scores, current_classes):
                label_text = CLASSES.get(cls_id, "Unknown")
                box_color = COLORS.get(cls_id, (255, 255, 255))
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 3)
                
                text_content = f"{label_text} {s:.2f}"
                (tw, th), _ = cv2.getTextSize(text_content, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, max(0, y-25)), (x+tw, y), box_color, -1)
                cv2.putText(frame, text_content, (x, max(0, y-5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # 画面加上警告文字提示 (闪烁效果)
            if fall_detected and int(time.time() * 4) % 2 == 0:  
                cv2.putText(frame, "WARNING: FALL DETECTED!", (args.w//2 - 250, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

            frame_cnt += 1
            now = time.perf_counter()
            if now - t_last >= 0.5:
                fps_display = frame_cnt / (now - t_last)
                frame_cnt = 0; t_last = now

            hud = f"UI Display FPS: {fps_display:.1f} | NPU Latency: {current_infer_ms:.1f}ms"
            cv2.putText(frame, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(win_name, frame)
            
            if (cv2.waitKey(1) & 0xFF) in [ord('q'), 27]:
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("[System] Releasing Shared Memory and Terminating processes...")
        if 'stop_event' in locals(): stop_event.set()
        if 'p_infer' in locals():
            p_infer.join(timeout=2)
            if p_infer.is_alive(): p_infer.terminate()
        if 'cap' in locals(): cap.release()
        cv2.destroyAllWindows()
        if shm is not None:
            shm.close()
            shm.unlink()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()