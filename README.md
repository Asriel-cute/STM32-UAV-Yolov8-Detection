# 基于STM32平台的低空飞行人体识别智能无人机 🚁

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Hardware](https://img.shields.io/badge/Hardware-STM32F405%20%7C%20STM32MP257-brightgreen)
![AI](https://img.shields.io/badge/AI-YOLOv8-orange)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **9th AI赋能设计, 设计点亮AI! (AI for Design & Design for AI!)**
> 本项目致力于解决传统低空搜救高度依赖人工视觉、效率低且易遗漏遇险人员的痛点，提供一种低成本、高可靠的智能化解决方案。

## 📖 项目简介

本项目设计并实现了一款空地异构协同架构的智能无人机系统。系统将高实时性的飞行控制与高算力的 AI 图像推理分离：
*   **天空端**：基于 STM32F405 飞控，搭载多源传感器（光流+GPS）实现复杂环境下的稳定悬停与精准定位。
*   **地面端**：采用正点原子 STM32MP257 开发板作为“AI大脑”，实时接收图传画面，并利用 NPU 硬件加速运行极度轻量化的 YOLOv8 模型，精准识别人员“正常站立 (Normal)”与“倒地 (Fallen)”状态，为紧急救援提供核心决策。

---

## ✨ 核心特性

*   **⚡ 空地异构协同算力架构**：打破传统算力集中于机载平台的局限，“算力卸载”大幅减轻机身起飞重量，延长续航并提高系统容错率。
*   **🚀 极致边缘 AI 推理性能**：模型经 INT8 量化并下沉至 STM32MP257 的 1.35 TOPS NPU。 Python 核心推理代码采用了**真·多进程 + Ping-Pong 共享内存 + 硬件解码 + 高速 Numpy NMS**架构，彻底榨干硬件性能。
*   **🛰️ 多传感器融合定位**：结合 Foxeer M10Q 250v2 高精度卫星定位与 MTF-02P 光流测速定高，保障近地低空平稳飞行。
*   **🔊 异步报警机制**：检测到人员倒地时，系统内置非阻塞唤起 ALSA 音频播放功能，实现低延迟语音报警，且支持防频发冷却时间设置。

---

## 🧰 硬件系统架构

### 天空端 (空中飞行平台)
*   **主控**：STM32F405 飞控主板 (运行 INAV 固件)
*   **动力系统**：2212 920KV 无刷马达 + Simonk 30A 无刷电调
*   **传感器**：Foxeer M10Q v2 (GPS) + MTF-02P (光流)
*   **图传/数传**：1500TVL 高清摄像头 + 1W 大功率塔式图传 + DL-20 数传模块

### 地面端 (智能主控)
*   **核心板**：正点原子 STM32MP257 (双核 Cortex-A35 + NPU)
*   **显示**：正点原子 5.5寸 MIPI 屏幕 (720*1280)
*   **交互**：迈克 C7 MINI 遥控器

---

## 📊 AI 模型性能指标

模型输入分辨率：`640x640 RGB` | 测试环境：`STM32MP257@1.5GHz + NPU`

| 模型格式 | 模型大小 | 推理时延 (端到端) | mAP@0.5 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| yolov8n.pt | 6.0MB | - | 0.995 | PyTorch 训练原模型 |
| ONNX | 11.8MB | 2682ms | 0.995 | 未优化，单线程 CPU |
| **.nb (INT8)** | **2.7MB** | **85.76ms** | **0.989** | **最终部署版，NPU 硬件加速** |

---

🚀 快速开始 (Quick Start)
1. 飞控端配置
使用 INAV Configurator 连接 STM32F405 飞控。

进入 CLI 命令行界面，将 config/INAV_8.0.1_cli_20260707_113301.txt 的内容粘贴并回车，输入 save 保存。

确保光流与 GPS 状态指示灯正常，并在试飞前完成罗盘校准。

2. 地面端 AI 部署

在STM32MP257上运行推理：

python3 run.py --model model/model.nb
