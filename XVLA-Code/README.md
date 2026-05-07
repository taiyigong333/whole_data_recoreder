# XVLA-Code

X-VLA 视觉-语言-动作模型部署 — UR7e 机器人抓取项目（代码仓库）

## 目录结构

```
scripts/          推理服务器启动脚本
  start_server_pt.py        X-VLA-Pt 推理服务器
  start_server_libero.py    X-VLA-Libero 推理服务器

evaluation/       评估客户端
  ur5e_client.py            UR5e MuJoCo 仿真评估客户端
  results.json              LIBERO libero_spatial 评估结果 (97.4%)

config/           配置文件
  ur5e_scene.xml            UR5e + Robotiq 2F-85 MuJoCo 仿真场景

models/           模型相关（不放权重文件）
```

## 环境要求

- Python 3.10
- PyTorch + CUDA
- 详见 X-VLA 上游仓库

## 相关仓库

- 上游模型: https://github.com/2toINF/X-VLA
- 采集数据变为lerobot dataset v2.1 https://github.com/taiyigong333/whole_data_recoreder
- 一些关于机器人操作的飞书连接：https://ycnnggob6r58.feishu.cn/wiki/Mc13wngA6iSpYlk0sMhcCa2unnh
- 完成的机器人操作：先留个空