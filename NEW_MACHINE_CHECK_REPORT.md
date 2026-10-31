# 新电脑环境与路径检查报告

检查日期：2026-10-31（Asia/Shanghai）。**依赖安装和训练前准备检查通过，尚未启动训练。**
本轮未执行一轮/五轮训练、模型预测、门控或最终 test；不把环境检查视为训练结果。

## 环境

| 项目 | 本机实测 |
| --- | --- |
| Python | 3.10.11 x64 |
| 解释器 | `C:\Users\wangwei\Documents\GitHub\YOLOv8\.venv\Scripts\python.exe` |
| GPU / 驱动 | NVIDIA GeForce RTX 5060，8151 MiB，610.60 |
| GPU 架构 | compute capability 12.0；Torch wheel 包含 `sm_120` |
| CPU / 内存 | Intel Core i7-14700，20 核 / 28 线程；系统可见约 23.66 GiB |
| Torch / Torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| CUDA / cuDNN 运行库 | 12.8 / 90701（9.7.1） |
| NumPy / OpenCV | 1.26.4 / 4.11.0.86 |
| Pillow / Matplotlib / PyYAML | 12.1.1 / 3.10.8 / 6.0.3 |
| Ultralytics | 8.4.32，editable 安装本仓库修改版 |
| 实际 Ultralytics 导入 | `C:\Users\wangwei\Documents\GitHub\YOLOv8\ultralytics\ultralytics\__init__.py` |

Python 来自已验证签名的 Python Software Foundation 官方安装程序。
PyTorch 来自官方 CUDA 12.8 wheel，安装前验证 SHA256：
`5174f02de8ca14df87c8e333c4c39cf3ce93a323c9d470d690301d110a053b3c`。
其余依赖由 pip 从官方 PyTorch 索引 / PyPI 安装；完整版本和直接安装来源见
[pip-freeze.txt](reports/local_machine/pip-freeze.txt) 与 [pip-inspect.json](reports/local_machine/pip-inspect.json)。

与旧机相比，Python 从 3.10.20 改为 3.10.11，Torch/Torchvision 从
2.3.1+cu121 / 0.18.1+cu121 改为 2.7.1+cu128 / 0.22.1+cu128，以适配新 GPU。
OpenCV 从 4.13.0.92 改为 4.11.0.86，保留 NumPy 1.26.4 并消除旧环境的依赖冲突。
部分间接依赖版本也不同，完整差异记入机器可读报告；不声称严格复现旧环境。
此训练流程不编译 CUDA 扩展，所需运行库由 Torch wheel 提供，没有另装 CUDA 12.1 Toolkit。
硬件适配依据见 [NVIDIA GPU 表](https://developer.nvidia.com/cuda/gpus) 和
[PyTorch Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)。

## 数据和路径

| 用途 | 本机路径 |
| --- | --- |
| 原始 B | `C:\Users\wangwei\Documents\GitHub\YOLOv8\ultralytics\best.pt` |
| 固定数据 | `D:\RSdataset\_derived\harbor_poc_20261007_124509` |
| 初始化参考实验 | `D:\RSdataset\_experiments\harbor_poc_20261007_124509` |
| 后续新实验输出 | `D:\RSdataset\experiments` |

旧 B 路径 `C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\best.pt`
已在新启动脚本中替换为当前仓库相对路径。D 盘保留原命名空间，清单内实际绝对路径均可访问；
无需编辑历史 JSON、重新划分数据或重算清单哈希。

原 B 为 23,885,172 字节，SHA256：
`50f306b0e2e50de49fe632377b8b4adcdd8c4f6eef50af3cff075a6a9b9415bd`。
19 个必需关键文件的大小与 SHA256 均匹配交接清单。

| split | RGB 图 / harbor 框 | SAR 图 / harbor 框 |
| --- | --- | --- |
| train | 318 / 1117 | 1290 / 3199 |
| val | 328 / 1205 | 161 / 391 |
| test（仅资产检查） | 810 / 3098 | 160 / 385 |

3,067 条记录的 12,268 个图像/来源注释/标签引用均存在；框范围检查通过。
train/val/test 清单字节哈希与原件相同，因此成员、框和类别未改变。
原始 DIOR（70,392 文件、7,846,302,896 字节）、SAR（116,601 文件、7,242,233,769 字节）
和固定派生数据（6,147 文件、988,265,489 字节）的盘点结果均与交接清单一致。
此盘点未对每张原始图像逐一解码或计算内容哈希。
可选的 `mobileclip_blt.ts` 缺失；当前已融合分类器的 harbor 流程不依赖它。

## 已通过的检查

- `pip check`：`No broken requirements found.`
- 依赖导入及实际仓库导入路径检查。
- 原 B 的 CPU 加载：YOLOESegModel，原类别 0=ship / 1=harbor / 2=tank。
- CPU 上转换为单类 harbor 专家：继承原通道 1，新随机参数为 0；可训练参数 4,085,571。
- 合成张量的 CUDA 矩阵运算、卷积反向传播和 torchvision CUDA NMS。
- 实际 train 清单固定 8 张样本的读取：640、batch=4、workers=4、shared cache、persistent workers。
- CLI help、启动脚本 DryRun、PowerShell 语法和 Git 空白检查。

CUDA 算子和小批数据读取通过，尚不等于完整 epoch 已验证。
完整训练初始化的 B/E0 GPU 输出一致性、正式训练及完整 val 将在用户启动训练时执行。
本轮没有创建新训练实验或修改旧实验。

## 代码与续训边界

HEAD：`d1a300cd5064be6829d55f08d1e6f55bc69b7c19`，分支：`harbor_expert_poc`。
本机新增安装、检查、启动脚本及说明；更新 `.gitignore` 和 VS Code 解释器配置。
训练核心源码、模型、科学参数、冻结策略和评估协议未修改；未提交或推送 Git。

新建 run 的环境与路径准备检查通过。**旧机 run 的严格续训仍未认证**：
历史快照的 25 个源码文件仅 7 个原始字节哈希匹配，25 个均在规范化换行后内容相同；
此外旧 run 仍记录旧解释器/权重路径，并且本机 Torch 与硬件不同。
没有绕过身份校验，也没有改写旧 checkpoint。新机实验建立后可用其自身保存的配置和身份续训。

## 后续命令

在仓库根目录执行。默认 DryRun 只显示配置：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_harbor_local.ps1 -Mode DryRun
```

需要开始一轮可行性测试时（150 轮学习率计划，完整第 1 轮后暂停）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_harbor_local.ps1 -Mode OneEpoch
```

需要直接开始新的正式训练时：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_harbor_local.ps1 -Mode Train
```

两种训练命令都会新建实验。继续已暂停的新机实验时使用 `-Mode Resume -ExperimentDir <本次实际实验目录>`。
更多说明见 [LOCAL_HARBOR_SETUP.md](docs/LOCAL_HARBOR_SETUP.md)。

本次安装日志：[setup_20261031_182624.log](.local-runtime/logs/setup_20261031_182624.log)。
检查产物：[资产检查](reports/local_machine/asset_check.json)、
[运行时检查](reports/local_machine/runtime_check.json)、[DryRun](reports/local_machine/dry_run.json)、
[机器可读报告](reports/local_machine/NEW_MACHINE_CHECK_REPORT.json)。
