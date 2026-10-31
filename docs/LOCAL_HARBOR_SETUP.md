# Windows 本机 Harbor 环境

本机使用仓库内 `.venv\Scripts\python.exe`。启动脚本自动定位仓库及 Git，
不再使用旧电脑 `C:\Users\DOCTOR\...` 下的 Python 和权重路径。
仓库内的修改版 Ultralytics 以 editable 方式安装，禁止用在线 Ultralytics 覆盖。

## 环境与路径

| 项目 | 配置 |
| --- | --- |
| Python | 3.10，当前安装为 3.10.11 x64 |
| GPU 依赖 | Torch 2.7.1 + cu128，Torchvision 0.22.1 + cu128 |
| NumPy / OpenCV | 1.26.4 / 4.11.0.86，消除旧环境的 NumPy 版本冲突 |
| 原始 B 权重 | `<仓库>\ultralytics\best.pt` |
| 固定输入 | `D:\RSdataset\_derived\harbor_poc_20261007_124509` |
| 初始化参考 | `D:\RSdataset\_experiments\harbor_poc_20261007_124509` |
| 新实验输出 | `D:\RSdataset\experiments\harbor_rgb_sar_<时间戳>` |
| 安装日志 | `<仓库>\.local-runtime\logs` |
| 本机检查报告 | `<仓库>\reports\local_machine` |

RTX 5060 是 Blackwell / compute capability 12.0。旧机 Torch 2.3.1 + cu121
不适用于本机；本次使用支持该架构的 CUDA 12.8 wheel。
PyTorch wheel 包含所需 CUDA 运行库，本流程不编译 CUDA 扩展，因此无需另装 CUDA 12.1 Toolkit / nvcc。
这属于新硬件适配，不声称与旧环境逐位复现。

官方参考：[NVIDIA GPU 架构](https://developer.nvidia.com/cuda/gpus)、
[PyTorch 2.7 Blackwell 支持](https://pytorch.org/blog/pytorch-2-7/)、
[PyTorch 2.7.1 配套安装命令](https://pytorch.org/get-started/previous-versions/#v271)。

## 准备和检查

在仓库根目录 PowerShell 中执行。安装脚本显示下载进度并保存日志；不会启动训练。
如需重建环境，先准备 Python 3.10 x64，再通过 `-PythonPath` 指定解释器。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\setup_harbor_env.ps1
& .\.venv\Scripts\python.exe -B -X utf8 .\tools\check_harbor_local.py
& .\.venv\Scripts\python.exe -B -X utf8 .\tools\check_harbor_local.py --runtime
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_harbor_local.ps1 -Mode DryRun
```

`--runtime` 检查依赖导入、原 B 的 CPU 加载及通道继承、基础 CUDA 算子，以及
8 张 train 样本的 4-worker 共享缓存读取；不运行模型预测，不训练，不评估 test。
2026-10-31 安装和以上检查均已通过，实际完成情况见
[NEW_MACHINE_CHECK_REPORT.md](../NEW_MACHINE_CHECK_REPORT.md)。
VS Code 解释器路径已设为 `${workspaceFolder}/.venv/Scripts/python.exe`；
如当前窗口仍缓存旧解释器，可用 “Python: Select Interpreter” 选择此路径。

## 后续由用户启动

一轮可行性测试（完整跑 train / val 后暂停，保留 150 轮的学习率计划）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_harbor_local.ps1 -Mode OneEpoch
```

全新正式训练：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_harbor_local.ps1 -Mode Train
```

每次 `OneEpoch` 或 `Train` 都从原 B 初始化新实验。继续本机已暂停的实验时，
把下方路径替换为该次启动打印的真实 `Experiment` 路径，使用 `Resume`，不要再次用 `Train`：

```powershell
$Experiment = 'D:\RSdataset\experiments\harbor_rgb_sar_<实际时间戳>'
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_harbor_local.ps1 -Mode Status -ExperimentDir $Experiment
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_harbor_local.ps1 -Mode Resume -ExperimentDir $Experiment
```

本次准备保留原始类别、样本、框、split、采样、冻结、损失和评估协议。
数据仍放在原来的 D 盘目录，清单内路径已可直接访问，无需改写历史 JSON。
`-DataRoot` 只设置顶层路径，不会自动重写清单内的绝对路径。
旧电脑实验的严格续训身份没有通过完整迁移验证；上面的恢复方法仅用于本机新建的实验。
