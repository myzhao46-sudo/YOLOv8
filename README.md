# YOLOE ship + YOLOv8m bridge 开发主线

本分支 `distill_pipeline_clean` 保留已经实际运行的双模型流程：冻结 YOLOE ship 基础模型，训练普通 YOLOv8m bridge 专家，再做预测级 late fusion 和自定义 box 评估。类别仍是基础模型 `ship/harbor/tank`、专家本地 `0 bridge`、融合 `0 ship / 1 bridge`。这只是迁移验收基线，最终增强目标类别留待后续研究确认。

**真实验收已通过**：固定源提交与清理后同权重预测完全一致；真实数据 → 1 epoch 训练 → 保存 → 新进程重载新权重 → 基础模型与专家推理 → 融合 → 自定义评估已跑通。证据见 [迁移记录](docs/migration_20260918.md) 和 [结构化验收记录](docs/acceptance_20260918.json)。

**尚未实现**：soft/logit KD、box KD、选择性响应蒸馏、分数校准、可靠性门控。保留的 hard pseudo 是数据准备功能，不是 soft KD；NMS 是预测去重，不是可靠性门控。未添加新算法或学生模型。

## 在当前 Windows 机器运行

原仓库保持在 `C:/Users/DOCTOR/Documents/GitHub/YOLOv8`，开发工作区在相邻的 `YOLOv8_distill_pipeline_clean`。打开后者继续开发。以下 PowerShell 初始化用于后面的所有命令：

```powershell
Set-Location C:/Users/DOCTOR/Documents/GitHub/YOLOv8_distill_pipeline_clean
$py = 'C:/Users/DOCTOR/anaconda3/envs/y8distill/python.exe'
$assets = 'C:/Users/DOCTOR/Documents/GitHub/YOLOv8'
$ship = "$assets/ultralytics/best.pt"
$expert = "$assets/runs/bridge_expert/yolov8m_total_bridge_split_img1024_ep150/weights/best.pt"
$stamp = Get-Date -Format yyyyMMdd_HHmmss
```

一键重新运行固定样本验收，包含与已保存源参照的比较；输出目录自动使用新的时间戳：

```powershell
& $py -B tools/run_clean_smoke.py --asset-root $assets --fixtures runs/migration/fixtures --reference runs/migration/before
```

已执行通过的完整命令还包含 `--out runs/clean_smoke/acceptance_20260918_final`；该目录已存在，不能再覆盖。若只需要新建样本并验收当前分支，可执行下面的独立命令；它不依赖源分支代码或迁移参照：

```powershell
& $py -B tools/run_clean_smoke.py --asset-root $assets
```

smoke 从**既有 train / val** 分别选 12 / 6 张，另从既有 test 选 ship / bridge 各 3 张，均覆盖 RGB / SAR / IR；seed=42，保存原始来源和图像、标签哈希。检查可读性、非空合法标签、类别和所选内容与其他原划分清单的交叉。未重新划分全量数据，test 未参与训练。

默认 `imgsz=1024, batch=2, workers=0, epochs=1, close_mosaic=0, nbs=2, amp=False`。保持原 `YOLO.train()` 和训练器内部验证，增加观察 callback 核实有限 loss、梯度、optimizer step 和参数更新。正式参数不被 smoke 改写。失败会抛异常并返回非零；新权重路径必须存在且在独立进程中使用，零候选会让验收失败。

## 唯一推荐入口

| 步骤 | 入口 | 说明 |
|---|---|---|
| 完整验收 | `tools/run_clean_smoke.py` | 使用下列训练与评估实现；没有重复训练器 |
| 专家训练 | `tools/train/train_bridge_expert_yolo.py` | 保留运行时 YAML；外部训练数据的标签和缓存隔离 |
| 基础模型 box-only 评估 | `tools/eval/eval_yoloeseg_box_only.py` | 参数化 CLI；内部复用 `eval_teacher_ship_external.py`，不调用 YOLOE `model.val()` |
| 专家统一评估 | `tools/eval/eval_bridge_expert_external_yolov8.py` | 普通 Detect 官方 AP 加固定阈值自定义指标，适用同源/外源单类数据 |
| 融合评估 | `tools/eval/eval_late_fusion_ship_bridge.py` | 显式专家 class 0 → 全局 class 1；输出单模型和融合预测 |
| 历史 held-out 口径复核 | `tools/eval/eval_bridge_expert_total_bridge_test.py` | 专用于历史 conf-list 扫描；不与统一 AP 混比 |
| 汇集原 bridge 数据 | `tools/datasets/make_total_bridge_dataset.py` | 原模态数据读取与类别转换 |
| 建立 bridge 固定划分 | `tools/datasets/split_total_bridge_train_val_test.py` | 已有数据正常运行时无需再次执行 |
| 当前两类评估集 / hard pseudo 准备 | `tools/datasets/build_bridge_ship_distill_2cls_v2.py` | 保留真实 frozen-teacher box 响应导出与 bridge 标签映射 |
| NWPU 外源准备 | `tools/datasets/build_external_bridge_nwpu_by_label.py` | 根据真实源类别显式选择与重映射；当前已建资产是 source 1 → target 0，不能盲用默认 9 |

`tools/check/`、`tools/vis/` 和既有 ship / bridge 各模态无重叠数据准备脚本仍保留。旧 student/global4 训练、探针和重复数据构建/导出入口已移除；清单及原因见迁移记录。整个本地 `ultralytics/ultralytics` 框架保留，以维持 YOLOESegModel / YOLOESegment checkpoint 导入路径和兼容性。

## 环境和外部资产

实测环境：Python 3.10.20，PyTorch 2.3.1+cu121，torchvision 0.18.1+cu121，本地 Ultralytics 8.4.32，RTX 4070 Laptop GPU 8GB，驱动 555.97。见 [已观察包版本](docs/environment_observed.json)。没有升级或重新安装环境，也没有依赖原分支 editable install。

每个运行入口优先绑定**本 worktree 的** `ultralytics/ultralytics`；验收打印并断言所有已加载项目模块来源。文本编码器 `mobileclip_blt.ts` 是外部权重，smoke 为其建立本地硬链接或复制，调用真实 `get_text_pe` / `set_classes`。源 checkpoint 的 `pe` 属性为空，但其融合 head 是原始三通道；初始化生成 `[1,3,512]` embedding 后再核查结构、类别顺序与实际分数通道。

资产仍位于 `$assets`，切换 Git 分支不会自动复制它们：

| 资产 | 原仓库相对路径 |
|---|---|
| 冻结基础权重 | `ultralytics/best.pt` |
| 文本编码器 | `mobileclip_blt.ts` |
| YOLOv8m 初始化 | `yolov8m.pt` |
| 历史完整专家 | `runs/bridge_expert/yolov8m_total_bridge_split_img1024_ep150/weights/best.pt` |
| 专家 train / val | `ultralytics/datasets/total_bridge_trainval` |
| 专家 held-out test | `ultralytics/datasets/total_bridge_test` |
| 融合评估集 | `ultralytics/datasets/bridge_ship_distill_2cls_v2` |
| ship val / 外源 | `ultralytics/datasets/ship_small_split` / `ship_extratest_no_overlap` |
| NWPU 单类 bridge | `ultralytics/datasets/external_bridge_nwpu_cls0_from1` |

本机 dataset YAML 明确引用这些外部路径。迁移到其他机器须改 YAML 的 `path` 和显式权重参数；不能把这里的 Windows 命令当作已验证的 Linux 命令。日志、缓存、运行 YAML、预测和新权重位于新工作区的 `runs/`，不提交。

环境有一个既存包声明冲突：`pip check` 报 OpenCV 4.13.0.92 要求 NumPy >=2，当前 NumPy 1.26.4；实际读取、训练、推理均通过。本次未擅自改动它。框架的 AMP 自检曾尝试加载额外 YOLO26n 并跳过检查；最终 smoke 使用 FP32，不依赖该检查模型。后续如要启用 AMP，需要单独验证。

## 正式命令与口径

下面使用上面的 PowerShell 变量。**长训练和下列全量评估未在本次执行**；CLI 已实际核对，训练 YAML 的 2481 / 532 张隔离转换已通过；评估入口均已在真实小子集上执行通过。输出名加 `$stamp`，重新运行前请重新生成时间戳。

当前 8GB 机器的正式专家训练命令：

```powershell
& $py -B tools/train/train_bridge_expert_yolo.py --model "$assets/yolov8m.pt" --data configs/datasets/total_bridge_trainval.yaml --imgsz 1024 --epochs 150 --batch 2 --device 0 --workers 0 --project runs/bridge_expert --name "yolov8m_total_bridge_img1024_ep150_$stamp" --patience 40 --close-mosaic 20 --seed 42 --nbs 64 --no-amp
```

这条命令为本机调整 batch / workers / AMP。历史 `args.yaml` 实际记录的是 150 epoch、1024、batch 16、workers 8、patience 40、close_mosaic 20、seed 42、AMP=true；data 是包装器产生的 `runs/bridge_expert/runtime_total_bridge_trainval.yaml`。这是保存配置还原的等效参数，不能视为找回了完整 shell 原文。

基础模型评估保留旧脚本的 CPU / 640 / 低候选阈值，不默默套用融合的 1024 设置：

```powershell
& $py -B tools/eval/eval_yoloeseg_box_only.py --weights $ship --data configs/datasets/ship_val.yaml --imgsz 640 --device cpu --batch 1 --conf 0.001 --pred-iou 0.7 --match-iou 0.5 --strict-label --out-dir "runs/eval/ship_$stamp"
```

专家统一评估（同源测试）；外源只把 `--data` 换成 `configs/datasets/bridge_external_nwpu.yaml` 并使用新的 name：

```powershell
& $py -B tools/eval/eval_bridge_expert_external_yolov8.py --weights $expert --data configs/datasets/total_bridge_test.yaml --imgsz 1024 --device 0 --batch 2 --conf 0.25 --iou 0.7 --project runs/eval --name "bridge_test_$stamp" --save-json --save-md --save-preds --max-vis 12
```

历史 held-out 阈值扫描，`--iou` 指 NMS，GT 匹配仍是 0.5：

```powershell
& $py -B tools/eval/eval_bridge_expert_total_bridge_test.py --weights $expert --test-root "$assets/ultralytics/datasets/total_bridge_test" --imgsz 1024 --batch 2 --conf-list 0.05 0.10 0.20 0.25 0.35 0.50 --iou 0.7 --device 0 --out-dir "runs/eval/heldout_$stamp"
```

历史完整专家的融合评估：

```powershell
& $py -B tools/eval/eval_late_fusion_ship_bridge.py --ship-weights $ship --bridge-weights $expert --dataset-root "$assets/ultralytics/datasets/bridge_ship_distill_2cls_v2" --imgsz 1024 --batch 2 --device 0 --ship-conf 0.001 --bridge-conf 0.001 --nms-iou 0.7 --report-conf 0.25 --fusion-nms class-aware --project runs/late_fusion --name "ship_bridge_$stamp" --save-preds
```

融合入口使用 101 点插值 AP50 / AP50:95；P/R 在 `report_conf=0.25`、GT IoU=0.5 下计算，混合集包含另一专家产生的 FP。基础模型单独入口及历史 held-out 用原来的阶梯积分 AP；held-out 每档 conf 都重新截断候选。专家统一入口的官方 AP 候选阈值是 0.001，另报 `--conf` 下的 P/R。**不同口径的数字不可直接互换**。本次同权重对照始终使用同一融合评估公式和参数。

当前最终 smoke 权重是 `runs/clean_smoke/acceptance_20260918_final/training/expert/weights/last.pt`；仅验证可运行性，不应替代历史完整专家做性能结论。新分支已在本地提交，不自动合并、不自动推送；需要发布时：

```powershell
git push -u origin distill_pipeline_clean
```
