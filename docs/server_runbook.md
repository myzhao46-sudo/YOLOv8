# 项目现状与服务器运行命令

## 现在的模型究竟是什么

当前可运行基线：真实数据/固定划分 → 冻结 YOLOE ship 基础模型 + 单独训练 YOLOv8m bridge 专家 → 两者分别预测 → 类别映射 → late fusion/NMS → 自定义评估。

| 对象 | 初始化/类别 | 是否训练 | 输出 |
|---|---|---|---|
| 基础模型 | `ultralytics/best.pt`，YOLOESegModel + YOLOESegment，原类 `0 ship / 1 harbor / 2 tank` | 冻结；本项目当前基线不重新训练它 | 通过真实文本 embedding 初始化后只消费 ship 框 |
| 新 bridge 专家 | 从原始 `yolov8m.pt` 初始化，普通 YOLOv8m Detect，`0 bridge` | 本次150 epoch计划训练的就是它，使用真实 bridge GT | 独立 `best.pt` / `last.pt` |
| “混合/融合模型” | 同时加载以上两个 checkpoint，专家本地0映射为全局1 | 没有独立训练，也没有融合 checkpoint | `0 ship / 1 bridge` 的融合预测、指标和日志 |
| hard pseudo | 可选 v2 数据准备工具导出教师硬框 | 不参与当前专家正式训练 | 可选伪标签；不等于 soft KD |
| 完整研究方法 | 选择性保留类响应蒸馏、soft/logit KD、box KD、非对称可靠性门控、校准、目标域适配 | 尚未实现 | 当前没有这些模块的有效训练命令 |

专家损失是原 `v8DetectionLoss` 的 box/cls/dfl，经原 trainer backward 和 optimizer 更新；没有教师损失。NMS只是去重，不是可靠性门控。最终增强类别研究仍未确定，本次继续 ship/bridge，不改成 tank。

用户已要求停止本机训练，现无本任务训练进程。正式训练从 YOLOv8m 初始化后完成1个完整 epoch，曾在第2 epoch中断并恢复，随后按用户要求停止；最后完整检查点仍是 epoch1，保存的有效 optimizer steps=301。这些未完成权重不能当作正式专家。历史完整权重的全量评估已经完成，新专家完整训练/全量测试留给服务器。

本机结果：`runs/baseline_verification/20260918_formal_v2/`；详细四张表见 [正式验收记录](baseline_verification.md)。核心结果：统一 bridge AP50=0.8122298843、NWPU AP50=0.2957840684；旧融合口径 combined AP50=0.7817934426，与历史JSON一致。清理前后832张旧行为完全一致。随后修复两张 EXIF 图像的专家坐标方向，最终 combined AP50=0.7818213803；这是明确的兼容修复影响，不是新专家提升。

## 服务器路径、环境、配置先准备好

以下是 **Bash 命令，参数已核对当前代码，尚未在你的服务器执行**。假定历史资产仍在 `/root/autodl-tmp/YOLOv8`，新分支代码放在相邻独立 worktree。若真实位置不同，只修改下面的 `REPO` 和 `ASSETS`。不要原样使用本仓库 YAML 中的 Windows 路径。

本机分支为 `distill_pipeline_clean`。若通过 Git 传代码，由你先在本机执行 `git push -u origin distill_pipeline_clean`；本任务没有推送。服务器尚未建立新目录时，可以从原仓库执行：

```bash
git -C /root/autodl-tmp/YOLOv8 fetch origin
git -C /root/autodl-tmp/YOLOv8 worktree add --detach \
  /root/autodl-tmp/YOLOv8_distill_pipeline_clean origin/distill_pipeline_clean
```

这是独立、固定提交的运行目录；原工作区不切分支。若目录已经存在，先检查它的 `git status` 和 `git log -1`，不要覆盖。数据和权重不会由 Git 自动带过去。

先激活服务器上已有的可用 Python 环境；不要求环境名字叫 y8distill，也不自动升级整个环境。随后执行：

```bash
set -euo pipefail
export REPO=/root/autodl-tmp/YOLOv8_distill_pipeline_clean
export ASSETS=/root/autodl-tmp/YOLOv8
export OUT="$REPO/runs/server_baseline_$(date +%Y%m%d_%H%M%S)"
export PY="$(command -v python)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export PYTHONUTF8=1
export YOLO_AUTOINSTALL=false
export YOLO_OFFLINE=true
cd "$REPO"
mkdir "$OUT"
mkdir "$OUT/logs"
export YOLO_CONFIG_DIR="$OUT/settings"
mkdir "$YOLO_CONFIG_DIR"
git rev-parse HEAD | tee "$OUT/code_commit.txt"

"$PY" -B - <<'PY'
import os, yaml
from pathlib import Path
from tools.clean_pipeline.runtime import setup_runtime, assert_local_imports, sha256
setup_runtime()
import torch, ultralytics
repo, assets, out = (Path(os.environ[k]).resolve() for k in ('REPO','ASSETS','OUT'))
print('Python:',os.sys.executable)
print('Torch/CUDA:',torch.__version__,torch.version.cuda)
assert torch.cuda.is_available(), 'This formal command requires CUDA'
print('GPU:',torch.cuda.get_device_name(0))
print('Ultralytics:',ultralytics.__version__,ultralytics.__file__)
assert_local_imports()
weights = [assets/'yolov8m.pt', assets/'ultralytics/best.pt',
 assets/'mobileclip_blt.ts',
 assets/'runs/bridge_expert/yolov8m_total_bridge_split_img1024_ep150/weights/best.pt']
for p in weights:
    assert p.is_file(), p
    print(p, sha256(p))
text = repo/'mobileclip_blt.ts'
if not text.exists():
    text.symlink_to(assets/'mobileclip_blt.ts')
assert sha256(text)==sha256(assets/'mobileclip_blt.ts')
config_dir=out/'configs'; config_dir.mkdir()
for name,dataset in {
 'total_bridge_trainval.yaml':'total_bridge_trainval',
 'total_bridge_test.yaml':'total_bridge_test',
 'bridge_external_nwpu.yaml':'external_bridge_nwpu_cls0_from1',
}.items():
    data=yaml.safe_load((repo/'configs/datasets'/name).read_text(encoding='utf-8'))
    path=assets/'ultralytics/datasets'/dataset
    assert path.is_dir(),path
    data['path']=str(path)
    (config_dir/name).write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
ship_data={'path':str(assets/'ultralytics/datasets/bridge_ship_distill_2cls_v2'),
           'val':'images/test_ship','nc':1,'names':{0:'ship'}}
(config_dir/'ship_test.yaml').write_text(yaml.safe_dump(ship_data,sort_keys=False),encoding='utf-8')
from tools.clean_pipeline.baseline import asset_inventory
inventory=asset_inventory(assets,out)
print('Verified full dataset:',inventory['splits'])
PY

export SHIP="$ASSETS/ultralytics/best.pt"
export OLD_EXPERT="$ASSETS/runs/bridge_expert/yolov8m_total_bridge_split_img1024_ep150/weights/best.pt"
export EVAL_DATA="$ASSETS/ultralytics/datasets/bridge_ship_distill_2cls_v2"
```

此步骤只生成本次输出目录下的服务器 YAML；不修改原数据或既有划分。应得到 train2481/val532、bridge test532/798框、ship test300/694框、NWPU67/124框；combined832/1492框。数据不足会明确失败，不抽样凑数。训练器仍会生成自己的 runtime YAML，并隔离标签缓存。

`bridge_external_nwpu.yaml` 的历史 train/val/test 字段都指外测目录；**它只交给评估入口，不能交给训练入口**。同样，`total_bridge.yaml` 的汇集配置不能替代已经分开的 `total_bridge_trainval.yaml`。

## 训练一个全新的 bridge 专家

下面恢复历史的 batch16/workers8，并显式沿用本次验证可用的 FP32。**服务器此组合尚未实测**；FP32比历史AMP更占显存，若显存不足，按16→8→4→2降低 batch并记录，Windows稳定配置为batch2/workers0。不要为了凑历史指标反复挑 seed。

```bash
"$PY" -B tools/train/train_bridge_expert_yolo.py \
  --model "$ASSETS/yolov8m.pt" \
  --data "$OUT/configs/total_bridge_trainval.yaml" \
  --imgsz 1024 --epochs 150 --batch 16 \
  --device 0 --workers 8 \
  --project "$OUT/training" --name expert \
  --patience 40 --close-mosaic 20 --seed 42 \
  --nbs 64 --no-amp --verify-updates \
  2>&1 | tee "$OUT/logs/train.log"

export NEW_EXPERT="$OUT/training/expert/weights/best.pt"
test -f "$NEW_EXPERT"
```

`set -o pipefail` 保证训练失败时，不会被 `tee` 的成功状态掩盖。正式训练以原验证集选 best；150是最大轮数，patience40触发提前停止也是正常结束。

| 参数 | 含义 |
|---|---|
| `--model yolov8m.pt` | 原始预训练初始化；不是历史bridge完整专家，也不是上次smoke模型 |
| `--data ...trainval.yaml` | 既有真实train/val；不重新切分，不读最终test参与训练 |
| `--imgsz 1024` | 训练目标输入尺寸；保持历史专家设置 |
| `--epochs 150` | 最多训练150轮，实际轮数可能因早停减少 |
| `--batch 16` | 每次送入模型的图像数；影响显存、BatchNorm与训练过程 |
| `--device 0` | 当前可见GPU的第0号 |
| `--workers 8` | 数据加载工作进程数；内存/平台受限可降至0 |
| `--project ... --name expert` | 输出目录为project/expert；默认拒绝覆盖已有run |
| `--patience 40` | 验证fitness连续40轮不改善则早停；不使用test选择模型 |
| `--close-mosaic 20` | 计划最后20轮关闭mosaic；早停提前发生时未必进入该阶段 |
| `--seed 42` | 固定随机种子；不能保证不同设备、精度、恢复运行逐位相同 |
| `--nbs 64` | 原优化器名义batch设置，参与梯度累积/衰减；warmup后batch2通常约累积32步，不等价于完全相同的batch64训练 |
| `--no-amp` | FP32，关闭混合精度；历史150轮args.yaml为AMP=true，条件差异必须记录 |
| `--verify-updates` | 观察有限loss/梯度、真实optimizer step、参数更新并保存证据；不引入新损失 |

当前包装器不暴露独立 `--optimizer`、`--lr0` 等参数，不要添加不支持的 flags；沿用本地框架默认/auto，并在实际 args.yaml、日志中核对选出的优化器。

完成后核对 `train_summary.json` 的 `completed/actual_epochs/best_epoch/termination_reason`，`training_update_audit.json` 的 passed、更新与loss，和 `results.csv`。`best.pt`用于最终评估，`last.pt`主要用于未完成训练恢复；不要误用本机epoch1的未完成权重作最终基线。

仅当服务器上**本次run中断**且没有仍在运行的训练进程时，可以恢复同一run：

```bash
"$PY" -B tools/train/train_bridge_expert_yolo.py \
  --model "$ASSETS/yolov8m.pt" \
  --data "$OUT/configs/total_bridge_trainval.yaml" \
  --imgsz 1024 --epochs 150 --batch 16 --device 0 --workers 8 \
  --project "$OUT/training" --name expert \
  --patience 40 --close-mosaic 20 --seed 42 --nbs 64 \
  --no-amp --verify-updates --exist-ok \
  --resume-from "$OUT/training/expert/weights/last.pt" \
  2>&1 | tee "$OUT/logs/resume_$(date +%Y%m%d_%H%M%S).log"
```

恢复前保留上一份日志、args.yaml、training_progress.json和检查点。若之前降低了batch/workers，这里也使用相同实际值。Windows本次run不能直接搬到Linux后无修改resume：checkpoint内保存了Windows运行时路径；服务器推荐按上面从原初始化新开正式run。

## 新专家完整同源测试与 NWPU 外测

以下均在训练正常结束、NEW_EXPERT存在后运行；如只评估历史完整专家，改为 `EXPERT="$OLD_EXPERT"` 并给评估 `--name` 换新名称。

```bash
export EXPERT="$NEW_EXPERT"

"$PY" -B tools/eval/eval_bridge_expert_external_yolov8.py \
  --weights "$EXPERT" --data "$OUT/configs/total_bridge_test.yaml" \
  --imgsz 1024 --batch 8 --device 0 --conf 0.25 --iou 0.7 \
  --project "$OUT/eval" --name bridge_test \
  --save-json --save-md --save-preds --max-vis 12 --max-gt-preview 0 \
  2>&1 | tee "$OUT/logs/bridge_test.log"

"$PY" -B tools/eval/eval_bridge_expert_external_yolov8.py \
  --weights "$EXPERT" --data "$OUT/configs/bridge_external_nwpu.yaml" \
  --imgsz 1024 --batch 8 --device 0 --conf 0.25 --iou 0.7 \
  --project "$OUT/eval" --name nwpu_external \
  --save-json --save-md --save-preds --max-vis 12 --max-gt-preview 0 \
  2>&1 | tee "$OUT/logs/nwpu_external.log"
```

官方AP候选下限为 .001；`--conf .25`用于另外的固定阈值P/R统计。`--iou .7`是NMS IoU，GT匹配为 .5。官方最佳F1工作点的P/R与固定.25的P/R分别保存。NWPU只外测，不训练、不调阈值、不挑模型。

## 融合：加载两个模型，预测后融合，不训练第三个模型

```bash
"$PY" -B tools/eval/eval_late_fusion_ship_bridge.py \
  --ship-weights "$SHIP" --bridge-weights "$NEW_EXPERT" \
  --dataset-root "$EVAL_DATA" \
  --imgsz 1024 --batch 2 --device 0 \
  --ship-conf 0.001 --bridge-conf 0.001 \
  --report-conf 0.25 --nms-iou 0.7 --max-det 300 \
  --fusion-nms class-aware \
  --project "$OUT/late_fusion" --name new_expert --save-preds \
  2>&1 | tee "$OUT/logs/fusion.log"
```

复现历史完整专家融合时，把 `--bridge-weights` 换成 `$OLD_EXPERT`，`--name` 换成 `historical_expert`。两套权重不能混称为同一次实验。入口评估完整 test_ship、test_bridge、combined，保存预测和指标。combined含跨场景未匹配候选，不能把两个独立数据集AP简单平均。

`--ship-conf/--bridge-conf` 是生成AP候选的最低分数；`--report-conf`是报告P/R的阈值；`--nms-iou`用于去重；GT IoU=.5对应AP50，.5:.05:.95对应mAP50-95。class-aware按类别NMS，class-agnostic跨类别去重；二者都不是可靠性门控。没有一条“训练融合checkpoint”的命令。

## 单独基础模型 box-only 评估

```bash
"$PY" -B tools/eval/eval_yoloeseg_box_only.py \
  --weights "$SHIP" --data "$OUT/configs/ship_test.yaml" \
  --imgsz 1024 --device 0 --batch 2 \
  --conf 0.001 --pred-iou 0.7 --match-iou 0.5 --max-det 300 \
  --target-class 0 --target-name ship --strict-label \
  --out-dir "$OUT/eval/ship_box_only" \
  2>&1 | tee "$OUT/logs/ship_box_only.log"
```

此独立入口保留历史阶梯积分AP，其P/R在候选conf下计算，不应拿它直接比较融合入口的101点AP/报告conf=.25。上面的服务器命令明确选择GPU/1024；脚本自身旧默认仍是CPU/640。若要对应历史 .9216，使用融合评估中的 test_ship/class0，保持其完整NMS与101点口径。

## 可选：保存统一候选，并无额外推理地比较两种NMS

这是验收适配器，不是新的训练器或另一套指标算法：

```bash
"$PY" -B tools/clean_pipeline/evaluate.py \
  --dataset-root "$EVAL_DATA" --ship-weights "$SHIP" \
  --bridge-weights "$NEW_EXPERT" --imgsz 1024 --batch 2 --device 0 \
  --ship-conf 0.001 --bridge-conf 0.001 --report-conf 0.25 \
  --nms-iou 0.7 --max-det 300 --fusion-nms class-aware \
  --out "$OUT/candidate_audit" \
  2>&1 | tee "$OUT/logs/candidate_audit.log"

"$PY" -B - <<'PY'
import os
from pathlib import Path
from tools.clean_pipeline.runtime import setup_runtime
setup_runtime()
from tools.clean_pipeline.baseline import replay
replay(Path(os.environ['OUT'])/'candidate_audit')
PY
```

缓存在专家各自NMS之后、跨专家融合NMS之前，与原融合对照等价；保存 class-aware/class-agnostic 及分模态结果。若评估第二个专家，可用适配器 `--ship-cache` 指向同协议、同数据、同基础权重的候选目录，它会校验代码/预测/资产指纹。

Windows用的一键 `run_baseline_verification.py --legacy-reference ...` 依赖本机固定源参照的绝对路径和资产记录；**不要把Windows缓存直接拿到Linux冒充跨平台已验证结果**。服务器按本手册的独立正式入口运行即可。

## 历史可选工具

- `eval_bridge_expert_total_bridge_test.py`：保留旧 `--conf-list` 扫描及阶梯积分，仅用于明确的历史口径复查，不用它挑本次test阈值。
- `build_bridge_ship_distill_2cls_v2.py --generate-ship-pseudo`：可选硬伪标签准备；当前监督专家和融合不需要运行它。
- `make_total_bridge_dataset.py`、`split_total_bridge_train_val_test.py`、NWPU构建器与模态外测构建器：数据来源和固定划分准备工具。既有资产完整时不必重跑，不覆盖GT、不重新划分。
- `run_clean_smoke.py`：12/6小样本可运行性检查，不能替代正式训练或全量指标。

所有服务器命令目前都是“已核对入口，未在服务器执行”。本机实际启动的正式训练参数为batch2/workers0/FP32，其余正式设置相同；完整 argv 保存在 state.json。原权重、数据、旧结果均保留。
