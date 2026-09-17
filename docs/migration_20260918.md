# 2026-09-18 迁移与验收

源分支 `bridge_clean_official`，固定源提交 **736cb86c865fa490e9ae77e3189b8098825897fd**。启动时原工作区干净，没有需要迁入的 staged / unstaged / untracked 代码；空补丁和状态记录保存在 `runs/migration/`。未拉取更新。

- 原工作区：`C:/Users/DOCTOR/Documents/GitHub/YOLOv8`。
- 新分支及 worktree：`distill_pipeline_clean`，`C:/Users/DOCTOR/Documents/GitHub/YOLOv8_distill_pipeline_clean`。
- detached 源对照 worktree：`C:/Users/DOCTOR/Documents/GitHub/YOLOv8_source_736cb86`，固定在上述 SHA。tracked 源文件未改；只添加验收适配器和本地文本权重链接。适配器复用该源快照的真实推理、融合、GT 匹配和指标函数，所有项目模块来源均断言在该 worktree。

恢复文件应使用固定 SHA，例如在新工作区运行 `git restore --source=736cb86c865fa490e9ae77e3189b8098825897fd -- 文件路径`；不要使用移动的分支头。本次无需从源补回误删文件：依赖审计先保留了真正需要的辅助模块和框架。未创建 legacy 目录，没有重写历史、覆盖原分支或提交任何数据/权重。

## 留存与清除依据

保留专家 `YOLO.train()` → `DetectionTrainer` → `DetectionModel.init_criterion()` → `v8DetectionLoss` → 原 trainer backward / optimizer 路径。损失是有 GT 的 box / cls / dfl；没有教师响应输入、KD 加权项或额外 KD 优化器参数。旧 `train_yolov8m_2cls_from_bridge_init.py` 也仅使用标准监督训练和硬伪标签，不构成已实现的 soft KD。源记录中的该 student 仅有 preflight 通过，不能据此宣称训练成功。

保留 `eval_teacher_ship_external.py`，因为 `eval_late_fusion_ship_bridge.infer_ship()` 和 v2 数据构建器实际使用其中的 YOLOESegment 原始 box 响应、letterbox 和 NMS；将 `eval_yoloeseg_box_only.py` 收为唯一推荐基础评估 CLI，复用该实现。保留整个本地框架及序列化类路径，无 head 替换、seg→detect 转换或 YOLOE `model.val()`。

删除旧 global4 / seg→detect / 两类 student 训练与初始化、probe、旧版两类构建、重复 hard-pseudo 导出、重复 bridge 预测导出、旧 inspect 实验、废弃 global4/student2 配置和 tracked pyc。当前 NWPU 构建器替代 `额外的bridge不同源数据集分离.py`；原工作区的同名打开文件保持原样。旧拆分/检查脚本中依赖 bridge_small/global4 的入口移除，保留当前 total_bridge 划分检查和两类数据检查。完整删除清单在 [验收 JSON](acceptance_20260918.json)。当前 tracked 源树没有主线依赖的独立 incremental/slicing/EWC/replay trainer，因此没有添加替代实现。

v2 的现有 `--generate-ship-pseudo` 功能继续使用冻结教师和真实文本嵌入。另实际检查三张原 train RGB/SAR/IR 图片，导出 6 个硬框、0 个空标签，结果在 `runs/migration/pseudo_check/result.json`。它不参与本次专家训练，也不被称为 soft/logit KD。门控、校准和选择性响应蒸馏均未实现。

## 最小修复及行为变化

- 原 Windows 代码根硬编码改为文件位置解析，所有运行入口绑定自己的框架并断言来源。数据/权重仍可读原位置。
- 专家包装器保留相对数据路径转运行 YAML，移除吞异常 fallback；runtime YAML 按 run 隔离，拒绝覆盖已有 run。外部训练数据镜像图片引用、复制标签，使 cache 写入本 worktree；实测原 train=2481、val=532，划分不变。
- 评估原先要求全量固定 694 / 798 GT 的检查改为标签合法、非空、正确类映射检查，因而可接受真实固定子集；GT 匹配、AP 公式、NMS 和默认阈值未改。
- 统一原三类 prompt 初始化，原模型 task=segment、YOLOESegModel / YOLOESegment、nc=3、实际分数通道=3。旧基础入口曾用四类名称描述三通道融合 head；现在明确只初始化原三类真实 embedding。原 checkpoint 的 `pe` 属性为空并非丢失模型；运行时 `[1,3,512]` text embedding 已验证。ship 推理输出与源快照逐项相同。
- 基础评估添加 CLI 和精确 JSON，保留旧 CPU/640/conf=.001、阶梯积分口径；其数据 YAML 相对项现在正确结合 `path`。推荐接口不再重复维护一套同功能评估代码。
- 外源专家的普通 Detect 官方验证复制标签并隔离 cache；保留官方 AP 与自定义固定阈值两种口径。held-out 增加显式 batch，固定 NMS/GT IoU 的含义，拒绝覆盖旧结果。
- 首次试跑 AMP 自检遇到原环境/框架组合的额外 YOLO26n 检查加载问题，框架自行警告并跳过；首次训练仍通过但只完成 1 次有效 step。最终采用 FP32 重跑完整验收，6 次 step 全部有效。没有新增教师/学生，没有升级框架；正式训练默认 AMP 不变，README 本机命令显式关闭。

## 实际结果

删除历史代码之前，先在 detached 源上运行参照；`runs/migration/source_reference.log`、`source_exit.txt=0`、`before/` 保留其输出。固定源适配器与 clean 适配器使用相同 6 张真实图片、原基础权重和历史完整专家、1024 / batch2 / CUDA0 / 候选 .001 / NMS .7 / report .25 / class-aware / max_det300 / PYTHONHASHSEED42。

| 对照内容 | 清理前 | 清理后 | 差异 |
|---|---:|---:|---:|
| ship 候选 | 193 | 193 | 0 |
| bridge 候选 | 77 | 77 | 0 |
| 融合候选 | 267 | 267 | 0 |
| 框坐标最大绝对差 | — | — | 0 px |
| 分数最大绝对差 | — | — | 0 |
| 融合 combined AP50 | 0.87613636 | 0.87613636 | 0 |
| 融合 report TP / FP / FN | 13 / 7 / 0 | 13 / 7 / 0 | 0 |

图像集合、GT、类别映射、逐图候选顺序/数量、框、分数、TP/FP 数组及全部同口径指标都通过比较。容差预设坐标 .001 px、分数 1e-6；本次实际完全相同。小子集数字不用于复现历史全量 AP。

最终真实短训练：`runs/clean_smoke/acceptance_20260918_final/`，12 train / 6 val / 6 held-out eval；train、val、ship test、bridge test 都覆盖 RGB/SAR/IR。1 epoch、1024、batch2、workers0、close_mosaic0、nbs2、FP32。6 batches、6 optimizer steps；`model.0.conv.weight` 在 optimizer 中、有非零有限梯度，最大更新 **0.0002858788**。平均 box / cls / dfl loss 为 **2.1754938 / 6.0488896 / 2.3385470**。

保存并新进程重载 `training/expert/weights/last.pt`（亦保存有效 best.pt）。该新权重产生 **1800** 个 .001 候选，基础模型仍产生 193 个；新专家在 .25 报告阈值下没有候选，属于一轮训练尚不足，低阈值推理/融合路径已实际非空验证。新融合 combined AP50 **0.46295102**，TP/FP/FN **10/4/3**。不能把它与完整专家的性能差别解释成清理回归。

基础 checkpoint SHA256 前后相同，加载后冻结模型 state_dict 哈希在推理前后相同。模块全部来自新 worktree；真实训练、保存、重载、评估各子进程退出 0。详细 loss、命令 argv、环境、源样本哈希、新权重哈希、单模型和融合预测、GT、TP/FP、JSON/CSV 均保存在该 run，并压缩摘录于版本控制中的 `acceptance_20260918.json`。

额外入口执行证据在 `runs/migration/entry_checks/commands.json`：基础 CLI 在 3 张测试图上按旧 CPU/640 口径通过；融合 CLI 显式传入本次新权重，6 张测试图通过；统一专家 CLI 用 6 张 val 测试隔离官方验证与自定义评估；held-out CLI 在 3 张测试图上跑完六档阈值。入口名称里的 external 不表示本次做了完整 NWPU 泛化实验。三个评估 CLI 的完整历史 shell 命令仍未可靠找全；README 给的是当前已核对参数的命令。

## 限制与最终完整性

未执行 100/150 epoch 重训或全量评估，未验证 Linux/4090、AMP 数值等价或新的研究类别。原环境 `pip check` 返回 1：OpenCV 4.13 声明 NumPy>=2，但当前为1.26.4；本次实际运行通过，包环境未改。这是既存环境声明冲突，不是清理回归。

原分支仍为 `bridge_clean_official@736cb86`，工作区 Git 状态干净。原位置 25,844 文件的大小/修改时间清单完全一致（排除 `.git`、`.venv`、`__pycache__`；关键资产额外校验 SHA256），没有修改源数据、权重或历史结果。源/新工作区与生成资产均保留。数据、权重、cache、大日志不进入清理提交；没有合并或推送。
