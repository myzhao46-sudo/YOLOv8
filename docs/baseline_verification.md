# 正式基线核对与全流程验收

本记录状态：**用户要求停止训练；正式新专家全流程未完成，转交服务器**。以 `state.json` 的退出状态和本次文件为准。

工作区：`C:/Users/DOCTOR/Documents/GitHub/YOLOv8_distill_pipeline_clean`；分支：`distill_pipeline_clean`。起点 `7b1216ded9c64e342b16cd22df953eb7f38c90c2`，开始时无工作区改动。固定源 `bridge_clean_official@736cb86c865fa490e9ae77e3189b8098825897fd`。本次执行代码提交 `8fdf2897a81b2fecffec8a1a9074a05ffa657099`；没有 reset、amend、合并或推送。
本次没有找到适用的 AGENTS.md。已读取本分支 README 与 migration_20260918.md；上次 smoke 只作为先前基础检查。本记录全量结果来自新的完整数据运行。
结果目录：`C:/Users/DOCTOR/Documents/GitHub/YOLOv8_distill_pipeline_clean/runs/baseline_verification/20260918_formal_v2`；旧行为全量对照：`C:/Users/DOCTOR/Documents/GitHub/YOLOv8_distill_pipeline_clean/runs/baseline_verification/20260918_legacy_reference`。首次严格检查失败记录位于相邻 `20260918_formal`，没有删除失败证据。

## 功能与设计对照

| 设计要求 | 实际入口/调用证据 | 是否本次运行 | 状态 | 缺失或偏差 |
|---|---|---|---|---|
| 数据读取、准备、切分 | baseline.asset_inventory → data.isolate_training_data；原 YAML → train 包装器运行时 YAML | 全量读取、隔离已执行；未重切分 | 已实现且本次执行 | train/val/test 使用固化清单；原始汇集/切分工具本次未重跑 |
| 冻结 YOLOE 推理 | evaluate.main → fusion.prepare_ship_expert/infer_ship → teacher.set_original_classes/get_raw_prediction | 832 张全量推理，基础参数前后哈希相同 | 已实现且本次执行 | YOLOESegModel/YOLOESegment，task=segment；names=ship/harbor/tank，nc=3，PE=[1,3,512]，真实分数通道=3；只消费 ship |
| bridge 专家训练 | train_bridge_expert_yolo.main → YOLO.train → DetectionTrainer → DetectionModel.init_criterion → v8DetectionLoss；BaseTrainer.backward/optimizer_step | 全量正式训练按用户要求停止 | 已实现且本次执行 | 监督 box/cls/dfl；YOLOv8m 初始化，单类 0 bridge；真实保存和重载是否完成以末表为准 |
| hard pseudo | build_bridge_ship_distill_2cls_v2.generate_ship_pseudo_labels，main 的 --generate-ship-pseudo 分支 | 否 | 已实现但非默认流程 | 可选数据准备/历史对照：导出教师硬框，v2 同时负责当前测试集映射；正式专家使用真实 bridge GT，不训练失败的二类 student |
| soft/logit KD | 上述 v8DetectionLoss 只有监督 box/cls/dfl，没有教师 logits 或 KD 加权项 | 否 | 未实现 | 没有进入反向传播/优化器的 KD 项 |
| box KD / 保留类辅助响应 | 专家训练调用链无保留类 teacher 输入/辅助输出消费 | 否 | 未实现 | 框架中通用 YOLOE 模块不代表当前专家具备辅助响应训练 |
| 预测融合与 NMS | fusion.infer_bridge 将 local0→global1；fuse_predictions → nms_predictions | class-aware 主线及同候选 class-agnostic 对照已执行 | 已实现且本次执行 | 拼接 box/score/class 后 NMS=.7；不拼 checkpoint |
| 分数校准、规则/学习式门控 | 正式预测链无校准函数、可靠性模型或门控损失 | 否 | 未实现 | 候选阈值/类别过滤/NMS 不是可靠性门控 |
| 单模型、融合、外部评估 | evaluate.main → fusion.evaluate_subset/match_class；external.main → Detect model.val + match_fixed_conf | 历史完整权重全量完成；新权重待训练结束 | 已实现且本次执行 | YOLOE 不调用 model.val；Detect 内部/统一验证保留官方实现 |
| 少样本目标域适配 | 无适配训练入口；NWPU 仅外部测试 | 否 | 未实现 | 外测不等于适配；未使用外测选择阈值、模型、seed |
| 完整研究设计说明 | README：冻结基础+增强专家+选择性保留响应蒸馏+非对称可靠性门控 | 未作为完整方法执行 | 仅保留接口或说明 | 这里只保留设计说明，没有为缺失模块添加空类或假 loss |

## 数据、环境与协议

完整原 bridge：3545 图/5152 框；RGB 1299/2589、SAR 1582/1851、IR 664/712。train 2481/3600、val 532/754、test 532/798；固化 70/15/15、seed42，不重新切分。ship test 300/694，combined 为无重复并集 832/1492；NWPU 67/124，只有 RGB。
所有本次选用的全量图像字节和标签重新 SHA256，检查非空合法标签、原 manifest、目录与清单一致、train/val/test 内容不重叠；bridge test 在 v2 中仅 class0→1，坐标逐项相同。NWPU 与 bridge 各划分无内容重复。清单和标签内容见 input_manifest.json。跨类别目标标注不保证完整，因此以下“FP”是相对现有标注的未匹配预测，未经人工确认，不等同真实误报。
环境：C:\Users\DOCTOR\anaconda3\envs\y8distill\python.exe；Python 3.10.20；PyTorch 2.3.1+cu121；CUDA 12.1；NVIDIA GeForce RTX 4070 Laptop GPU；Ultralytics 8.4.32；本次 CUDA/FP32。具体依赖、sys.path、direct_url、模块 __file__ 见 state.json。未升级环境；OpenCV/NumPy 包声明冲突与 AMP 自检问题沿用先前记录，未声称 AMP 通过。

融合协议：imgsz1024，batch2，原 PIL RGB/letterbox；基础 raw YOLOESegment box 分支保留三类，再选 ship。专家修复为同一原像素方向的 PIL RGB→BGR 数组，避免路径加载器 EXIF 旋转。候选下限 .001，专家各自 NMS IoU=.7；融合 NMS=.7，max_det300；报告 P/R 在 score≥.25、GT IoU=.5；AP 用 .5:.05:.95、score 降序贪心 GT 匹配与原 101 点插值积分，不使用 .25 截断 AP 候选。所有分模态从本次全量候选重放，不重新选择阈值。
专家统一验证：imgsz1024、batch8、workers0、FP32、官方 rect 验证/官方匹配与 AP、候选 .001、NMS=.7、max_det300；官方 P/R 是验证器最佳 F1 工作点。另报 predict(conf=.25) 的自定义 GT IoU=.5 P/R/TP/FP/FN，二者不可混为同一工作点。外部官方验证允许 model.val，YOLOE 基础自定义 box-only 评估不替换。
缓存阶段：每个专家 NMS 之后、跨专家融合 NMS 之前，与原融合实现等价。class-agnostic 只重放该候选缓存；新专家融合复用本次基础缓存，校验基础权重 SHA、数据/标签指纹、预处理/候选/NMS/精度、推理代码 SHA、预测文件 SHA，不复用旧专家候选。

## 历史完整权重对照

差值方向为“本次−历史”，AP 使用 0—1。严格同条件比较预设框绝对容差 .001px、分数 1e-6、指标 1e-8；历史四位小数容差 .00005，未事后放宽。下面旧行为列来自本次完整源快照/清理分支对照，最终列来自 EXIF 修复后的代码。

| 模型/数据/协议 | 历史值及来源 | 本次旧行为 / 最终代码 | 差值（最终−历史） | 严格可比条件 | 原因/未解决事项 |
|---|---|---|---|---|---|
| ship/test_ship，历史融合后 class0 AP50 | 0.9215801790；历史 late_fusion_summary.json | 0.9215801790 / 0.9215801790 | 0.0000000000 | 是（旧行为完整复现；最终修复影响另列） | 一致 |
| bridge/test_bridge，历史融合后 class1 AP50 | 0.7978427218；历史 late_fusion_summary.json | 0.7978427218 / 0.7978427218 | 0.0000000000 | 是（旧行为完整复现；最终修复影响另列） | 一致 |
| combined ship AP50 | 0.7841940187；历史 late_fusion_summary.json | 0.7841940187 / 0.7841940187 | 0.0000000000 | 是（旧行为完整复现；最终修复影响另列） | 一致 |
| combined bridge AP50 | 0.7793928665；历史 late_fusion_summary.json | 0.7793928665 / 0.7794487419 | 0.0000558754 | 是（旧行为完整复现；最终修复影响另列） | EXIF 修复仅两张 ship 图的 bridge 预测；其他图相同 |
| combined class-aware ap50 | 0.7817934426；历史 late_fusion_summary.json | 0.7817934426 / 0.7818213803 | 0.0000279377 | 是（旧行为完整复现；最终修复影响另列） | EXIF 修复仅两张 ship 图的 bridge 预测；其他图相同 |
| combined class-aware map50_95 | 0.5090238692；历史 late_fusion_summary.json | 0.5090238692 / 0.5090334473 | 0.0000095780 | 是（旧行为完整复现；最终修复影响另列） | EXIF 修复仅两张 ship 图的 bridge 预测；其他图相同 |
| combined class-aware precision | 0.5157981124；历史 late_fusion_summary.json | 0.5157981124 / 0.5160098522 | 0.0002117398 | 是（旧行为完整复现；最终修复影响另列） | EXIF 修复仅两张 ship 图的 bridge 预测；其他图相同 |
| combined class-aware recall | 0.8424932976；历史 late_fusion_summary.json | 0.8424932976 / 0.8424932976 | 0.0000000000 | 是（旧行为完整复现；最终修复影响另列） | 一致 |
| combined class-agnostic AP50 | 0.7819718943；历史 classagnostic JSON | 0.7819718943 / 0.7819995362 | 0.0000276419 | 是（旧行为完整复现；最终修复影响另列） | EXIF 修复仅两张 ship 图的 bridge 预测；其他图相同 |
| total_bridge_test_same_source 官方 ap50 | 0.8122298843；历史统一 metrics.json | 0.8122298843 / 0.8122298843 | 0.0000000000 | 是；完整配置已恢复、数值一致 | 一致 |
| total_bridge_test_same_source 官方 map50_95 | 0.5364099932；历史统一 metrics.json | 0.5364099932 / 0.5364099932 | 0.0000000000 | 是；完整配置已恢复、数值一致 | 一致 |
| total_bridge_test_same_source 官方 precision | 0.8518297247；历史统一 metrics.json | 0.8518297247 / 0.8518297247 | 0.0000000000 | 是；完整配置已恢复、数值一致 | 一致 |
| total_bridge_test_same_source 官方 recall | 0.7531328321；历史统一 metrics.json | 0.7531328321 / 0.7531328321 | 0.0000000000 | 是；完整配置已恢复、数值一致 | 一致 |
| nwpu_external_cls1_as_bridge 官方 ap50 | 0.2957840684；历史统一 metrics.json | 0.2957840684 / 0.2957840684 | 0.0000000000 | 是；完整配置已恢复、数值一致 | 一致 |
| nwpu_external_cls1_as_bridge 官方 map50_95 | 0.0653218204；历史统一 metrics.json | 0.0653218204 / 0.0653218204 | 0.0000000000 | 是；完整配置已恢复、数值一致 | 一致 |
| nwpu_external_cls1_as_bridge 官方 precision | 0.4698437784；历史统一 metrics.json | 0.4698437784 / 0.4698437784 | 0.0000000000 | 是；完整配置已恢复、数值一致 | 一致 |
| nwpu_external_cls1_as_bridge 官方 recall | 0.3287817584；历史统一 metrics.json | 0.3287817584 / 0.3287817584 | 0.0000000000 | 是；完整配置已恢复、数值一致 | 一致 |
| ship/rgb AP50 | 0.951（用户历史近似线索） | 0.9575336571 | 不计算提升/下降 | 否 | 未恢复该组近似值完整配置；不能冒充本次统一口径基准 |
| ship/sar AP50 | 0.897（用户历史近似线索） | 0.8887149426 | 不计算提升/下降 | 否 | 未恢复该组近似值完整配置；不能冒充本次统一口径基准 |
| ship/ir AP50 | 0.916（用户历史近似线索） | 0.9135279125 | 不计算提升/下降 | 否 | 未恢复该组近似值完整配置；不能冒充本次统一口径基准 |
| bridge/rgb AP50 | 0.659（历史按模态 JSON） | 0.6589737611 | -0.0000262389 | 是；四位舍入范围内 | 相同融合后单类口径 |
| bridge/sar AP50 | 0.9504（历史按模态 JSON） | 0.9503806029 | -0.0000193971 | 是；四位舍入范围内 | 相同融合后单类口径 |
| bridge/ir AP50 | 0.8928（历史按模态 JSON） | 0.8927622042 | -0.0000377958 | 是；四位舍入范围内 | 相同融合后单类口径 |

0.8122 与 0.7978 均已恢复并分别复现。后者来源是 class-aware 融合后 test_bridge 的 class1，额外融合 NMS 会改变少量候选；单专家仅其自身 NMS 的本次 AP 是下表中的独立结果，不能把二次 NMS 或官方验证的差异笼统叫“conf=.25 差异”。更早 0.7674653327 属于旧 conf=.05 截断候选、阶梯积分脚本；该脚本 conf=.25 AP=0.7439724231，也不等于 0.7978。没有用旧值覆盖统一基准。
旧行为跨场景统计精确复现：bridge→ship 79 个/64 张；ship→bridge 770 个/312 张，RGB382/SAR263/IR125。class-agnostic 仅减少4个，combined FP1180→1176。最终 EXIF 修复后 bridge→ship 78 个，combined FP1179；agnostic 为1175，ship→bridge不变。
首次全量严格坐标检查发现源流程既有缺陷：ship_test_ir_1_4533.jpg（EXIF3）、ship_test_ir_1_4558.jpg（EXIF6）被专家自动旋转，而基础模型和 GT 使用原像素。历史预测文件也含越界框。先独立记录 832 张旧行为完全相同，再统一专家解码；最终所有坐标检查通过，且只这两张图的专家/融合候选改变，ship 候选全部不变。未改权重、标签、阈值、匹配或 AP 算法。无缺失依赖文件需要从源分支补回。详见 historical_full/exif_fix_audit.json。

## 本次新专家结果

正式初始化为原 yolov8m.pt；不是 smoke 权重，不 resume。原 train2481/val532、imgsz1024、epochs150、seed42、patience40、close_mosaic20、nbs64、optimizer=auto。为本机8GB显存/16GB内存使用 batch2/workers0/FP32，历史为 batch16/workers8/AMP。有效 batch 的梯度累积仍由原 nbs64 机制决定；不宣称两次训练条件逐位等价。只由原 val 选 best，test/NWPU 没有参与训练或模型选择。

中断与恢复：原训练在第2 epoch验证阶段随上次会话中断，没有正常退出状态；恢复自本次新训练第1 epoch的 last.pt（保留 optimizer/EMA，已归档）。恢复代码提交：`0f3273f562d5fae86e89301bde688d96c928261c`，见 resume_archive/ 和 train_resume_1.log。不是 resume 历史完整专家。框架原生恢复使用 checkpoint 中的 EMA/优化器，随机增强序列未逐位恢复，故不承诺与不间断训练逐位相同；恢复原验证早停历史，150 epoch 总计划不变。

最后完整检查点：epoch 1/150，best epoch 1，已保存 optimizer steps 301；**尚未完成，不提供新专家最终指标**。恢复后的第2 epoch在用户要求停止时仍未保存新检查点，未把中断部分计为完成。

| 训练配置与权重/模式 | 数据集/模态 | AP50 / mAP50-95 / P / R / TP / FP / FN | 差异（新−本次历史专家） |
|---|---|---|---|
| 历史完整权重/单专家 | test_ship | 0.9215471245 / 0.6020606657 / 0.8242894057 / 0.9193083573 / 638 / 136 / 56 | 0.0000000000 个百分点 |
| 新专家组合/独立专家 | test_ship | 尚未执行 | — |
| 历史完整权重/单专家 | test_bridge | 0.7976909817 / 0.5114121037 / 0.7604422604 / 0.7756892231 / 619 / 195 / 179 | 0.0000000000 个百分点 |
| 新专家组合/独立专家 | test_bridge | 尚未执行 | — |
| 新 best + 冻结 ship / class-aware | test_ship class=0 | 尚未执行 | — |
| 新 best + 冻结 ship / class-aware | test_bridge class=1 | 尚未执行 | — |
| 新 best + 冻结 ship / class-aware | combined class=overall | 尚未执行 | — |
| 新 best + 冻结 ship / class-aware | combined class=0 | 尚未执行 | — |
| 新 best + 冻结 ship / class-aware | combined class=1 | 尚未执行 | — |
| 新 best + 冻结 ship / class-aware | test_ship/rgb class=0 | 尚未执行 | — |
| 新 best + 冻结 ship / class-aware | test_ship/sar class=0 | 尚未执行 | — |
| 新 best + 冻结 ship / class-aware | test_ship/ir class=0 | 尚未执行 | — |
| 新 best + 冻结 ship / class-aware | test_bridge/rgb class=1 | 尚未执行 | — |
| 新 best + 冻结 ship / class-aware | test_bridge/sar class=1 | 尚未执行 | — |
| 新 best + 冻结 ship / class-aware | test_bridge/ir class=1 | 尚未执行 | — |
| 新 best | new_bridge_official | 尚未执行 | — |
| 新 best | new_nwpu | 尚未执行 | — |
| 新 best + 冻结 ship / 跨场景未匹配 | test_ship class=1 | 尚未执行 | — |
| 新 best + 冻结 ship / 跨场景未匹配 | test_ship/rgb class=1 | 尚未执行 | — |
| 新 best + 冻结 ship / 跨场景未匹配 | test_ship/sar class=1 | 尚未执行 | — |
| 新 best + 冻结 ship / 跨场景未匹配 | test_ship/ir class=1 | 尚未执行 | — |
| 新 best + 冻结 ship / 跨场景未匹配 | test_bridge class=0 | 尚未执行 | — |
| 新 best + 冻结 ship / 跨场景未匹配 | test_bridge/rgb class=0 | 尚未执行 | — |
| 新 best + 冻结 ship / 跨场景未匹配 | test_bridge/sar class=0 | 尚未执行 | — |
| 新 best + 冻结 ship / 跨场景未匹配 | test_bridge/ir class=0 | 尚未执行 | — |

单模型与融合不能只对两个数据集 AP 取平均；combined 的两个类别都包含跨场景未匹配候选。分模态、class-agnostic、跨场景候选数和涉及图像数完整保存在 old/new 对应 replayed_metrics.json/csv。上述官方 AP 与固定 P/R 明确分开来源，不制造不存在的历史指标。

## 分支清洁性

| 代码导入来源/文件 | 业务入口与配置用途 | 必要外部资产 | 无关残留 | 本次修复 | 证据与结论 |
|---|---|---|---|---|---|
| 全部实际运行入口 | 本 worktree ultralytics/ultralytics 和 tools；打印并断言 __file__ | 原目录只读数据/权重 | 本次检查未发现运行链调用旧实验 | 禁用进程内旧 editable finder；不改共享环境 | state.json/modules、子进程日志；源对照进程只导入固定源 worktree |
| tools/train/train_bridge_expert_yolo.py | 主流程监督训练、runtime YAML、更新观察 | yolov8m.pt / trainval GT | 无 KD/student 训练 | 新增 epoch 进度观察；原训练计算未变 | 训练 args.yaml、progress、update_audit、train_summary |
| tools/eval/eval_late_fusion_ship_bridge.py | 主流程双模型推理/映射/融合/自定义指标 | 两个只读 checkpoint、测试集 | 无可靠性门控 | 统一 raw EXIF 像素方向 | 832 全量参照及仅两张图变化审计 |
| tools/eval/eval_teacher_ship_external.py | 必要 YOLOE raw box/letterbox/文本嵌入支持 | base 和 mobileclip_blt.ts | 不是多余重复实现 | 无额外修复 | YOLOESegModel/YOLOESegment 保留，PE 和通道检查 |
| tools/eval/eval_yoloeseg_box_only.py | 唯一独立旧口径基础 CLI，复用 teacher helper | ship YAML | 非本次历史融合 AP 的默认 CLI | 无 | CPU640/阶梯积分默认值未改；本次基线调用相同底层推理+融合指标 |
| tools/eval/eval_bridge_expert_external_yolov8.py | 推荐统一单类 held-out/external 评估 | 同源/NWPU YAML | 无 | 显式workers0/FP32，隔离cache | 两种全量数据正式执行 |
| tools/eval/eval_bridge_expert_total_bridge_test.py | 可选历史 conf-list/阶梯 AP 对照 | 同源 held-out | 明确非默认 | 无 | 本次只审计旧 JSON，不重扫阈值 |
| tools/run_baseline_verification.py；tools/clean_pipeline/baseline.py | 正式调度/全量资产校验/原指标缓存重放 | 外部资产与固定旧行为证据 | replay函数指候选重放，非回放训练 | 本次新增最小调度与证据收集 | state.json 顺序子进程、非零失败传播 |
| tools/run_clean_smoke.py；clean_pipeline/data.py、evaluate.py、runtime.py | 可选小回归；必要数据隔离、推理验收、来源断言 | 已保存 smoke fixtures 可选 | 无 | 缓存验证和来源隔离增强 | 本次未把 smoke 当正式结果 |
| tools/datasets/make_total_bridge_dataset.py；split_total_bridge_train_val_test.py | 可选原数据汇集/固化划分支持 | 原多模态数据 | 无 | 无 | 本次沿用已有清单，不执行重新切分 |
| tools/datasets/build_bridge_ship_distill_2cls_v2.py | 可选 v2 数据与 hard pseudo 导出 | 既有 ship/bridge GT 和教师权重 | 不参与本次专家训练 | 无 | 保留当前评估集构造来源；无失败student训练入口 |
| tools/datasets/build_external_bridge_nwpu_by_label.py | 可选 NWPU 筛选/重映射工具 | NWPU 原标签 source1→target0 | 无 | 无 | 本次直接读取已构建资产，不覆盖GT |
| tools/datasets/split_ship_small.py | 可选原 ship 划分工具 | ship 原资产 | 无 | 无 | 未执行/未声称本次新划分 |
| tools/datasets/外部数据集制作make_ship_extratest_no_overlap.py | 可选 ship 去重外测准备 | 原 ship 数据、去重参照 | 无 | 无 | 未执行；与运行时外部只读资产区别明确 |
| tools/datasets/make_bridge_rgb_extratest_no_overlap.py；make_bridge_extratest_sar_ir_no_overlap.py | 可选 DIOR/RGB 与 SAR/IR 外测准备 | E:/ 等原数据源，需另备 | 无算法依赖 | 更正历史 Student 字样为融合类别 | 本次没有把未运行数据准备工具宣称通过 |
| tools/check/check_total_bridge_split.py；check_2cls_dataset_splits.py；tools/vis/preview_yolo_dataset_boxes.py | 可选数据检查/人工预览支持 | 用户指定数据 | 无 | 无 | 此次全量检查由资产清单完成，未逐个运行可选 CLI |
| configs/datasets/total_bridge_trainval.yaml；total_bridge_test.yaml；bridge_external_nwpu.yaml | 正式训练、同源测试、外测的唯一推荐配置 | 原目录只读数据 | 无 | 未改 split | 完整执行；缓存和日志只写新 worktree |
| configs/datasets/ship_val.yaml；ship_external.yaml；total_bridge.yaml | 可选旧基础评估/汇集配置 | 原 ship/完整 bridge 资产 | 无 | 无 | total_bridge.yaml 不用于正式训练：train/val 不分离；正式只用 trainval 配置 |
| 本地 ultralytics 完整框架与通用示例 | 必要模型类/注册/序列化、Detect trainer 和 loss | 第三方 torch、clip 等 | 无旧自定义 global4/student/incremental/EWC/replay/slicing 入口或引用 | 不盲删 seg；保留 Linux checkpoint 路径兼容/本地模块别名 | 加载类均来自新树；通用分割/跟踪/示例不等于历史业务依赖 |
| README、迁移记录、environment/acceptance JSON、.vscode/settings.json、Git 配置 | 主说明/先前证据/普通编辑器设置 | 无 | 历史记录明确时间和检查范围 | 更新 README 正式验收入口与结论 | 没有旧实验 legacy 目录；原工作区 Git 状态与25844文件元数据无变化 |

共享 Python 的 direct_url 仍登记原 editable 路径，但本次先禁用该项目的进程内 finder/path hook，再固定本地导入，所有已加载项目模块均通过来源断言。该环境登记不再是本次运行代码来源，也未修改用户共享安装。图片使用校验后的引用/硬链接、标签复制；缓存、runtime YAML、新权重与日志在独立 runs 下。原始 best.pt 及历史专家/初始化/文本权重 SHA256 见 state.json，训练后再次校验。

## 分项结论

- **通过**：当前主流程的全量既有权重运行不导入原工作区代码；审计范围内无旧实验业务依赖。可选准备工具只检查源码，不承诺其历史外部输入资产仍齐全。
- **未通过（未完成，用户要求停止并转服务器）**：正式训练尚未正常结束，新专家全量评估未执行，不能用已完成第一部分冒充新专家全流程通过。
- **通过（限定）**：7b1216d 清理前后同条件832张全量候选/框/分数/匹配/指标完全一致。最终代码另有明确 EXIF 修复，改变两张图的专家预测；不能声称最终修复版逐预测与旧代码完全相同。
- **部分通过**：有完整 JSON/配置的统一同源、NWPU、旧融合总体/按类与桥梁分模态、跨场景候选和 agnostic 数字已复现；ship 模态约0.951/0.897/0.916仍缺协议，不作严格比较。
- **基线层通过；完整研究设计未通过（未实现）**：本次为冻结基础 + bridge监督专家 + late fusion/NMS。soft/logit KD、box KD、选择性保留类响应、分数校准、非对称可靠性门控和目标域适配没有实现；最终增强类别研究尚未重新确定。

## 本机复跑命令

下面正式调度命令按本次实际执行入口生成，省略 --out 会创建新时间戳目录；没有覆盖历史输出。它会先全量评估再最多150 epoch训练，并等待完成后重载新 best 做全量评估。只有本报告已完成部分才可视为已执行通过。

```powershell
Set-Location C:/Users/DOCTOR/Documents/GitHub/YOLOv8_distill_pipeline_clean
$py = 'C:/Users/DOCTOR/anaconda3/envs/y8distill/python.exe'
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONUTF8='1'; $env:PYTHONHASHSEED='42'
& $py -B tools/run_baseline_verification.py --legacy-reference runs/baseline_verification/20260918_legacy_reference
```

本次执行过的精确父命令加 `--out runs/baseline_verification/20260918_formal_v2`。该目录已存在，不能重复使用。所有子命令的 argv、cwd、起止时间、PID、日志与退出状态见 state.json。报告生成与旧候选重放不改变预测/训练计算。

历史基础模型 box-only 与融合完整结果均由 evaluate.py 保存 ship/bridge/fusion 单独预测和指标；推荐使用正式调度避免误混 CPU640 旧 AP 与本次1024融合 AP。基础 CLI旧默认、历史阈值扫描和可选数据准备在本次只核对入口，未声称重新全量运行。
正式训练等效独立命令见 state.json train argv；输入仍是原 trainval YAML，包装器生成的 runtime YAML 隔离在 training/runtime/expert。统一专家与融合的全部实际 CLI 命令同样保存，未添加未知 flags。
输出清单：state.json、input_manifest.json、business_code_manifest.json、historical_metadata/、各步骤 .log；historical_full 与 new_full 中 GT/逐模型预测/匹配/JSON/CSV/cache元数据；training/expert 中 args.yaml/results.csv/progress/update_audit/train_summary/weights；各官方评估目录 metrics.json 与独立缓存。
不提交数据、权重、缓存或大日志；只提交代码与简短说明。本机资产不会随 Git 分支出现。后续推送命令（本次未执行）：`git push -u origin distill_pipeline_clean`。
