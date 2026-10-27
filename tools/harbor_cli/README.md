# Harbor显式参数训练入口

入口：`tools/train_harbor_expert.py`。Windows单GPU；本机Python为
`C:\Users\DOCTOR\anaconda3\envs\y8distill\python.exe`。参数解析与dry-run只用标准库。
新入口从原始三类YOLOE B生成单类E0，短检后重新设置正式随机状态，再训练、验证、选优、保存和导出。
没有门控、融合、蒸馏或最终test。原B、派生数据与旧实验保持只读。

```powershell
Set-Location 'C:\Users\DOCTOR\Documents\GitHub\YOLOv8_harbor_expert_poc'
$env:WANDB_MODE = 'disabled'

& 'C:\Users\DOCTOR\anaconda3\envs\y8distill\python.exe' -B -u -X utf8 `
  '.\tools\train_harbor_expert.py' `
  --base_weights 'C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\best.pt' `
  --data_root_dir 'D:\RSdataset\_derived\harbor_poc_20261007_124509' `
  --run_root_dir 'D:\RSdataset\experiments' `
  --run_id_note harbor_rgb_sar `
  --device 0 `
  --epochs 150 --patience 30 --seed 43 `
  --image_size 640 `
  --batch_size 4 --grad_accumulation_steps 4 --val_batch_size 4 `
  --num_workers 4 --val_num_workers 0 `
  --num_threads 8 --num_interop_threads 1 `
  --prefetch_factor 2 --persistent_workers True `
  --pin_memory True --cache_images shared `
  --optimizer AdamW --learning_rate 1e-4 --weight_decay 5e-4 `
  --warmup_epochs 3 --lrf 0.1 --gradient_clip 10 `
  --rgb_repeat 4 --sar_repeat 1 `
  --horizontal_flip 0.5 --vertical_flip 0.5 `
  --box_loss_gain 7.5 --cls_loss_gain 0.5 --dfl_loss_gain 1.5 `
  --amp False
```

追加`--dry_run`只打印最终配置和调用命令，不导入torch、不创建实验、不初始化模型或训练。
每次从头运行自动创建`D:\RSdataset\experiments\harbor_rgb_sar_<时间戳>`；用`--experiment_dir`可指定精确的新路径。
`--base_weights`当前仍需通过原B SHA256和已验证YOLOE三类结构检查；不是任意模型/类别的通用转换器。
数据目录须为已固定的train/val JSON清单版本，不能将原DIOR XML目录当作这个参数。

| 参数 | 实际控制 |
|---|---|
| batch_size | 训练DataLoader每个微批的图像数，确实影响GPU批量 |
| grad_accumulation_steps | 每次更新累积微批数；末尾按实际图数归一化 |
| num_workers | 训练DataLoader的CPU进程数，不是GPU数 |
| num_threads | 主进程PyTorch CPU intra-op线程数，覆盖旧setup的固定4 |
| num_interop_threads | 主进程PyTorch inter-op线程数 |
| prefetch_factor | 每个worker提前取的批次数；workers=0时不传给DataLoader |
| persistent_workers | workers>0时跨epoch复用；workers=0时实际为false |
| cache_images shared | uint8共享内存缓存，Windows spawn不为每个worker复制整套图像 |
| cache_images none | 按需读图，可由workers并行读取；增加磁盘/PIL工作 |
| cache_images ram | 主进程私有缓存，只允许workers=val_workers=0 |
| val_batch_size / val_num_workers | 独立控制验证加载，不由训练batch隐式覆盖 |
| device | 导入torch前设置CUDA_VISIBLE_DEVICES，单张物理GPU |

主进程默认CPU线程4、workers0；上面的4 workers/8 threads是显式示例，不是已完成的吞吐率最优搜索。
worker内部torch线程限制为1，减少CPU进程与线程嵌套的争抢。batch4在旧链路真实训练过；新CLI本轮只做CPU检查，未做GPU训练或速度基准。
增加workers或线程不保证更快；新history.csv/json分开记录training_seconds、data_loader_wait_seconds、validation_seconds，后续应以这些实测判断。

训练范围仍为最后P5融合块＋分类/框头；DFL和全部BN统计固定。FP32是当前支持的数值路径，amp=True会明确拒绝。
翻转概率、重复采样、损失系数、学习率、epoch预算等参数真实进入相应计算；EMA使用既定.9999/tau2000。

为保证持久worker及恢复时的增强可追溯，采用`seed+epoch+呈现位置`确定翻转的`harbor_stateless_flip_v1`。
RGB重复呈现仍能得到各自翻转，worker数/worker重建不改变该序列。分布仍是指定的水平/垂直翻转概率，
但与旧10轮实验的全局random调用轨迹不同，不能把两者称作只改变线程数的严格消融。

日志在新实验`logs/`，每轮history和候选在`runs/<run_id_note>/`；真正完成/早停后导出`inference/harbor_expert.pt`。
resume与status可使用相同入口：

```powershell
# 将路径设为这次启动时打印的新CLI实验根；RUN_COMMANDS.md也保存了精确命令。
$Experiment = Read-Host '粘贴本次Experiment绝对路径'
& 'C:\Users\DOCTOR\anaconda3\envs\y8distill\python.exe' -B -u -X utf8 `
  '.\tools\train_harbor_expert.py' --experiment_dir $Experiment --resume

& 'C:\Users\DOCTOR\anaconda3\envs\y8distill\python.exe' -B -X utf8 `
  '.\tools\train_harbor_expert.py' --experiment_dir $Experiment --status --tail 12
```

恢复要求保存的科学配置与源代码一致；CLI不能在resume时静默换batch、workers或预算。
`--pause_after_epoch`只控制完整epoch边界暂停，不改变总预算；完整续训不传此参数。
旧10轮实验仍用它原有的Continue_Training.ps1，原16个执行文件哈希保持不变；新CLI不会冒充旧run的无变化恢复。
本入口未实现DDP，不应使用torchrun启动多个进程。单GPU直接python即可，线程和数据worker都在本进程树内配置。
