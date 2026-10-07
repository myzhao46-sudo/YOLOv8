# -*- coding: utf-8 -*-
import argparse,csv,datetime,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.harbor_poc.core import read,save,sha,WORK

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in r)+' |' for r in rows])
def main(root):
    man=read(root/'experiment_manifest.json');test=read(root/'final_test/comparison.json');sel=read(root/'final_selection.json');data=read(Path(man['derived_root'])/'data_summary.json');freeze=read(root/'freeze_manifest.json')
    configs={r:read(root/'runs'/r/'effective_config.json') for r in ['A_seed42','A_seed43']};runs={r:read(root/'runs'/r/'result.json') for r in configs}
    vals={'B':read(root/'evaluations/B_val/metrics.json'),'E0':read(root/'evaluations/E0_val/metrics.json'),**{r:v['metrics'] for r,v in runs.items()}}
    flat=[]
    def metric_table(stage,models):
        rows=[]
        for name,mm in models.items():
            for mod in ['rgb','sar']:
                m=mm[mod];row={'stage':stage,'model':name,'modality':mod,**m};flat.append(row)
                rows.append([name,mod,m['images'],m['gt'],*[f'{100*m[k]:.3f}' for k in ['ap50','ap50_95','precision','recall']],m['tp'],m['fp'],m['fn'],m['candidates']])
        return table(['模型','模态','图','框','AP50%','AP50–95%','P%','R%','TP','FP','FN','低阈值候选'],rows)
    valtable=metric_table('validation',vals);testtable=metric_table('test',{k:test[k] for k in ['B','E0','E']})
    delta=test['delta_percentage_points'];both=all(delta[m]['ap50_95']>0 for m in ['rgb','sar'])
    conclusion='固定测试集的两个模态均有harbor AP50–95收益。' if both else '固定测试集未证实两个模态同时提升；以下按实际结果报告，不因测试结果改模型或阈值。'
    replica_both=all(vals['A_seed43'][m]['ap50_95']>vals['B'][m]['ap50_95'] for m in ['rgb','sar'])
    seednote='第二seed在验证集也改善两种模态，支持本原型趋势；两次种子不是跨数据集/场景泛化的充分证明。' if replica_both else '第二seed未重复两模态同时改善，不能称稳定提升；主模型仍按预先约定使用seed42。'
    trainingtable=table(['run','实际轮数','best epoch','结束原因','训练小时','验证宏平均AP50–95%','best SHA256'],[[r,v['epochs'],v['best_epoch'],v['termination'],f"{v['training_seconds']/3600:.3f}",f"{v['best_fitness']*100:.3f}",v['best_sha256']] for r,v in runs.items()])
    dt=table(['模态/split','原图','派生图','原框','派生框','排除图','排除框','空图负样本'],[[k,v['raw_images'],v['images'],v['raw_harbor_boxes'],v['harbor_boxes'],v['excluded_images'],v['excluded_harbor_boxes'],v['negative_images']] for k,v in data.items()])
    diff=table(['模态','ΔAP50(百分点)','ΔAP50–95(百分点)','ΔP(百分点)','ΔR(百分点)','ΔTP','ΔFP','ΔFN'],[[m,*[f'{delta[m][k]:+.3f}' for k in ['ap50','ap50_95','precision','recall']],delta[m]['tp_change'],delta[m]['fp_change'],delta[m]['fn_change']] for m in ['rgb','sar']])
    cfg=configs['A_seed42'];failures=[str(p.relative_to(root)) for p in root.rglob('*failure*.json')]
    content=f'''# Harbor 强化专家原型实验报告

{conclusion} {seednote}

本报告来自真实训练、重载及固定测试。测试数据是 **harbor标注正样本子集**，没有经人工核验的空图负样本；所报FP/P仅覆盖这些港口场景内的预测，不能代表自然全场景误报率。RGB和SAR分别报告，红外没有专家增强验证，继续保留B原路径。没有实现门控或响应蒸馏。

## 最终测试对照（未参与选择）

{testtable}

{diff}

报告阈值固定0.25，候选下限0.001；NMS IoU0.7，报告GT匹配IoU0.5，AP在0.5:0.05:0.95计算。AP差值是**百分点**。没有分数校准或测试后调阈值；B原三类响应中选harbor通道后独立NMS，E用单类harbor通道，二者相同协议。不是拿旧bridge报告中的不同口径数字比较。

## 验证与选优（与测试分开）

{valtable}

预先固定主指标为RGB/SAR harbor AP50–95等权平均，每个epoch评估EMA并按该指标保存best，平局保留较早epoch；没有使用被SAR图片量主导的训练器默认fitness。A在两种模态验证上均超过B后结束配置搜索，没有凑满B/C。正式配置只有A，seed43是同配置验证复核。主seed42的best epoch={sel['epoch']}，seed43只检验验证敏感性，不以test在种子间选优。

{trainingtable}

每轮loss、LR、两种模态全指标、FP/FN、耗时、显存、optimizer step、冻结校验见各run的`history.json`、`epochs.csv`、`progress.json`。完整逐图候选及匹配见`evaluations/`、每run的`reloaded_best_val/`和`final_test/`。失败记录：{failures if failures else '无运行失败记录'}。真实短训练不是性能交付：它仅做8步更新/恢复验证，正式实验重新从E0开始。

## 权重、结构与继承

- 原始B：`{man['base_weights']}`；SHA256 `{man['base_sha256']}`，结束复核未变。
- B为`YOLOESegModel / YOLOESegment`，三类`0 ship,1 harbor,2 tank`，不是bridge专家；checkpoint中记录原混合三模态训练配置，但无上游原训练文件清单，不能排除重叠。
- 推荐专家：**`{test['recommended_weights']}`**；SHA256 **`{test['recommended_sha256']}`**。
- E为标准本地`DetectionModel / Detect`，基于B的YOLOE-v8s骨干/neck及已融合分类头，`0=harbor`，适用模态仅rgb/sar。可用`core.load_expert`或本地兼容Ultralytics重载，冻结B的ship输出由独立原模型产生。
- 分割转检测：完整复制检测计算所需模块，删去掩膜分支及未使用提示模块；三个尺度的融合分类卷积明确复制**原harbor第1行→新第0行**。未新随机初始化任何学习参数。已融合语义保存在这些权重内，不重新调用文本编码器或误取ship通道。逐张量继承/删除/形状及数量见`initialization/inheritance_manifest.json`。
- E0原始响应与B harbor分支数值核验通过，完整验证指标一致；E0独立保存于`initialization/E0.pt`。test仍分别运行B和E0作可复核对照。
- 最终导出再次实际重载并完成同一测试，所有候选与选中checkpoint逐项一致；见`final_test/reloaded_export`和`comparison.json`。训练可恢复权重保留在各run的`weights/best.pt,last.pt`，推荐导出权重仅用于推理。

## 冻结、优化与恢复

冻结全部骨干及前部多尺度neck，只训练最后P5融合C2f（model.21）及检测头（model.22），固定DFL投影；这是按结构用途选择，并非任意层数搜索。总参数{freeze['total_parameters']:,}，可训练{freeze['trainable_parameters']:,}，冻结{freeze['frozen_fraction']*100:.3f}%。所有BN运行均值/方差/计数均保持eval，包括可训练模块内BN；选中模块BN affine可训练。每次`model.train()`后重新应用，逐epoch比较所有冻结参数和buffer的哈希；EMA仅平均可训练参数，冻结参数不做会产生舍入漂移的无效平均。

明确使用AdamW，lr0={cfg['lr0']}、betas={cfg['betas']}、weight_decay={cfg['weight_decay']}（bias/一维参数0衰减）；batch={cfg['batch']}、累积={cfg['gradient_accumulation']}、有效batch/nbs={cfg['effective_batch']}、workers=0、FP32/AMP关闭。预热3轮后cosine至0.1倍LR；最大150epoch，patience30，clip10。输入640，等比例PIL双线性letterbox114；SAR放大不增加成像信息。增强仅水平/垂直各0.5翻转，无mosaic、mixup、HSV、旋转或尺寸搜索。RGB每个epoch重复4次、SAR1次；每张独立训练图每轮必出现，按seed+epoch打乱。没有上游训练集回放、伪标签或蒸馏。

训练损失是当前本地`v8DetectionLoss`的box/cls/DFL（7.5/0.5/1.5），不传空mask伪装分割监督。使用显式训练循环控制冻结和跨模态验证，未调用旧bridge/三模态入口。`short_training/update_validation.json`证明有限loss、非零梯度、真实optimizer step、可训练参数变化、原模型及EMA冻结状态不变、保存重载和恢复后下一步一致（数值误差记录其中）。

每个last/best checkpoint包含原训练模型、EMA、optimizer、scheduler、epoch、EMA步数、early-stop状态、Python/NumPy/Torch/CUDA RNG、配置和清单哈希。未按框架结束默认动作剥离恢复状态。恢复从最近**已完成epoch**继续，正在进行的epoch不冒充完成；配置变动须新run并记父权重，不能叫同次resume。执行器单独保存stdout/stderr和子进程真实exit_code，不经tee吞错。

## 数据与协议

正确源路径为`D:\\RSdataset\\DIOR`和`D:\\RSdataset\\Sar_det100k`；本次不沿用旧审计对Sar路径的错误归因。复用了`{man['audit_root']}`，只对本次harbor子集核对真实像素、内容哈希和近重复线索。派生目录：`{man['derived_root']}`。

{dt}

DIOR只用已有HBB；SAR按categories实际解析并以JSON image_id关联文件。像素几何不做无依据±1：DIOR保留原xyxy作为连续边界，SAR转为x,y,x+w,y+h，以真实W/H归一化。原始端点文档仍未确认，该工程约定及反投影样例写入preparation_policy和data_previews；归一化反算误差小于1e-5像素。

对44张存在harbor负起点/越界记录的SAR图整图隔离，未猜测或裁切。原其他类和ship标注完整保存在来源manifest；不因训练单类就删除来源ship记录。图片采用独立复制，不写原件。exact decoded重复按test>val>train保留较高split，split内保留固定字典序；跨split强近重复按预定pHash/像素差规则隔离较低split，所有处理与预测结果无关，详情见`duplicate_review.json`和`all_source_records.json`。这只筛查明显线索，不能证明排除了地理相邻裁片或上游训练重叠。

原train/val/test出处保持不变，原标签从未修改。无额外空图负样本。门控本轮不抽走训练图，记录未来仅在原train按重复/场景组5折交叉拟合的方案，使用折外候选；不从test拟合门控或选阈值。联合视图明确0 ship、1 harbor，单类专家0→联合1；原global1=bridge没有进入本任务。

## 最小组合与限制

组合在同一输入上保留B原三类NMS路径中选出的ship列表，加入E独立NMS后的harbor，不再次跨类NMS、不共享最终max_det。`final_test/ship_preservation.json`检查每个ship框及分数完全一致，船在港口内部不会被删除；这是候选保持性验证，不是宣称全场景ship性能已验证。SAR没有完整ship真值，未做严格SAR双类AP。DIOR原ship标签尚有待核记录，也未据此包装为完整联合评估。

是否值得进入门控：若测试harbor收益成立，可作为下一阶段候选专家；仍须补充经核验的背景/负样本和困难场景评估，依据真实FP/召回代价再设计门控。没有设置用户未授权的ship可接受下降阈值，也没有用“以后门控会解决”忽略FP。两次seed与单一固定子集不证明广泛稳定性；没有验证热红外增强、未知上游重叠、跨地理场景泛化或完整响应蒸馏。

## 复现与文件位置

独立工作树：`{man['worktree']}`，分支`harbor_expert_poc`；未合并、未推送。原工作区保留。Python `{man['python']}`；完整版本与GPU/CUDA见`environment.json`、`pip_freeze.txt`。没有安装/升级依赖。所有日志、权重及评估绝对路径以`experiment_manifest.json`、run的effective_config与process记录为准，不查找“最新runs”。

```powershell
$py = '{man['python']}'
$exp = '{root}'
Set-Location '{WORK}'
# 原数据准备：新实验应使用新的experiment_manifest及派生目录，不覆盖本次输出。
& $py -B -X utf8 tools/harbor_poc/prepare.py --root $exp --stage preview
& $py -B -X utf8 tools/harbor_poc/prepare.py --root $exp --stage finalize
# 新run须独立命名；以下是本次实际训练参数入口。
& $py -B -X utf8 tools/harbor_poc/train.py --root $exp --run A_seed42 --config "$exp/config_A_seed42.json"
# 仅进程已停止且需要继续时恢复，不要启动重复训练。
& $py -B -X utf8 tools/harbor_poc/train.py --root $exp --run A_seed42 --config "$exp/runs/A_seed42/effective_config.json" --resume
# 此命令要求预先锁定final_selection.json；已有final_test目录时主动拒绝覆盖。
& $py -B -X utf8 tools/harbor_poc/final_evaluate.py --root $exp
```

可独立重放推理和指标的入口见`tools/harbor_poc/evaluate.py`，使用显式权重、split与新输出目录；清单和协议保持一致。每轮曲线见`training_curves.png`。代表性成功/误报对照在`final_test/examples/`，绿框GT、红框固定阈值预测；它们仅用于展示，不决定数据清单或指标。机器可读对照表为`metrics_comparison.csv/json`。
'''
    (root/'EXPERIMENT_REPORT.md').write_text(content,encoding='utf8')
    save(root/'metrics_comparison.json',flat)
    fields=['stage','model','modality','images','gt','ap50','ap50_95','precision','recall','tp','fp','fn','candidates']
    with (root/'metrics_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(flat)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for name in runs:
        hist=read(root/'runs'/name/'history.json');x=[h['epoch'] for h in hist]
        for ax,mod in zip(axs[:2],['rgb','sar']):ax.plot(x,[h['metrics'][mod]['ap50_95']*100 for h in hist],label=name);ax.set_title(mod.upper()+' validation AP50-95');ax.set_xlabel('epoch');ax.set_ylabel('%');ax.legend()
        axs[2].plot(x,[h['metrics']['macro_ap50_95']*100 for h in hist],label=name)
    axs[2].set_title('Equal-weight macro AP50-95');axs[2].set_xlabel('epoch');axs[2].set_ylabel('%');axs[2].legend();fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=180);plt.close(fig)
    man.update(status='completed',recommended_weights=test['recommended_weights'],recommended_sha256=test['recommended_sha256'],report=str(root/'EXPERIMENT_REPORT.md'),final_implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip());save(root/'experiment_manifest.json',man)
    (root/'HANDOFF.md').write_text('# Harbor POC completed\n\n'+conclusion+'\n\n'+f'- Report: `{root}/EXPERIMENT_REPORT.md`\n- Root: `{root}`\n- Worktree: `{WORK}`\n- Derived: `{man["derived_root"]}`\n- Weight: `{test["recommended_weights"]}`\n- SHA256: `{test["recommended_sha256"]}`\n\n'+trainingtable+'\n\n'+diff+'\n\nAll planned training, seed replication, locked test and final export reload are complete. No test-based retuning. No gating/IR enhancement. Do not resume completed runs or overwrite old outputs. Commands and complete record locations are in the report.\n',encoding='utf8')
    print('REPORT',root/'EXPERIMENT_REPORT.md',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
