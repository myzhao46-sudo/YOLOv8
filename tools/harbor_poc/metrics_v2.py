"""Versioned harbor evaluator: COCO-style 101 recall samples, area=all/maxDets300.

Stable manifest/score order; continuous pixel boxes; no crowd/ignore. This is not
the complete default COCO protocol. No interpolation into unachieved recall.
"""
import numpy as np

VERSION = 'harbor_coco101_v2_20261024'
IOUS = np.linspace(.5, .95, 10)

def ap101(total, tp):
    if total <= 0 or len(tp) == 0:
        return 0.
    tc = np.cumsum(tp, dtype=np.float64)
    recall = tc / total
    precision = tc / np.arange(1, len(tp) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, np.linspace(0, 1, 101), side='left')
    q = np.zeros(101)
    valid = indices < len(envelope)
    q[valid] = envelope[indices[valid]]
    return float(q.mean())

def ious(box, gt):
    inter = np.prod(np.maximum(np.minimum(box[2:], gt[:, 2:]) - np.maximum(box[:2], gt[:, :2]), 0), axis=1)
    area = np.prod(np.maximum(box[2:] - box[:2], 0))
    ga = np.prod(np.maximum(gt[:, 2:] - gt[:, :2], 0), axis=1)
    return inter / np.maximum(area + ga - inter, np.finfo(float).eps)

def evaluate(records, predictions, details=False):
    if set(predictions) != {r['key'] for r in records}:
        raise ValueError('Prediction coverage differs from fixed manifest')
    result = {'metric_version': VERSION}
    all_matches = {}
    for mod in ['rgb', 'sar']:
        rr = [r for r in records if r['modality'] == mod]
        total = sum(len(r['harbor']) for r in rr)
        flat = []; matrices = {}; orders = {}; gt_by = {}
        for r in rr:
            key = r['key']; gt = np.asarray([o['xyxy'] for o in r['harbor']], dtype=np.float64).reshape(-1, 4)
            gt_by[key] = gt
            pp = sorted(predictions[key], key=lambda p: -p['score'])
            if len(pp) > 300 or any(p['class_id'] != 0 for p in pp):
                raise ValueError('Expected single-class harbor NMS output, max_det=300')
            matrix = np.asarray([ious(np.asarray(p['xyxy']), gt) for p in pp]).reshape(len(pp), len(gt))
            matrices[key] = matrix
            # Source GT scan in independent audit prefers later GT on exact ties.
            orders[key] = [np.lexsort((-np.arange(len(gt)), -v)) for v in matrix]
            flat.extend((p['score'], key, j, p) for j, p in enumerate(pp))
        flat.sort(key=lambda p: -p[0])
        aps = []; fixed = {}; matches50 = {}
        for threshold in IOUS:
            used = {r['key']: set() for r in rr}; tp = np.zeros(len(flat)); assignments = []
            for n, (score, key, j, p) in enumerate(flat):
                matched = None
                for gi in orders[key][j]:
                    if matrices[key][j, gi] < threshold: break
                    if int(gi) not in used[key]:
                        matched = int(gi); used[key].add(matched); tp[n] = 1; break
                if threshold == IOUS[0] and details:
                    assignments.append((key, {'prediction_index_in_score_order':j, **p, 'tp_iou50':matched is not None,
                        'gt_index':matched, 'matched_iou':float(matrices[key][j, matched]) if matched is not None else None,
                        'at_report_threshold':score >= .25}))
            aps.append(ap101(total, tp))
            if threshold == IOUS[0]:
                at = np.asarray([p[0] >= .25 for p in flat], dtype=bool)
                nt = int(tp[at].sum()); nf = int(at.sum()) - nt
                fixed = {'tp':nt,'fp':nf,'fn':total-nt,'precision':nt/max(nt+nf,1),'recall':nt/max(total,1)}
                if details:
                    matches50 = {r['key']:{'predictions':[],'gt':gt_by[r['key']].tolist(),
                        'unmatched_gt_at_candidate_floor':[j for j in range(len(gt_by[r['key']])) if j not in used[r['key']]]} for r in rr}
                    for key, assignment in assignments: matches50[key]['predictions'].append(assignment)
        result[mod] = {'images':len(rr),'gt':total,'candidates':len(flat),'ap50':aps[0],'ap50_95':float(np.mean(aps)),
            'ap_by_iou':aps, **fixed}
        all_matches.update(matches50)
    result['macro_ap50_95'] = float(np.mean([result[m]['ap50_95'] for m in ['rgb','sar']]))
    if details: result['per_image_matches'] = all_matches
    return result

def boundary_checks():
    checks = {'no_predictions':ap101(10, []) == 0., 'no_TP':ap101(10, np.zeros(3)) == 0.,
        'low_recall':abs(ap101(10, [1])-11/101) < 1e-15,
        'perfect':ap101(3, [1,1,1]) == 1., 'duplicate_after_TP_AP':ap101(1,[1,0]) == 1.,
        'max_det_300_of_301_recall':abs(ap101(301,np.ones(300))-100/101) < 1e-15}
    r = {'key':'example','modality':'rgb','harbor':[{'xyxy':[0,0,10,10]}]}
    p = {'example':[{'xyxy':[0,0,10,10],'score':.9,'class_id':0},{'xyxy':[0,0,10,10],'score':.8,'class_id':0}]}
    m = evaluate([r],p); checks['duplicate_fixed_TP_FP'] = m['rgb']['tp']==1 and m['rgb']['fp']==1
    try: evaluate([r],{'example':p['example']*151})
    except ValueError: checks['over_cap_rejected'] = True
    else: checks['over_cap_rejected'] = False
    assert all(checks.values()), checks
    return {'passed':True,'checks':checks,'note':'candidate cap is enforced in inference and checked here; AP uses post-NMS candidates'}
