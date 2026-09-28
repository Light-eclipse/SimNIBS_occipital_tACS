"""
stage1_headmodel.py — 두부모델 1단계: 입력 점검 → (charm) → 후두부 QC → 표준 몽타주 전기장 → ROI 요약

대상: ernie(예제)로 먼저 통과시키고, 같은 스크립트를 LEMON·환자 데이터에 그대로 쓴다.
실행: SimNIBS 4.6 동봉 파이썬(simnibs_python). 작업 폴더는 C:\\simnibs_work 같은 짧은 ASCII 경로.

하위 명령
  precheck  T1/T2 헤더 점검 + 실행할 charm 명령 출력           (수 초)
  qc        m2m 폴더의 분할 결과로 후두부 QC (두께·FOV·그림)    (수십 초)
  sim       4x1 Oz 표준 몽타주 전기장 계산                     (수 분~수십 분)
  roi       sim 결과에서 ROI 요약 (analyze_occ.py 확장판)       (수십 초)
  compare   내가 돌린 m2m과 기준 m2m(배포판 m2m_ernie) 비교      (수십 초)
  all       qc → sim → roi 연속 실행

예시 (C:\\simnibs_work 에서)
  simnibs_python stage1_headmodel.py precheck --t1 org/ernie_T1.nii.gz --t2 org/ernie_T2.nii.gz --sub ernie_test
  simnibs_python stage1_headmodel.py all --m2m m2m_ernie
  (charm 완료 후)
  simnibs_python stage1_headmodel.py all --m2m m2m_ernie_test
  simnibs_python stage1_headmodel.py compare --m2m m2m_ernie_test --ref m2m_ernie

산출물: stage1_out/<sub>/ 아래 precheck.json, qc.json, thickness.csv, qc_occipital.png,
        roi.json, compare.json 그리고 stage1_out/summary.csv(대상자당 한 줄, 코호트 집계용)

검증 상태: numpy/nibabel 부분(두께·FOV·Dice·그림)은 합성 영상으로 동작 확인.
          SimNIBS API 부분은 2026-09-13 run_occ.py/analyze_occ.py에서 실제로 돌아간 호출만 재사용.
          라벨 번호와 EEG 좌표 파일 형식은 SimNIBS 4 문서 기준이며, 첫 실행에서 qc.json의
          'labels_present'로 확인할 것.
"""
import argparse
import csv
import datetime
import glob
import json
import os
import sys

import numpy as np
import nibabel as nib

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
OUT_ROOT = 'stage1_out'

# charm final_tissues 라벨 (SimNIBS 4). 4(BONE)는 charm 출력에선 보통 7/8로 나뉘어 나온다.
LABELS = {1: 'WM', 2: 'GM', 3: 'CSF', 4: 'Bone', 5: 'Scalp', 6: 'Eyes',
          7: 'Compact_bone', 8: 'Spongy_bone', 9: 'Blood', 10: 'Muscle'}
BONE = {4, 7, 8}
BRAIN = {1, 2}

# 후두부 QC 대상 전극 (없는 이름은 건너뜀). Cz는 비교 기준.
QC_ELECTRODES = ['Cz', 'Pz', 'POz', 'Oz', 'O1', 'O2', 'PO7', 'PO8', 'Iz', 'O9', 'O10']
FOV_ELECTRODES = ['Oz', 'O1', 'O2', 'Iz', 'O9', 'O10', 'PO7', 'PO8']

# 경고 기준 (근거: 일반 성인 두피 3–8 mm, 두개골 4–10 mm 범위의 대략값. 판정이 아니라 눈으로 볼 대상 표시)
FOV_MARGIN_WARN_MM = 20.0     # 후두부 전극 아래로 영상이 이만큼은 남아야 함
SCALP_RANGE_MM = (2.0, 10.0)
BONE_RANGE_MM = (2.5, 14.0)

# 표준 몽타주: 2026-09-13 run_occ.py와 동일 (결과 비교 가능하도록 고정)
MONTAGE = ['Oz', 'O1', 'O2', 'Pz', 'POz']
CURRENTS = [2e-3, -0.5e-3, -0.5e-3, -0.5e-3, -0.5e-3]

# ROI 정의 두 가지 (2026-09-22 미팅: 표면 전기장과 ROI 평균이 다르게 보여 ROI 정의를 분리)
#  (1) mni_r20  : analyze_occ.py와 동일한 MNI 좌표 구. 9/13 결과와 비교 가능하도록 고정
#  (2) oz_r20/r10: Oz 전극 바로 아래 피질점을 중심으로 한 구. 자극 표적에 붙은 정의
MNI_CENTER = [0, -80, 25]
RADIUS_MM = 20.0
OZ_RADII_MM = (20.0, 10.0)
OZ_ELECTRODE = 'Oz'
THRESHOLDS = (0.30, 0.35)


# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------
def sub_from_m2m(m2m):
    base = os.path.basename(os.path.normpath(m2m))
    return base[4:] if base.startswith('m2m_') else base


def outdir(sub):
    d = os.path.join(OUT_ROOT, sub)
    os.makedirs(d, exist_ok=True)
    return d


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
    print(f'  -> {path}')


def update_summary(sub, fields):
    """stage1_out/summary.csv 에서 sub 행을 갱신(없으면 추가). 열은 합집합."""
    path = os.path.join(OUT_ROOT, 'summary.csv')
    rows = []
    if os.path.exists(path):
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    row = next((r for r in rows if r.get('sub') == sub), None)
    if row is None:
        row = {'sub': sub}
        rows.append(row)
    row.update({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in fields.items()})
    row['updated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    cols = ['sub'] + sorted({k for r in rows for k in r} - {'sub', 'updated'}) + ['updated']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'  -> {path} ({sub} 행 갱신)')


def load_canonical(path):
    """RAS+ 방향으로 맞춘 3D 영상. (x,y,z,1) 형태면 3D로 줄인다. 이후 모든 계산은 이 격자에서 한다."""
    img = nib.load(path)
    if img.ndim > 3:
        img = nib.Nifti1Image(np.asarray(img.dataobj)[..., 0], img.affine, img.header)
    return nib.as_closest_canonical(img)


def world_to_vox(affine, xyz):
    xyz = np.atleast_2d(np.asarray(xyz, float))
    ijk = (np.linalg.inv(affine) @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    return ijk


def sample_labels(lab, affine, xyz):
    """세계좌표 점들의 라벨(최근접). 영상 밖이면 -1."""
    ijk = np.rint(world_to_vox(affine, xyz)).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.array(lab.shape)), axis=1)
    out = np.full(len(ijk), -1, int)
    out[inside] = lab[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]
    return out


def read_eeg_positions(m2m):
    """m2m/eeg_positions/EEG10-10_UI_Jurak_2007.csv → {이름: xyz}.
    SimNIBS 형식: 종류,x,y,z,이름 (Electrode/Fiducial ...). 형식이 달라도 숫자 3개+이름이면 읽는다."""
    cands = sorted(glob.glob(os.path.join(m2m, 'eeg_positions', 'EEG10-10*.csv')))
    if not cands:
        cands = sorted(glob.glob(os.path.join(m2m, 'eeg_positions', '*.csv')))
    if not cands:
        raise FileNotFoundError(f'{m2m}/eeg_positions 에 csv가 없음')
    pos = {}
    with open(cands[0], newline='', encoding='utf-8') as f:
        for r in csv.reader(f):
            nums, name = [], None
            for c in r:
                c = c.strip()
                try:
                    nums.append(float(c))
                except ValueError:
                    if nums and len(nums) >= 3:
                        name = c
            if len(nums) >= 3 and name:
                pos[name] = np.array(nums[:3])
    return pos, cands[0]


# ---------------------------------------------------------------------------
# precheck
# ---------------------------------------------------------------------------
def header_info(path):
    img = nib.load(path)
    hdr = img.header
    zooms = [float(z) for z in hdr.get_zooms()[:3]]
    shape = [int(s) for s in img.shape[:3]]
    q, qc = img.get_qform(coded=True)
    s, sc = img.get_sform(coded=True)
    qs_diff = float(np.abs(q - s).max()) if (q is not None and s is not None) else None
    corners = np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)])
    w = (img.affine @ np.c_[corners, np.ones(8)].T).T[:, :3]
    return {
        'path': path,
        'shape': shape,
        'voxel_mm': zooms,
        'fov_mm': [round(a * b, 1) for a, b in zip(shape, zooms)],
        'orientation': ''.join(nib.aff2axcodes(img.affine)),
        'qform_code': int(qc), 'sform_code': int(sc),
        'qform_sform_maxdiff': qs_diff,
        'world_center_mm': [round(v, 1) for v in w.mean(0)],
        'world_z_range_mm': [round(float(w[:, 2].min()), 1), round(float(w[:, 2].max()), 1)],
    }


def cmd_precheck(a):
    sub = a.sub
    info = {'t1': header_info(a.t1)}
    warn = []
    if a.t2:
        info['t2'] = header_info(a.t2)
    else:
        warn.append('T2 없음: charm 두개골 분할 신뢰도가 떨어진다(SimNIBS 문서). 가능하면 T2 확보.')

    for key in [k for k in ('t1', 't2') if k in info]:
        h = info[key]
        if max(h['voxel_mm']) > 1.2:
            warn.append(f'{key}: 복셀 {h["voxel_mm"]} mm — 1 mm 등방 권장보다 큼.')
        if h['qform_sform_maxdiff'] is not None and h['qform_sform_maxdiff'] > 1e-3 and h['sform_code'] > 0 and h['qform_code'] > 0:
            warn.append(f'{key}: qform/sform 불일치(최대 {h["qform_sform_maxdiff"]:.3g}). 변환 과정 확인. '
                        f'charm의 --forceqform 옵션 검토.')
        if h['fov_mm'][2] < 180 and h['orientation'][2] in 'SI':
            warn.append(f'{key}: 상하 FOV {h["fov_mm"][2]} mm — 두피 상단/후두부 하단이 잘렸을 수 있음. qc에서 확인.')
    if 't2' in info:
        d = np.linalg.norm(np.subtract(info['t1']['world_center_mm'], info['t2']['world_center_mm']))
        info['t1_t2_center_dist_mm'] = round(float(d), 1)
        if d > 30:
            warn.append(f'T1–T2 영상 중심 거리 {d:.0f} mm — 초기 정합이 어려울 수 있음. charm_report에서 T2 정합 확인.')

    cmd = f'charm {sub} {a.t1} {a.t2}' if a.t2 else f'charm {sub} {a.t1}'
    info['charm_command'] = cmd
    info['warnings'] = warn

    print(f'\n[precheck] {sub}')
    for key in [k for k in ('t1', 't2') if k in info]:
        h = info[key]
        print(f'  {key}: {h["shape"]} vox, {h["voxel_mm"]} mm, FOV {h["fov_mm"]} mm, {h["orientation"]}, '
              f'q/s code {h["qform_code"]}/{h["sform_code"]}')
    for w in warn:
        print('  [주의]', w)
    print('\n  다음 명령을 PowerShell에서 실행 (창을 닫지 말 것, 로그 남김):')
    print(f'    {cmd} 2>&1 | Tee-Object charm_{sub}.log\n')
    save_json(info, os.path.join(outdir(sub), 'precheck.json'))


# ---------------------------------------------------------------------------
# qc
# ---------------------------------------------------------------------------
def ray_thickness(lab, affine, start, target, step=0.2, back=8.0, maxlen=60.0):
    """전극점에서 뇌 중심 방향으로 광선을 쏘아 첫 GM/WM까지 조직별 통과 길이(mm).
    비스듬히 통과하므로 수직 두께보다 약간 크게 나온다 — 대상자 간 비교용 지표."""
    d = target - start
    d = d / np.linalg.norm(d)
    t = np.arange(-back, maxlen, step)
    labs = sample_labels(lab, affine, start + t[:, None] * d)
    res = {'scalp': 0.0, 'bone': 0.0, 'csf': 0.0, 'other': 0.0, 'other_detail': {},
           'reached_brain': False, 'outside_image': bool((labs[t >= 0] == -1).any())}
    started = False
    for L in labs:
        if not started:
            if L in (5,) or L in BONE:        # 두피(혹은 바로 뼈)에서부터 센다
                started = True
            else:
                continue
        if L in BRAIN:
            res['reached_brain'] = True
            break
        if L == 5:
            res['scalp'] += step
        elif L in BONE:
            res['bone'] += step
        elif L == 3:
            res['csf'] += step
        else:
            res['other'] += step
            k = LABELS.get(int(L), f'label{int(L)}') if L >= 0 else 'outside'
            res['other_detail'][k] = res['other_detail'].get(k, 0.0) + step
    res['skin_to_cortex'] = res['scalp'] + res['bone'] + res['csf'] + res['other']
    res['other_detail'] = ' '.join(f'{k}:{v:.1f}' for k, v in res['other_detail'].items())
    return res


def fov_margin(lab, affine, xyz, step=1.0, maxlen=200.0):
    """전극에서 세계좌표 -z(아래)로 내려가 영상 경계까지 거리(mm). 전극이 영상 밖이면 음수."""
    if sample_labels(lab, affine, xyz)[0] == -1:
        return -1.0
    t = np.arange(0, maxlen, step)
    pts = xyz + t[:, None] * np.array([0, 0, -1.0])
    labs = sample_labels(lab, affine, pts)
    out = np.where(labs == -1)[0]
    return float(t[out[0]]) if len(out) else maxlen


def draw_qc(t1, lab, affine, pos, png, title):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print('  [건너뜀] matplotlib 없음 → simnibs_python -m pip install matplotlib')
        return False
    colors = ['none', '#e8e8e8', '#7f7f7f', '#4aa3df', '#f2d16b', '#e9967a', '#9b59b6',
              '#f5b041', '#fad7a0', '#c0392b', '#a04000']
    cmap = ListedColormap(colors[:11])
    oz = pos.get('Oz')
    c = np.rint(world_to_vox(affine, oz)[0]).astype(int) if oz is not None else np.array(lab.shape) // 2
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(0))
    cy = int(np.clip(c[1] + round(25 / zooms[1]), 0, lab.shape[1] - 1))  # 관상면은 Oz보다 25 mm 앞 (Oz 높이의 후두엽 단면)
    c = np.clip(c, 0, np.array(lab.shape) - 1)
    plane = {0: c[0], 1: c[2], 2: cy}   # 각 단면이 고정하는 축의 복셀 위치
    fixed = {0: 0, 1: 2, 2: 1}
    lo, hi = np.percentile(t1[t1 > 0], [1, 99.5]) if (t1 > 0).any() else (0, 1)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=110)
    views = [('sagittal (x=Oz)', lambda v: v[c[0], :, :].T, 1, 2),
             ('axial (z=Oz)', lambda v: v[:, :, c[2]].T, 0, 1),
             ('coronal (Oz + 25 mm anterior)', lambda v: v[:, cy, :].T, 0, 2)]
    for col, (name, sl, ax_h, ax_v) in enumerate(views):
        for row in range(2):
            ax = axs[row, col]
            ax.imshow(sl(t1), cmap='gray', origin='lower', vmin=lo, vmax=hi)
            if row == 1:
                L = sl(lab).astype(float)
                L[L <= 0] = np.nan
                ax.imshow(L, cmap=cmap, origin='lower', vmin=0, vmax=10, alpha=0.55, interpolation='nearest')
            for n, p in pos.items():
                if n not in QC_ELECTRODES:
                    continue
                v = world_to_vox(affine, p)[0]
                if abs(v[fixed[col]] - plane[col]) * zooms[fixed[col]] > 12:   # 단면에서 12 mm 이내 전극만
                    continue
                ax.plot(v[ax_h], v[ax_v], 'o', ms=4, mfc='#00e5ff', mec='k', mew=0.5)
                if col != 1:
                    ax.text(v[ax_h] + 2, v[ax_v] + 2, n, fontsize=6, color='#00e5ff')
            ax.set_title(f'{name}' + (' + tissues' if row == 1 else ''), fontsize=9)
            ax.axis('off')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)
    print(f'  -> {png}')
    return True


def cmd_qc(a):
    m2m, sub = a.m2m, sub_from_m2m(a.m2m)
    od = outdir(sub)
    tis_img = load_canonical(os.path.join(m2m, 'final_tissues.nii.gz'))
    lab = np.asarray(tis_img.dataobj).astype(int)
    if lab.ndim == 4:
        lab = lab[..., 0]
    aff = tis_img.affine
    vox_ml = float(np.prod(tis_img.header.get_zooms()[:3])) / 1000.0

    t1_path = os.path.join(m2m, 'T1.nii.gz')
    t1_img = load_canonical(t1_path)
    if t1_img.shape[:3] != lab.shape or not np.allclose(t1_img.affine, aff, atol=1e-3):
        from nibabel.processing import resample_from_to
        t1_img = resample_from_to(t1_img, (lab.shape, aff), order=1)
    t1 = np.asarray(t1_img.dataobj, dtype=float)

    pos, pos_file = read_eeg_positions(m2m)
    present = sorted(int(v) for v in np.unique(lab))
    vols = {LABELS.get(v, f'label{v}'): round(float((lab == v).sum() * vox_ml), 1) for v in present if v > 0}

    brain = np.isin(lab, list(BRAIN))
    ijk = np.argwhere(brain).mean(0)
    center = (aff @ np.r_[ijk, 1])[:3]

    rows, warn = [], []
    for n in QC_ELECTRODES:
        if n not in pos:
            continue
        r = ray_thickness(lab, aff, pos[n], center)
        r['electrode'] = n
        r['label_at_electrode'] = int(sample_labels(lab, aff, pos[n])[0])
        r['fov_margin_below_mm'] = fov_margin(lab, aff, pos[n]) if n in FOV_ELECTRODES else None
        rows.append(r)
        if not r['reached_brain']:
            warn.append(f'{n}: 60 mm 안에 뇌 조직에 닿지 않음 — 전극 위치나 분할 확인')
        if r['reached_brain'] and not (SCALP_RANGE_MM[0] <= r['scalp'] <= SCALP_RANGE_MM[1]):
            warn.append(f'{n}: 두피 통과 길이 {r["scalp"]:.1f} mm (범위 밖)')
        # Iz는 외후두융기 위라 뼈가 두꺼운 게 정상 → 상한 18 mm
        bone_hi = 18.0 if n == 'Iz' else BONE_RANGE_MM[1]
        if r['reached_brain'] and not (BONE_RANGE_MM[0] <= r['bone'] <= bone_hi):
            warn.append(f'{n}: 두개골 통과 길이 {r["bone"]:.1f} mm (범위 밖) — 후두골 분할 확인')
        m = r['fov_margin_below_mm']
        if m is not None and m < FOV_MARGIN_WARN_MM:
            warn.append(f'{n}: 아래쪽 영상 여유 {m:.0f} mm — 후두부 하단 FOV 부족 의심')
    missing = [n for n in MONTAGE if n not in pos]
    if missing:
        warn.append(f'몽타주 전극 좌표 없음: {missing}')

    with open(os.path.join(od, 'thickness.csv'), 'w', newline='', encoding='utf-8') as f:
        cols = ['electrode', 'scalp', 'bone', 'csf', 'other', 'other_detail', 'skin_to_cortex',
                'reached_brain', 'label_at_electrode', 'fov_margin_below_mm', 'outside_image']
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})

    print(f'\n[qc] {sub}  ({m2m})')
    print(f'  라벨: {present}')
    print(f'  {"전극":<5} {"두피":>6} {"뼈":>6} {"CSF":>6} {"기타":>6} {"피부→피질":>9} {"아래여유":>8}  기타 내역')
    for r in rows:
        m = r['fov_margin_below_mm']
        print(f'  {r["electrode"]:<5} {r["scalp"]:6.1f} {r["bone"]:6.1f} {r["csf"]:6.1f} {r["other"]:6.1f} '
              f'{r["skin_to_cortex"]:9.1f} {("" if m is None else f"{m:.0f}"):>8}  {r["other_detail"]}')
    for w_ in warn:
        print('  [주의]', w_)

    ok_png = draw_qc(t1, lab, aff, pos, os.path.join(od, 'qc_occipital.png'), f'{sub} - slices through Oz')
    res = {'m2m': m2m, 'eeg_positions_file': pos_file, 'labels_present': present,
           'tissue_volumes_ml': vols, 'brain_center_world_mm': [round(v, 1) for v in center],
           'electrodes': rows, 'warnings': warn, 'png': ok_png}
    save_json(res, os.path.join(od, 'qc.json'))

    oz = next((r for r in rows if r['electrode'] == 'Oz'), {})
    margins = [r['fov_margin_below_mm'] for r in rows if r['fov_margin_below_mm'] is not None]
    update_summary(sub, {
        'Oz_scalp_mm': oz.get('scalp', ''), 'Oz_bone_mm': oz.get('bone', ''),
        'Oz_csf_mm': oz.get('csf', ''), 'Oz_skin_to_cortex_mm': oz.get('skin_to_cortex', ''),
        'Oz_other_mm': oz.get('other', ''), 'Oz_other_detail': oz.get('other_detail', ''),
        'O1O2_bone_mm': float(np.mean([r['bone'] for r in rows if r['electrode'] in ('O1', 'O2')]))
        if any(r['electrode'] in ('O1', 'O2') for r in rows) else '',
        'O1O2_skin_to_cortex_mm': float(np.mean([r['skin_to_cortex'] for r in rows if r['electrode'] in ('O1', 'O2')]))
        if any(r['electrode'] in ('O1', 'O2') for r in rows) else '',
        'fov_margin_min_mm': min(margins) if margins else '',
        'qc_warnings': len(warn),
    })


# ---------------------------------------------------------------------------
# sim / roi  (SimNIBS 필요)
# ---------------------------------------------------------------------------
def cmd_sim(a):
    from simnibs import sim_struct, run_simnibs
    m2m, sub = a.m2m, sub_from_m2m(a.m2m)
    pathfem = os.path.join(outdir(sub), f'sim_4x1_{a.aniso}')
    if glob.glob(os.path.join(pathfem, '*_TDCS_1_*.msh')) and not a.force:
        print(f'[sim] 이미 결과 있음: {pathfem} (다시 돌리려면 --force)')
        return
    S = sim_struct.SESSION()
    S.subpath = m2m
    S.pathfem = pathfem
    S.fields = 'eE'
    S.map_to_surf = True
    S.open_in_gmsh = False
    tdcs = S.add_tdcslist()
    tdcs.anisotropy_type = a.aniso
    tdcs.currents = CURRENTS
    for i, p in enumerate(MONTAGE):
        el = tdcs.add_electrode()
        el.channelnr = i + 1
        el.centre = p
        el.shape = 'ellipse'
        el.dimensions = [10, 10]
        el.thickness = [4, 1]
    print(f'[sim] {sub}: 4x1 Oz, {a.aniso} → {pathfem}')
    run_simnibs(S)


def find_sim_msh(sub, aniso):
    c = sorted(glob.glob(os.path.join(OUT_ROOT, sub, f'sim_4x1_{aniso}', '*_TDCS_1_*.msh')))
    if not c:
        raise FileNotFoundError(f'{sub}의 sim 결과(.msh)가 없음. 먼저 sim 실행.')
    return c[0]


def roi_stats(E, vols, mask):
    """구 안 GM 요소의 부피 가중 통계."""
    if mask.sum() == 0:
        return {'n_elm': 0}
    w = vols[mask]
    out = {'n_elm': int(mask.sum()),
           'mean_volw': float(np.average(E[mask], weights=w)),
           'p50': float(np.percentile(E[mask], 50)),
           'p99': float(np.percentile(E[mask], 99))}
    for th in THRESHOLDS:
        out[f'frac_gt_{th:.2f}'] = float(100 * np.average(E[mask] > th, weights=w))
    return out


def cmd_roi(a):
    import simnibs
    m2m, sub = a.m2m, sub_from_m2m(a.m2m)
    msh = find_sim_msh(sub, a.aniso)
    mesh = simnibs.read_msh(msh)
    gm = mesh.crop_mesh(simnibs.ElementTags.GM)
    vols = gm.elements_volumes_and_areas()[:]
    E = gm.field['magnE'][:]
    bc = gm.elements_baricenters()[:]

    res = {'msh': msh, 'anisotropy': a.aniso, 'montage': MONTAGE, 'currents_A': CURRENTS,
           'gm_p50': float(np.percentile(E, 50)), 'gm_p95': float(np.percentile(E, 95)),
           'gm_p99': float(np.percentile(E, 99)),
           'gm_mean_volw': float(np.average(E, weights=vols)),
           'rois': {}}

    # (1) MNI 좌표 구 — 9/13 결과와 직접 비교되는 정의
    ctr_mni = np.asarray(simnibs.mni2subject_coords(MNI_CENTER, m2m), float)
    m = np.linalg.norm(bc - ctr_mni, axis=1) < RADIUS_MM
    res['rois'][f'mni_r{RADIUS_MM:.0f}'] = {
        'definition': f'MNI {MNI_CENTER} 를 개인 공간으로 변환한 점, 반경 {RADIUS_MM:.0f} mm',
        'center_mm': [float(v) for v in ctr_mni], **roi_stats(E, vols, m)}
    if m.sum() == 0:
        print('  [주의] MNI ROI 안에 GM 요소가 없음 — toMNI 변환 확인')

    # (2) Oz 전극 바로 아래 피질점 구 — 자극 표적 기준 정의
    ctr_oz = None
    try:
        pos, _ = read_eeg_positions(m2m)
        if OZ_ELECTRODE not in pos:
            raise KeyError(f'{OZ_ELECTRODE} 좌표 없음')
        d_el = np.linalg.norm(bc - pos[OZ_ELECTRODE], axis=1)
        ctr_oz = bc[int(np.argmin(d_el))]          # 전극에서 가장 가까운 GM 요소 = 전극 아래 피질점
        for r in OZ_RADII_MM:
            m = np.linalg.norm(bc - ctr_oz, axis=1) < r
            res['rois'][f'oz_r{r:.0f}'] = {
                'definition': f'{OZ_ELECTRODE} 전극에서 가장 가까운 GM 요소 중심, 반경 {r:.0f} mm',
                'center_mm': [float(v) for v in ctr_oz],
                'electrode_to_cortex_mm': float(d_el.min()), **roi_stats(E, vols, m)}
        res['mni_center_to_oz_center_mm'] = float(np.linalg.norm(ctr_oz - ctr_mni))
    except Exception as e:  # noqa
        res['rois_oz_error'] = f'{type(e).__name__}: {e}'
        print('  [주의] Oz 기준 ROI 계산 실패:', res['rois_oz_error'])

    print(f'\n[roi] {sub} ({a.aniso})')
    print(f'  GM p50/p95/p99 : {res["gm_p50"]:.3f} / {res["gm_p95"]:.3f} / {res["gm_p99"]:.3f} V/m')
    print(f'  {"ROI":<10} {"n_elm":>8} {"mean":>8} {"p99":>8} ' + ' '.join(f'>{th:.2f}'.rjust(8) for th in THRESHOLDS))
    for name, r in res['rois'].items():
        if r.get('n_elm', 0) == 0:
            print(f'  {name:<10} {"비어 있음":>8}')
            continue
        print(f'  {name:<10} {r["n_elm"]:8d} {r["mean_volw"]:8.3f} {r["p99"]:8.3f} '
              + ' '.join(f'{r[f"frac_gt_{th:.2f}"]:7.1f}%' for th in THRESHOLDS))
    if ctr_oz is not None:
        print(f'  Oz 아래 피질점과 MNI ROI 중심 거리 : {res["mni_center_to_oz_center_mm"]:.1f} mm')
        print(f'  Oz 전극 → 피질 거리               : {res["rois"][f"oz_r{OZ_RADII_MM[0]:.0f}"]["electrode_to_cortex_mm"]:.1f} mm')

    save_json(res, os.path.join(outdir(sub), f'roi_{a.aniso}.json'))
    fields = {f'{a.aniso}_gm_p95_Vm': res['gm_p95']}
    for name, r in res['rois'].items():
        if r.get('n_elm', 0):
            fields[f'{a.aniso}_{name}_mean_Vm'] = r['mean_volw']
            fields[f'{a.aniso}_{name}_gt030_pct'] = r['frac_gt_0.30']
    if ctr_oz is not None:
        fields['mni_oz_center_dist_mm'] = res['mni_center_to_oz_center_mm']
    update_summary(sub, fields)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float('nan')


def cmd_compare(a):
    sub, ref = sub_from_m2m(a.m2m), sub_from_m2m(a.ref)
    A = load_canonical(os.path.join(a.m2m, 'final_tissues.nii.gz'))
    B = load_canonical(os.path.join(a.ref, 'final_tissues.nii.gz'))
    if A.shape[:3] != B.shape[:3] or not np.allclose(A.affine, B.affine, atol=1e-3):
        from nibabel.processing import resample_from_to
        A = resample_from_to(A, (B.shape[:3], B.affine), order=0)
        print('  (격자가 달라 기준 격자로 최근접 재표본화)')
    la = np.asarray(A.dataobj).astype(int)
    lb = np.asarray(B.dataobj).astype(int)
    la = la.reshape(la.shape[:3]) if la.ndim > 3 else la   # (x,y,z,1) 형태 대비
    lb = lb.reshape(lb.shape[:3]) if lb.ndim > 3 else lb
    classes = {'WM': {1}, 'GM': {2}, 'CSF': {3}, 'Bone(all)': BONE, 'Scalp': {5}}
    res = {'m2m': a.m2m, 'ref': a.ref,
           'dice': {k: dice(np.isin(la, list(v)), np.isin(lb, list(v))) for k, v in classes.items()}}

    # 후두부만 따로: 기준 모델 Oz 주변 반경 30 mm
    try:
        pos, _ = read_eeg_positions(a.ref)
        oz = pos['Oz']
        ijk = np.indices(lb.shape).reshape(3, -1).T
        w = (B.affine @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
        near = (np.linalg.norm(w - oz, axis=1) < 30).reshape(lb.shape)
        res['dice_occipital_r30'] = {k: dice(np.isin(la, list(v)) & near, np.isin(lb, list(v)) & near)
                                     for k, v in classes.items()}
    except Exception as e:  # noqa
        res['dice_occipital_r30'] = f'계산 안 함: {type(e).__name__}: {e}'
        print('  [주의] 후두부 Dice 계산 실패:', res['dice_occipital_r30'])

    def roi_means(path):
        """roi_*.json → {roi 이름: 평균}. 구버전(단일 'roi' 키)도 읽는다."""
        if not os.path.exists(path):
            return {}
        d = json.load(open(path, encoding='utf-8'))
        if 'rois' in d:
            return {k: v['mean_volw'] for k, v in d['rois'].items() if v.get('n_elm', 0)}
        r = d.get('roi', {})
        return {'mni_r20': r['mean_volw']} if 'mean_volw' in r else {}

    for aniso in ('scalar',):
        ma = roi_means(os.path.join(OUT_ROOT, sub, f'roi_{aniso}.json'))
        mb = roi_means(os.path.join(OUT_ROOT, ref, f'roi_{aniso}.json'))
        common = {k: {sub: ma[k], ref: mb[k], 'diff_pct': 100 * (ma[k] / mb[k] - 1)}
                  for k in ma if k in mb and mb[k]}
        if common:
            res[f'roi_mean_{aniso}'] = common

    print(f'\n[compare] {sub} vs {ref}')
    print('  Dice 전체     : ' + ', '.join(f'{k} {v:.3f}' for k, v in res['dice'].items()))
    if isinstance(res['dice_occipital_r30'], dict):
        print('  Dice 후두 r30 : ' + ', '.join(f'{k} {v:.3f}' for k, v in res['dice_occipital_r30'].items()))
    for name, r in res.get('roi_mean_scalar', {}).items():
        print(f'  ROI {name:<9}: {r[sub]:.3f} vs {r[ref]:.3f} V/m ({r["diff_pct"]:+.1f} %)')
    save_json(res, os.path.join(outdir(sub), f'compare_vs_{ref}.json'))


# ---------------------------------------------------------------------------
def main():
    try:  # Windows 콘솔/Tee-Object에서 인코딩 오류로 죽지 않게
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass
    p = argparse.ArgumentParser(description='두부모델 1단계 파이프라인')
    sp = p.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('precheck')
    s.add_argument('--t1', required=True)
    s.add_argument('--t2')
    s.add_argument('--sub', required=True)
    for name in ('qc', 'sim', 'roi', 'all'):
        s = sp.add_parser(name)
        s.add_argument('--m2m', required=True)
        s.add_argument('--aniso', default='scalar', help="scalar(등방) | vn | dir | mc")
        s.add_argument('--force', action='store_true')
    s = sp.add_parser('compare')
    s.add_argument('--m2m', required=True)
    s.add_argument('--ref', required=True)
    a = p.parse_args()

    if a.cmd == 'precheck':
        cmd_precheck(a)
    elif a.cmd == 'qc':
        cmd_qc(a)
    elif a.cmd == 'sim':
        cmd_sim(a)
    elif a.cmd == 'roi':
        cmd_roi(a)
    elif a.cmd == 'compare':
        cmd_compare(a)
    elif a.cmd == 'all':
        cmd_qc(a)
        cmd_sim(a)
        cmd_roi(a)


if __name__ == '__main__':
    sys.exit(main())
