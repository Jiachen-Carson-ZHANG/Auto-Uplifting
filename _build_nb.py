"""Rebuild human_baseline_uplift.ipynb with fetch_x5 loading and Windows UTF-8 fix."""
import json

SETUP = """\
# Windows GBK locale fix: force UTF-8 for all file opens before importing fetch_x5
import builtins as _bt
_orig_open = _bt.open
def _utf8_open(file, mode='r', *a, **kw):
    if 'b' not in str(mode) and 'encoding' not in kw:
        kw['encoding'] = 'utf-8'
    return _orig_open(file, mode, *a, **kw)
_bt.open = _utf8_open

import warnings; warnings.filterwarnings('ignore')
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from sklift.datasets import fetch_x5

# Restore original open() after the import that reads the .rst description file
_bt.open = _orig_open

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier; HAS_XGB = True
except ImportError:
    HAS_XGB = False; print('XGBoost not installed')

try:
    from lightgbm import LGBMClassifier; HAS_LGB = True
except ImportError:
    HAS_LGB = False; print('LightGBM not installed')

plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.float_format', '{:.4f}'.format)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path('artifacts/human_baseline')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print('Setup complete. Output dir:', OUTPUT_DIR)
"""

LOAD_DATA = """\
print('Loading X5 RetailHero dataset via fetch_x5()...')
print('(First run downloads ~200 MB — cached afterwards)')
t0 = time.time()

_bt.open = _utf8_open          # patch open() for the .rst description read
dataset   = fetch_x5()
_bt.open  = _orig_open         # restore

data      = dataset.data
target    = dataset.target      # Series: binary purchase indicator
treatment = dataset.treatment   # Series: binary coupon flag
clients   = data.clients        # demographics: client_id, first_issue_date, first_redeem_date, age, gender
train_raw = data.train          # labeled rows: client_id, first_issue_date, first_redeem_date
purchases = data.purchases      # full transaction history

print(f'Loaded in {time.time()-t0:.1f}s')
print(f'clients  : {clients.shape}   cols={list(clients.columns)}')
print(f'train_raw: {train_raw.shape} cols={list(train_raw.columns)}')
print(f'purchases: {purchases.shape} cols={list(purchases.columns)[:8]}...')
"""

BUILD_LABELED = """\
labeled_raw = train_raw.copy().reset_index(drop=True)
labeled_raw['target']    = target.values
labeled_raw['treatment'] = treatment.values
labeled_raw['client_id'] = labeled_raw['client_id'].astype(str)
clients['client_id']     = clients['client_id'].astype(str)
purchases['client_id']   = purchases['client_id'].astype(str)

ENTITY_KEY    = 'client_id'
TREATMENT_COL = 'treatment'
TARGET_COL    = 'target'

n_treated = int(treatment.sum())
n_control = int((treatment == 0).sum())
rr_t = float(target[treatment == 1].mean())
rr_c = float(target[treatment == 0].mean())
print(f'Treated  : {n_treated:,} ({n_treated/len(treatment):.1%})')
print(f'Control  : {n_control:,} ({n_control/len(treatment):.1%})')
print(f'RR_treat : {rr_t:.4f}  |  RR_ctrl : {rr_c:.4f}  |  ATE : {rr_t-rr_c:+.4f}')
display(pd.crosstab(labeled_raw['treatment'], labeled_raw['target'], margins=True))
"""

EDA_PLOT = """\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

rr = pd.Series({'Control (T=0)': rr_c, 'Treated (T=1)': rr_t})
rr.plot(kind='bar', ax=axes[0], color=['steelblue', 'tomato'], rot=15)
axes[0].set_title('Response Rate by Treatment Group')
axes[0].set_ylabel('Response Rate')
for bar, val in zip(axes[0].patches, rr):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11)

age_v = pd.to_numeric(clients['age'], errors='coerce').dropna()
age_v = age_v[(age_v >= 14) & (age_v <= 100)]
axes[1].hist(age_v, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
axes[1].set_title('Age Distribution'); axes[1].set_xlabel('Age')

gc = clients['gender'].fillna('U').value_counts()
gc.plot(kind='bar', ax=axes[2], color=['tomato', 'steelblue', 'gray'], rot=0)
axes[2].set_title('Gender Distribution')

plt.suptitle('X5 RetailHero Dataset Overview', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'eda_overview.png', bbox_inches='tight', dpi=120)
plt.show()

print(f'Purchase date range: {purchases["transaction_datetime"].min()} to {purchases["transaction_datetime"].max()}')
display(purchases[['purchase_sum', 'product_quantity', 'regular_points_received',
                   'regular_points_spent']].describe())
"""

DEMO_FEAT = """\
labeled_ids = set(labeled_raw['client_id'])
print(f'Labeled customers: {len(labeled_ids):,}')

def build_demographic_features(clients_df, labeled_df):
    demo = labeled_df[['client_id', 'first_issue_date', 'first_redeem_date']].copy()
    demo = demo.merge(clients_df[['client_id', 'age', 'gender']], on='client_id', how='left')

    age = pd.to_numeric(demo['age'], errors='coerce')
    demo['age_clean']       = age.where(age.between(14, 100), np.nan).fillna(-1)
    demo['age_invalid_flag']= (~age.between(14, 100) | age.isna()).astype(int)

    bins = [0, 25, 35, 45, 55, 65, 200]
    lbls = ['le25', '26_35', '36_45', '46_55', '56_65', 'gt65']
    bkt  = pd.cut(demo['age_clean'].replace(-1, np.nan), bins=bins, labels=lbls, right=True)
    for lb in lbls:
        demo[f'age_{lb}'] = (bkt == lb).astype(int)

    gender = demo['gender'].fillna('U').astype(str).str.upper()
    for g in ['F', 'M', 'U']:
        demo[f'gender_{g}'] = (gender == g).astype(int)

    ref    = pd.Timestamp('2019-03-18')
    issue  = pd.to_datetime(demo['first_issue_date'],  errors='coerce')
    redeem = pd.to_datetime(demo['first_redeem_date'], errors='coerce')
    demo['issue_date_missing']   = issue.isna().astype(int)
    demo['account_age_days']     = (ref - issue).dt.total_seconds().div(86400).fillna(-1).round(1)
    demo['has_redeemed']         = (~redeem.isna()).astype(int)
    demo['days_to_first_redeem'] = (redeem - issue).dt.total_seconds().div(86400).fillna(-1).round(1)

    return demo.drop(columns=['age', 'gender', 'first_issue_date', 'first_redeem_date'])

demo_features = build_demographic_features(clients, labeled_raw)
print(f'Demographic features: {demo_features.shape}')
display(demo_features.head(3))
"""

PURCH_FEAT = """\
def build_purchase_features(purchases_df, clients_df, cohort_ids, windows=[30, 60, 90]):
    t0 = time.time()
    purch = purchases_df[purchases_df['client_id'].isin(cohort_ids)].copy()
    purch['transaction_datetime'] = pd.to_datetime(purch['transaction_datetime'], errors='coerce')

    for col in ['purchase_sum', 'product_quantity', 'regular_points_received',
                'express_points_received', 'regular_points_spent', 'express_points_spent']:
        if col in purch.columns:
            purch[col] = pd.to_numeric(purch[col], errors='coerce').fillna(0.0)

    # Anti-leakage: remove pre-card-issue transactions
    issue_map = (clients_df.set_index('client_id')['first_issue_date']
                 .apply(lambda x: pd.to_datetime(x, errors='coerce')).to_dict())
    purch['__iss__'] = purch['client_id'].map(issue_map)
    purch = purch[~(purch['__iss__'].notna() & (purch['transaction_datetime'] < purch['__iss__']))]
    purch = purch.drop(columns=['__iss__'])

    # Collapse product lines to transaction level
    if 'transaction_id' in purch.columns:
        txn = purch.groupby(['client_id', 'transaction_id', 'transaction_datetime'], as_index=False).agg(
            purchase_sum=('purchase_sum', 'max'),
            product_quantity=('product_quantity', 'sum'),
            rpr=('regular_points_received', 'max'),
            epr=('express_points_received', 'max'),
            rps=('regular_points_spent', 'max'),
            eps=('express_points_spent', 'max'),
        )
        txn['points_received'] = txn['rpr'] + txn['epr']
        txn['points_spent']    = txn['rps'] + txn['eps']
        txn = txn.drop(columns=['rpr', 'epr', 'rps', 'eps'])
    else:
        txn = purch.copy()
        txn['points_received'] = txn.get('regular_points_received', 0) + txn.get('express_points_received', 0)
        txn['points_spent']    = txn.get('regular_points_spent', 0)    + txn.get('express_points_spent', 0)

    ref_dt = txn['transaction_datetime'].max()
    print(f'  Reference date: {ref_dt}  |  Txns after anti-leakage: {len(txn):,}')

    def agg_w(df, sfx, ref):
        base = pd.DataFrame({'client_id': list(cohort_ids)})
        if df.empty:
            for c in ['txn_count','purchase_sum','avg_txn','basket_qty','recency_days',
                      'pts_recv','pts_spent','pts_recv_ratio','pts_spent_ratio']:
                base[f'{c}_{sfx}'] = 0.0 if c != 'recency_days' else -1.0
            return base
        g = df.groupby('client_id', as_index=False).agg(
            txn_count=('transaction_datetime', 'count'),
            purchase_sum=('purchase_sum', 'sum'),
            basket_qty=('product_quantity', 'sum'),
            last_dt=('transaction_datetime', 'max'),
            pts_recv=('points_received', 'sum'),
            pts_spent=('points_spent', 'sum'),
        )
        g = base.merge(g, on='client_id', how='left')
        cnt = g['txn_count'].fillna(0);  ps  = g['purchase_sum'].fillna(0)
        pr  = g['pts_recv'].fillna(0);   psp = g['pts_spent'].fillna(0)
        bq  = g['basket_qty'].fillna(0)
        sc  = cnt.where(cnt > 0);        sps = ps.where(ps > 0)
        rec = (ref - pd.to_datetime(g['last_dt'], errors='coerce')).dt.total_seconds().div(86400).fillna(-1)
        r = pd.DataFrame({'client_id': g['client_id']})
        r[f'txn_count_{sfx}']       = cnt.astype(int)
        r[f'purchase_sum_{sfx}']    = ps.round(4)
        r[f'avg_txn_{sfx}']         = (ps / sc).fillna(0).round(4)
        r[f'basket_qty_{sfx}']      = bq.round(4)
        r[f'recency_days_{sfx}']    = rec.round(2)
        r[f'pts_recv_{sfx}']        = pr.round(4)
        r[f'pts_spent_{sfx}']       = psp.round(4)
        r[f'pts_recv_ratio_{sfx}']  = (pr / sps).fillna(0).round(4)
        r[f'pts_spent_ratio_{sfx}'] = (psp / sps).fillna(0).round(4)
        return r

    frames = [agg_w(txn, 'lifetime', ref_dt)]
    for w in windows:
        cutoff = ref_dt - pd.Timedelta(days=w)
        frames.append(agg_w(txn[txn['transaction_datetime'] >= cutoff], f'{w}d', ref_dt))
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on='client_id', how='left')
    print(f'  Purchase features: {result.shape}  elapsed={time.time()-t0:.0f}s')
    return result

print('Building purchase features...')
purchase_features = build_purchase_features(purchases, clients, labeled_ids)
display(purchase_features.head(3))
"""

ASSEMBLE_FEAT = """\
feature_table = demo_features.merge(purchase_features, on='client_id', how='left')
assert feature_table['client_id'].duplicated().sum() == 0, 'Duplicate client_ids!'
assert 'target'    not in feature_table.columns, 'Leakage: target!'
assert 'treatment' not in feature_table.columns, 'Leakage: treatment!'
print(f'Feature table: {feature_table.shape}  — validation passed')
null_pct = feature_table.isnull().mean()
if null_pct.max() > 0:
    display(null_pct[null_pct > 0].sort_values(ascending=False))
else:
    print('No nulls in feature table.')
feature_table.to_csv(OUTPUT_DIR / 'feature_table.csv', index=False)
display(feature_table.describe().T)
"""

SPLIT = """\
labeled = labeled_raw[['client_id', TARGET_COL, TREATMENT_COL]].merge(
    feature_table, on='client_id', how='inner')
print(f'Modeling frame: {labeled.shape}')

strat = labeled[TREATMENT_COL].astype(str) + '_' + labeled[TARGET_COL].astype(str)
print('Stratification key:'); display(strat.value_counts())

idx_all = np.arange(len(labeled))
idx_train, idx_rest = train_test_split(
    idx_all, test_size=0.30, random_state=RANDOM_SEED, stratify=strat)
idx_val, idx_test = train_test_split(
    idx_rest, test_size=0.50, random_state=RANDOM_SEED, stratify=strat.iloc[idx_rest])

train_df = labeled.iloc[idx_train].reset_index(drop=True)
val_df   = labeled.iloc[idx_val].reset_index(drop=True)
test_df  = labeled.iloc[idx_test].reset_index(drop=True)

print(f'train={len(train_df):,} ({len(train_df)/len(labeled):.0%})  '
      f'val={len(val_df):,} ({len(val_df)/len(labeled):.0%})  '
      f'test={len(test_df):,} ({len(test_df)/len(labeled):.0%})')
for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    rt = df.loc[df[TREATMENT_COL]==1, TARGET_COL].mean()
    rc = df.loc[df[TREATMENT_COL]==0, TARGET_COL].mean()
    print(f'  {name}: treat={df[TREATMENT_COL].mean():.3f}  target={df[TARGET_COL].mean():.3f}  '
          f'RR_t={rt:.3f}  RR_c={rc:.3f}  ATE={rt-rc:+.4f}')

FEATURE_COLS = [c for c in train_df.columns
                if c not in {ENTITY_KEY, TREATMENT_COL, TARGET_COL}]
print(f'\\nFeatures: {len(FEATURE_COLS)}  |  {FEATURE_COLS[:10]} ...')
"""

METRICS = """\
def _sf(y, t, u):
    d = pd.DataFrame({'target': np.asarray(y),
                      'treatment': np.asarray(t),
                      'uplift': np.asarray(u, dtype=float)})
    return d.sort_values('uplift', ascending=False, kind='mergesort').reset_index(drop=True)

def qini_curve_data(y, t, u):
    f = _sf(y, t, u); tr = (f['treatment'] == 1); ct = ~tr
    qt = tr.cumsum(); qc = ct.cumsum()
    yt = (f['target'] * tr.astype(int)).cumsum()
    yc = (f['target'] * ct.astype(int)).cumsum()
    sc = qt / qc.replace(0, np.nan)
    return pd.DataFrame({'fraction': (np.arange(len(f))+1)/len(f),
                         'qini': (yt - sc.fillna(0) * yc).astype(float)})

def uplift_curve_data(y, t, u):
    f = _sf(y, t, u); tr = (f['treatment'] == 1); ct = ~tr
    qt = tr.cumsum(); qc = ct.cumsum()
    yt = (f['target'] * tr.astype(int)).cumsum()
    yc = (f['target'] * ct.astype(int)).cumsum()
    ur = ((yt / qt.replace(0, np.nan)) - (yc / qc.replace(0, np.nan))).fillna(0).astype(float)
    return pd.DataFrame({'fraction': (np.arange(len(f))+1)/len(f), 'uplift': ur})

def qini_auc(y, t, u):
    c = qini_curve_data(y, t, u); return round(float(np.trapz(c['qini'], c['fraction'])), 6)

def uplift_auc(y, t, u):
    c = uplift_curve_data(y, t, u); return round(float(np.trapz(c['uplift'], c['fraction'])), 6)

def uplift_at_k(y, t, u, k=0.3):
    f = _sf(y, t, u); n = max(1, int(np.ceil(len(f)*k))); top = f.head(n)
    tr = top[top['treatment']==1]; ct = top[top['treatment']==0]
    if tr.empty or ct.empty: return float('nan')
    return round(float(tr['target'].mean() - ct['target'].mean()), 6)

def decile_table(y, t, u, n_bins=10):
    f = _sf(y, t, u); bins = np.array_split(f.index.to_numpy(), min(n_bins, len(f)))
    rows = []
    for i, idx in enumerate(bins, 1):
        p = f.loc[idx]; tr = p[p['treatment']==1]; ct = p[p['treatment']==0]
        tr_r = float(tr['target'].mean()) if not tr.empty else 0.0
        ct_r = float(ct['target'].mean()) if not ct.empty else 0.0
        rows.append({'decile': i, 'n': len(p), 'treated_n': len(tr), 'control_n': len(ct),
                     'treated_rr': round(tr_r,4), 'control_rr': round(ct_r,4),
                     'obs_uplift': round(tr_r-ct_r,4), 'pred_uplift': round(float(p['uplift'].mean()),4)})
    return pd.DataFrame(rows)

def eval_all(y, t, u, name='model'):
    return {'model': name, 'qini_auc': qini_auc(y,t,u), 'uplift_auc': uplift_auc(y,t,u),
            'uplift@10%': uplift_at_k(y,t,u,0.10), 'uplift@20%': uplift_at_k(y,t,u,0.20),
            'uplift@30%': uplift_at_k(y,t,u,0.30)}

print('Metric functions ready.')
"""

MODEL_CODE = """\
def make_pipeline(base, seed, p=None):
    p = p or {}
    if base == 'logistic_regression':
        return Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()),
            ('clf', LogisticRegression(max_iter=500, solver='liblinear', random_state=seed,
                                       C=p.get('C', 1.0)))])
    if base == 'gradient_boosting':
        return Pipeline([('imp', SimpleImputer(strategy='median')),
            ('clf', GradientBoostingClassifier(
                n_estimators=p.get('n_estimators', 100), learning_rate=p.get('learning_rate', 0.05),
                max_depth=p.get('max_depth', 3), subsample=p.get('subsample', 0.8),
                random_state=seed))])
    if base == 'random_forest':
        return Pipeline([('imp', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=p.get('n_estimators', 200), max_depth=p.get('max_depth', 10),
                min_samples_leaf=p.get('min_samples_leaf', 20), random_state=seed, n_jobs=-1))])
    if base == 'xgboost' and HAS_XGB:
        return Pipeline([('imp', SimpleImputer(strategy='median')),
            ('clf', XGBClassifier(
                n_estimators=p.get('n_estimators', 200), max_depth=p.get('max_depth', 5),
                learning_rate=p.get('learning_rate', 0.05), subsample=p.get('subsample', 0.8),
                colsample_bytree=p.get('colsample_bytree', 0.8),
                eval_metric='logloss', random_state=seed, verbosity=0))])
    if base == 'lightgbm' and HAS_LGB:
        return Pipeline([('imp', SimpleImputer(strategy='median')),
            ('clf', LGBMClassifier(
                n_estimators=p.get('n_estimators', 200), max_depth=p.get('max_depth', 5),
                learning_rate=p.get('learning_rate', 0.05), subsample=p.get('subsample', 0.8),
                colsample_bytree=p.get('colsample_bytree', 0.8),
                random_state=seed, verbose=-1, n_jobs=-1))])
    raise ValueError(f'Unknown: {base}')

def fit_clf(X, y, base, seed, p=None):
    ya = np.asarray(y).astype(int)
    if len(np.unique(ya)) < 2:
        class C:
            def __init__(self, v): self.v = v
            def predict_proba(self, X): return np.column_stack([1-self.v, np.full(len(X), self.v)])
        return C(float(ya.mean()))
    clf = make_pipeline(base, seed, p); clf.fit(X, ya); return clf

def pp(m, X): return np.asarray(m.predict_proba(X))[:, 1]

class UpliftModel:
    def __init__(self, name, family, base, seed=42):
        self.name = name; self.family = family; self.base = base; self.seed = seed
        self.model = self.treat_model = self.ctrl_model = None; self.feat_cols = None

    def fit(self, df, fc, tc, yc, p=None):
        self.feat_cols = fc
        X = df[fc].apply(pd.to_numeric, errors='coerce')
        y = df[yc].astype(int); t = df[tc].astype(int)
        if self.family == 'random': pass
        elif self.family == 'response_model':
            self.model = fit_clf(X, y, self.base, self.seed, p)
        elif self.family == 'two_model':
            mask = (t == 1)
            self.treat_model = fit_clf(X[mask],  y[mask],  self.base, self.seed, p)
            self.ctrl_model  = fit_clf(X[~mask], y[~mask], self.base, self.seed, p)
        elif self.family == 'solo_model':
            Xs = X.copy(); Xs['__t__'] = t.values
            self.model = fit_clf(Xs, y, self.base, self.seed, p)
        elif self.family == 'class_transformation':
            self.model = fit_clf(X, (y == t).astype(int), self.base, self.seed, p)
        return self

    def predict_uplift(self, df):
        if self.family == 'random':
            return np.random.RandomState(self.seed).random_sample(len(df))
        X = df[self.feat_cols].apply(pd.to_numeric, errors='coerce')
        if self.family == 'response_model':    return pp(self.model, X)
        if self.family == 'two_model':         return pp(self.treat_model, X) - pp(self.ctrl_model, X)
        if self.family == 'solo_model':
            Xt = X.copy(); Xt['__t__'] = 1
            Xc = X.copy(); Xc['__t__'] = 0
            return pp(self.model, Xt) - pp(self.model, Xc)
        if self.family == 'class_transformation': return 2.0 * pp(self.model, X) - 1.0

print('UpliftModel + classifier factory ready.')
"""

GRID = """\
MODEL_GRID = [
    ('random_baseline',      'random',               'random'),
    ('response_lr',          'response_model',       'logistic_regression'),
    ('response_gbm',         'response_model',       'gradient_boosting'),
    ('two_model_lr',         'two_model',            'logistic_regression'),
    ('two_model_gbm',        'two_model',            'gradient_boosting'),
    ('two_model_rf',         'two_model',            'random_forest'),
    ('solo_model_lr',        'solo_model',           'logistic_regression'),
    ('solo_model_gbm',       'solo_model',           'gradient_boosting'),
    ('class_transform_lr',   'class_transformation', 'logistic_regression'),
    ('class_transform_gbm',  'class_transformation', 'gradient_boosting'),
]
if HAS_XGB:
    MODEL_GRID += [('two_model_xgb', 'two_model', 'xgboost'),
                   ('solo_model_xgb', 'solo_model', 'xgboost')]
if HAS_LGB:
    MODEL_GRID += [('two_model_lgbm', 'two_model', 'lightgbm'),
                   ('solo_model_lgbm', 'solo_model', 'lightgbm')]
print(f'{len(MODEL_GRID)} models in grid')
for n, f, b in MODEL_GRID:
    print(f'  {n}  ({f} / {b})')
"""

TRAIN_ALL = """\
print('Training all models...')
print('=' * 72)
y_val = val_df[TARGET_COL].values; t_val = val_df[TREATMENT_COL].values
fitted_models = {}; val_results = []

for name, family, base in MODEL_GRID:
    t0 = time.time()
    try:
        m = UpliftModel(name, family, base, seed=RANDOM_SEED)
        m.fit(train_df, FEATURE_COLS, TREATMENT_COL, TARGET_COL)
        sc = m.predict_uplift(val_df)
        met = eval_all(y_val, t_val, sc, name=name)
        val_results.append(met); fitted_models[name] = m
        print(f'  [{name:<28}]  qini={met["qini_auc"]:.4f}  '
              f'u@30%={met["uplift@30%"]:.4f}  ({time.time()-t0:.1f}s)')
    except Exception as e:
        print(f'  [{name}]  FAILED: {e}')

results_df = pd.DataFrame(val_results).set_index('model').sort_values('qini_auc', ascending=False)
print('\\n=== Validation Leaderboard ===')
display(results_df)
"""

TUNE = """\
GBM_GRID = [
    {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 2, 'subsample': 0.8},
    {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3, 'subsample': 0.8},
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3, 'subsample': 0.8},
    {'n_estimators': 200, 'learning_rate': 0.03, 'max_depth': 3, 'subsample': 0.8},
    {'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 3, 'subsample': 0.7},
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 4, 'subsample': 0.8},
]
XGB_GRID = [
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.7},
    {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8},
]
LGB_GRID = [
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.7, 'colsample_bytree': 0.7},
    {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8},
]

def tune(family, base, grid, prefix):
    best_s, best_p, best_m = -np.inf, None, None; log = []
    for i, p in enumerate(grid):
        try:
            m = UpliftModel(f'{prefix}_c{i}', family, base, seed=RANDOM_SEED)
            m.fit(train_df, FEATURE_COLS, TREATMENT_COL, TARGET_COL, p=p)
            s = qini_auc(y_val, t_val, m.predict_uplift(val_df))
            log.append({'i': i, 'params': p, 'qini': s})
            if s > best_s: best_s = s; best_p = p; best_m = m; best_m.name = prefix
            print(f'    cfg{i}: qini={s:.5f}  {p}')
        except Exception as e: print(f'    cfg{i}: FAILED {e}')
    return best_m, best_p, best_s, log

tuned_models = {}; tuning_logs = {}
top3 = results_df.drop(index='random_baseline', errors='ignore').head(3)
top_names = list(top3.index)
print('Top-3 for tuning:'); display(top3)

if any('gbm' in n for n in top_names):
    bn = next(n for n in top_names if 'gbm' in n)
    fam = fitted_models[bn].family
    print(f'\\nTuning GBM: {bn} ({fam})')
    m, p, s, log = tune(fam, 'gradient_boosting', GBM_GRID, f'tuned_{bn}')
    if m: tuned_models[m.name] = m; tuning_logs[m.name] = log; print(f'  Best qini={s:.5f}')

if HAS_XGB and any('xgb' in n for n in top_names):
    bn = next(n for n in top_names if 'xgb' in n)
    fam = fitted_models[bn].family
    print(f'\\nTuning XGB: {bn} ({fam})')
    m, p, s, log = tune(fam, 'xgboost', XGB_GRID, f'tuned_{bn}')
    if m: tuned_models[m.name] = m; tuning_logs[m.name] = log; print(f'  Best qini={s:.5f}')

if HAS_LGB and any('lgbm' in n for n in top_names):
    bn = next(n for n in top_names if 'lgbm' in n)
    fam = fitted_models[bn].family
    print(f'\\nTuning LGB: {bn} ({fam})')
    m, p, s, log = tune(fam, 'lightgbm', LGB_GRID, f'tuned_{bn}')
    if m: tuned_models[m.name] = m; tuning_logs[m.name] = log; print(f'  Best qini={s:.5f}')
"""

FULL_LB = """\
all_models = {**fitted_models, **tuned_models}
all_val    = list(val_results)
for name, m in tuned_models.items():
    all_val.append(eval_all(y_val, t_val, m.predict_uplift(val_df), name=name))
full_lb = pd.DataFrame(all_val).set_index('model').sort_values('qini_auc', ascending=False)
print('=== Full Validation Leaderboard (with tuned) ==='); display(full_lb)
"""

CHAMPION = """\
champion_name = full_lb.drop(index='random_baseline', errors='ignore').index[0]
champion = all_models[champion_name]
print(f'Champion: {champion_name}  (family={champion.family}, base={champion.base})')
print(f'Val Qini AUC: {full_lb.loc[champion_name, "qini_auc"]:.5f}')
"""

TEST_EVAL = """\
y_test = test_df[TARGET_COL].values; t_test = test_df[TREATMENT_COL].values
test_scores = champion.predict_uplift(test_df)

test_res = []
for name, m in all_models.items():
    try: test_res.append(eval_all(y_test, t_test, m.predict_uplift(test_df), name=name))
    except Exception as e: print(f'  {name} failed: {e}')

test_lb = pd.DataFrame(test_res).set_index('model').sort_values('qini_auc', ascending=False)
print('=== Held-Out Test Leaderboard ==='); display(test_lb)

cmp = full_lb[['qini_auc']].join(test_lb[['qini_auc']], lsuffix='_val', rsuffix='_test')
cmp['gap'] = (cmp['qini_auc_val'] - cmp['qini_auc_test']).round(5)
print('\\n=== Generalisation Gap ===')
display(cmp.sort_values('qini_auc_val', ascending=False))
print(f'Champion gap: {cmp.loc[champion_name, "gap"]:.5f}')
"""

CURVES = """\
top5 = test_lb.drop(index='random_baseline', errors='ignore').head(5).index.tolist()
plot_m = ['random_baseline'] + top5
colors = plt.cm.tab10(np.linspace(0, 1, len(plot_m)))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for i, mn in enumerate(plot_m):
    m = all_models.get(mn)
    if m is None: continue
    sc = m.predict_uplift(test_df)
    lw = 2.5 if mn == champion_name else 1.2
    lbl = f'{mn} ({test_lb.loc[mn, "qini_auc"]:.4f})'
    q = qini_curve_data(y_test, t_test, sc)
    axes[0].plot(q['fraction'], q['qini'], color=colors[i], lw=lw, label=lbl)
    u = uplift_curve_data(y_test, t_test, sc)
    axes[1].plot(u['fraction'], u['uplift'], color=colors[i], lw=lw, label=lbl)

for ax, ttl, yl in [(axes[0], 'Qini Curve (Test)', 'Cumulative Qini'),
                    (axes[1], 'Uplift Curve (Test)', 'Treat-Ctrl Response Rate')]:
    ax.axhline(0, color='black', linestyle='--', lw=0.8)
    ax.set_title(ttl, fontsize=12)
    ax.set_xlabel('Fraction targeted (descending uplift)')
    ax.set_ylabel(yl); ax.legend(fontsize=7, loc='upper left')

plt.suptitle(f'Champion: {champion_name}', fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'qini_uplift_curves.png', bbox_inches='tight', dpi=120)
plt.show()
"""

DECILE = """\
champ_dec = decile_table(y_test, t_test, test_scores)
print(f'=== Champion Decile Table ({champion_name}) ==='); display(champ_dec)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bc = ['tomato' if v < 0 else 'steelblue' for v in champ_dec['obs_uplift']]
axes[0].bar(champ_dec['decile'], champ_dec['obs_uplift'], color=bc, edgecolor='white')
axes[0].axhline(0, color='black', linestyle='--', lw=1)
axes[0].set_title('Observed Uplift by Decile (1 = highest predicted)')
axes[0].set_xlabel('Decile'); axes[0].set_ylabel('Treat-Ctrl Response Rate')

x = np.arange(len(champ_dec)); w = 0.35
axes[1].bar(x-w/2, champ_dec['treated_rr'], w, label='Treated', color='tomato', alpha=0.85)
axes[1].bar(x+w/2, champ_dec['control_rr'], w, label='Control', color='steelblue', alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(champ_dec['decile'])
axes[1].set_title('Response Rates by Decile')
axes[1].set_xlabel('Decile'); axes[1].legend()

plt.suptitle(f'Champion: {champion_name}', fontsize=13); plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'champion_decile.png', bbox_inches='tight', dpi=120); plt.show()
"""

FEAT_IMP = """\
def get_native_imp(m):
    underlying = m.treat_model if m.family == 'two_model' else m.model
    if underlying is None: return None
    if hasattr(underlying, 'named_steps'):
        underlying = underlying.named_steps.get('clf', underlying)
    cols = m.feat_cols
    if hasattr(underlying, 'feature_importances_'):
        return pd.Series(underlying.feature_importances_, index=cols).sort_values(ascending=False)
    if hasattr(underlying, 'coef_'):
        return pd.Series(np.abs(underlying.coef_[0]), index=cols).sort_values(ascending=False)
    return None

nimp = get_native_imp(champion)
if nimp is not None:
    top25 = nimp.head(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    top25[::-1].plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Top-25 Feature Importances — {champion_name}\\n(treatment arm)')
    ax.set_xlabel('Importance'); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', bbox_inches='tight', dpi=120); plt.show()
    display(top25.to_frame('importance'))
else:
    print('Native importance not available for this model type.')

# Permutation importance
print('\\nPermutation importance on 2000-row val sample...')
vs = val_df.sample(min(2000, len(val_df)), random_state=RANDOM_SEED)
ys = vs[TARGET_COL].values; ts = vs[TREATMENT_COL].values
bq = qini_auc(ys, ts, champion.predict_uplift(vs))
print(f'Baseline Qini (sample): {bq:.5f}')

pimp = {}
for feat in FEATURE_COLS:
    sh = vs.copy(); sh[feat] = sh[feat].sample(frac=1, random_state=RANDOM_SEED).values
    pimp[feat] = bq - qini_auc(ys, ts, champion.predict_uplift(sh))

ps = pd.Series(pimp).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 8))
t25 = ps.head(25)
bc = ['tomato' if v < 0 else 'steelblue' for v in t25.values[::-1]]
t25[::-1].plot(kind='barh', ax=ax, color=bc)
ax.axvline(0, color='black', lw=0.8)
ax.set_title(f'Permutation Importance — {champion_name}\\n(Qini AUC drop when feature shuffled)')
ax.set_xlabel('Qini drop'); plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'permutation_importance.png', bbox_inches='tight', dpi=120); plt.show()
display(ps.head(25).to_frame('qini_drop'))
"""

SEGMENT = """\
X_t = test_df[champion.feat_cols].apply(pd.to_numeric, errors='coerce')
if champion.family == 'two_model':
    p_t = pp(champion.treat_model, X_t); p_c = pp(champion.ctrl_model, X_t)
elif champion.family == 'solo_model':
    Xt = X_t.copy(); Xt['__t__'] = 1; Xc = X_t.copy(); Xc['__t__'] = 0
    p_t = pp(champion.model, Xt); p_c = pp(champion.model, Xc)
elif champion.family == 'response_model':
    p_t = pp(champion.model, X_t); p_c = p_t.copy()
else:
    p_t = test_scores; p_c = np.zeros_like(p_t)

seg_df = pd.DataFrame({
    'client_id': test_df['client_id'].values,
    'uplift_score': test_scores, 'p_treat': p_t, 'p_ctrl': p_c,
    'true_target': y_test, 'true_treatment': t_test,
})

up75 = np.percentile(test_scores, 75); up25 = np.percentile(test_scores, 25)
PH = 0.40; PL = 0.25
print(f'Uplift p25={up25:.4f}  p75={up75:.4f}  |  P_HIGH={PH}  P_LOW={PL}')

def segment(r):
    u = r['uplift_score']; pt = r['p_treat']; pc = r['p_ctrl']
    if u >= up75 and pt >= PH and pc < PH:  return 'Persuadable'
    if u < 0  and pc >= PH and pt < PH:     return 'Sleeping Dog'
    if pt >= PH and pc >= PH:               return 'Sure Thing'
    if pt < PL  and pc < PL:                return 'Lost Cause'
    return 'Uncertain'

seg_df['segment'] = seg_df.apply(segment, axis=1)
ss = seg_df.groupby('segment').agg(
    count=('client_id','count'), avg_uplift=('uplift_score','mean'),
    avg_p_treat=('p_treat','mean'), avg_p_ctrl=('p_ctrl','mean')).reset_index()
ss['pct'] = (ss['count'] / len(seg_df) * 100).round(1)
ss = ss.sort_values('avg_uplift', ascending=False)
print('=== Customer Segment Summary ==='); display(ss)
"""

SEG_PLOT = """\
SC = {'Persuadable': '#2196F3', 'Sure Thing': '#4CAF50',
      'Lost Cause': '#9E9E9E', 'Sleeping Dog': '#F44336', 'Uncertain': '#FF9800'}

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

ax0 = fig.add_subplot(gs[0, 0])
cnts = seg_df['segment'].value_counts()
ax0.pie(cnts.values, labels=cnts.index,
        colors=[SC.get(s,'gray') for s in cnts.index],
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
ax0.set_title('Segment Distribution')

ax1 = fig.add_subplot(gs[0, 1])
for seg, color in SC.items():
    sub = seg_df[seg_df['segment'] == seg]['uplift_score']
    if len(sub) > 5: ax1.hist(sub, bins=30, alpha=0.55, label=seg, color=color, density=True)
ax1.axvline(0, color='black', linestyle='--', lw=0.8)
ax1.set_title('Uplift Score by Segment'); ax1.set_xlabel('Score')
ax1.set_ylabel('Density'); ax1.legend(fontsize=8)

ax2 = fig.add_subplot(gs[1, 0])
smp = seg_df.sample(min(4000, len(seg_df)), random_state=RANDOM_SEED)
for seg, color in SC.items():
    sub = smp[smp['segment'] == seg]
    ax2.scatter(sub['p_ctrl'], sub['p_treat'], c=color, alpha=0.4, s=8, label=seg)
ax2.plot([0,1],[0,1],'k--',lw=0.8); ax2.set_xlim(0,1); ax2.set_ylim(0,1)
ax2.set_xlabel('P(buy | control)'); ax2.set_ylabel('P(buy | treatment)')
ax2.set_title('P(treat) vs P(ctrl)'); ax2.legend(fontsize=7, markerscale=2)

ax3 = fig.add_subplot(gs[1, 1])
so = ss.sort_values('avg_uplift', ascending=True)
ax3.barh(so['segment'], so['avg_uplift'],
         color=[SC.get(s,'gray') for s in so['segment']], edgecolor='white')
ax3.axvline(0, color='black', lw=0.8)
ax3.set_title('Avg Predicted Uplift by Segment'); ax3.set_xlabel('Mean Uplift')

plt.suptitle('Customer Uplift Segmentation', fontsize=14, y=1.01)
plt.savefig(OUTPUT_DIR / 'customer_segmentation.png', bbox_inches='tight', dpi=120); plt.show()
"""

PROFILE = """\
seg_w = seg_df.merge(feature_table, on='client_id', how='left')
prof_cols = [c for c in ['age_clean','gender_F','gender_M','has_redeemed','account_age_days',
    'days_to_first_redeem','txn_count_lifetime','purchase_sum_lifetime',
    'recency_days_lifetime','txn_count_90d','purchase_sum_90d'] if c in seg_w.columns]
print('=== Segment Profiles ===')
display(seg_w.groupby('segment')[prof_cols].mean().round(2).T)

prs = seg_w[seg_w['segment'] == 'Persuadable']
oth = seg_w[seg_w['segment'] != 'Persuadable']
print(f'\\nPersuadables: {len(prs):,}  Others: {len(oth):,}')
rows = []
for col in prof_cols:
    a = prs[col].dropna(); b = oth[col].dropna()
    if len(a) > 5 and len(b) > 5:
        _, pv = stats.mannwhitneyu(a, b, alternative='two-sided')
        rows.append({'feature': col, 'persuadable_mean': round(a.mean(),3),
                     'others_mean': round(b.mean(),3), 'diff': round(a.mean()-b.mean(),3),
                     'p_value': round(pv,5), 'sig': pv < 0.05})
display(pd.DataFrame(rows).sort_values('p_value'))
"""

POLICY = """\
cutoffs = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
cost_sc = {'zero_cost': 0.0, 'low_cost': 0.01, 'medium_cost': 0.05}
n_t = len(test_df); rows = []

for cutoff in cutoffs:
    obs = uplift_at_k(y_test, t_test, test_scores, k=cutoff)
    nc = int(np.ceil(n_t * cutoff))
    for sc, cost in cost_sc.items():
        gain = obs * nc - nc * cost if not np.isnan(obs) else np.nan
        rows.append({'cutoff%': int(cutoff*100), 'n_contacted': nc,
                     'obs_uplift': round(obs,5), 'scenario': sc,
                     'gain': round(float(gain),2) if not np.isnan(gain) else np.nan})

pol = pd.DataFrame(rows)
display(pol.pivot_table(index=['cutoff%','n_contacted','obs_uplift'],
                         columns='scenario', values='gain').reset_index())

opt = pol[pol['scenario']=='zero_cost']
opt = opt.loc[opt['gain'].idxmax()]
print(f"Optimal (zero-cost): top {opt['cutoff%']}% ({opt['n_contacted']:,} customers, gain={opt['gain']:.1f})")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for sc in cost_sc:
    sub = pol[pol['scenario'] == sc]
    axes[0].plot(sub['cutoff%'], sub['gain'], marker='o', label=sc)
axes[0].axhline(0, color='black', linestyle='--', lw=0.8)
axes[0].set_title('Policy Gain by Cutoff'); axes[0].legend()
axes[0].set_xlabel('% Targeted'); axes[0].set_ylabel('Gain')

zc = pol[pol['scenario'] == 'zero_cost']
axes[1].plot(zc['cutoff%'], zc['obs_uplift'], marker='o', color='steelblue')
axes[1].axhline(0, color='black', linestyle='--', lw=0.8)
axes[1].set_title('Observed Uplift@k'); axes[1].set_xlabel('% Targeted')
axes[1].set_ylabel('Treat-Ctrl Response Rate')

plt.suptitle('Policy Analysis', fontsize=13); plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'policy_analysis.png', bbox_inches='tight', dpi=120); plt.show()
"""

SUMMARY = """\
full_lb.to_csv(OUTPUT_DIR / 'val_leaderboard.csv')
test_lb.to_csv(OUTPUT_DIR / 'test_leaderboard.csv')
champ_dec.to_csv(OUTPUT_DIR / 'champion_decile_table.csv', index=False)
ss.to_csv(OUTPUT_DIR / 'segment_summary.csv', index=False)
seg_df.to_csv(OUTPUT_DIR / 'segment_assignments_test.csv', index=False)
print(f'Artefacts saved to {OUTPUT_DIR}/')

print('\\n' + '='*70)
print('HUMAN BASELINE — FINAL SUMMARY')
print('='*70)
print(f'Champion : {champion_name}  (family={champion.family}, base={champion.base})')
print('\\n[Validation]')
for k in ['qini_auc','uplift_auc','uplift@10%','uplift@30%']:
    print(f'  {k:<18}: {full_lb.loc[champion_name,k]:.5f}')
print('\\n[Held-out Test]')
for k in ['qini_auc','uplift_auc','uplift@10%','uplift@30%']:
    print(f'  {k:<18}: {test_lb.loc[champion_name,k]:.5f}')
print('\\n[Segments (test set)]')
for _, r in ss.iterrows():
    print(f'  {r["segment"]:14s}: {r["count"]:6,} ({r["pct"]:.1f}%)  avg_uplift={r["avg_uplift"]:.4f}')
print('\\n[Business Actions]')
print('  PROMOTE  → Persuadables : incremental buyers — best promotion ROI')
print('  SKIP     → Sure Things  : buy regardless — promotion is wasted spend')
print('  SKIP     → Lost Causes  : low propensity — promotion has no effect')
print('  AVOID    → Sleeping Dogs: promotion REDUCES purchase — do not contact')
print('='*70)
"""

def md(source): return {"cell_type": "markdown", "metadata": {}, "source": [source]}
def code(source): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [source]}

cells = [
    md("# Human Baseline: X5 RetailHero Uplift Modeling\n\n"
       "Identifies **persuadable** customers (buy only when promoted) and flags where promotion is wasted.\n\n"
       "**Pipeline:** Data loading (fetch_x5) → Feature engineering → Split 70/15/15 → "
       "Model grid training → Hyperparameter tuning → Champion selection → "
       "XAI → Customer segmentation → Policy analysis"),
    md("## 0. Setup"),
    code(SETUP),
    md("## 1. Data Loading & EDA\n\n"
       "`fetch_x5()` downloads and caches the dataset (~200 MB on first run)."),
    code(LOAD_DATA),
    code(BUILD_LABELED),
    code(EDA_PLOT),
    md("## 2. Feature Engineering\n\n"
       "Customer-level table (one row per customer) with demographic, lifetime RFM, "
       "and rolling 30d/60d/90d features. No leakage: `target` and `treatment` never used as features."),
    code(DEMO_FEAT),
    code(PURCH_FEAT),
    code(ASSEMBLE_FEAT),
    md("## 3. Train / Validation / Test Split (70 / 15 / 15)\n\n"
       "Stratified on the joint `treatment × target` key to preserve both rates across splits."),
    code(SPLIT),
    md("## 4. Uplift Metric Functions\n\n"
       "Qini AUC, Uplift AUC, Uplift@k, and decile table implemented from scratch "
       "(matching `src/uplift/metrics.py` for a fair comparison with the agentic pipeline)."),
    code(METRICS),
    md("## 5. Uplift Model Training\n\n"
       "| Learner | Description |\n|---|---|\n"
       "| Random | Lower bound |\n"
       "| Response model | P(Y=1\\|X) as proxy — ignores treatment |\n"
       "| T-Learner (two-model) | Separate models for T=1 / T=0; uplift = P_treat − P_ctrl |\n"
       "| S-Learner (solo-model) | Single model with T as feature; counterfactual diff |\n"
       "| Class transformation | Relabel y'=1 iff Y=T; uplift = 2P−1 |"),
    code(MODEL_CODE),
    code(GRID),
    code(TRAIN_ALL),
    md("## 6. Hyperparameter Tuning for Top-3 Candidates\n\n"
       "Manual grid search evaluated on validation Qini AUC."),
    code(TUNE),
    code(FULL_LB),
    md("## 7. Champion Selection & Held-Out Test Evaluation"),
    code(CHAMPION),
    code(TEST_EVAL),
    md("## 8. Qini Curves, Uplift Curves & Decile Analysis"),
    code(CURVES),
    code(DECILE),
    md("## 9. Feature Importance & XAI\n\n"
       "Native importance (from the treatment-arm model) + permutation importance "
       "(Δ Qini AUC when each feature is shuffled on a 2k validation sample)."),
    code(FEAT_IMP),
    md("## 10. Customer Segmentation\n\n"
       "| Segment | Uplift | P(buy\\|T) | P(buy\\|C) | Action |\n|---|---|---|---|---|\n"
       "| **Persuadables** | High | High | Low | **Promote** |\n"
       "| **Sure Things** | ~0 | High | High | **Skip** — buy anyway |\n"
       "| **Lost Causes** | ~0 | Low | Low | **Skip** — promo won't help |\n"
       "| **Sleeping Dogs** | Negative | Low | High | **Avoid** — promo hurts |"),
    code(SEGMENT),
    code(SEG_PLOT),
    code(PROFILE),
    md("## 11. Policy Analysis\n\n"
       "Policy gain at different targeting cutoffs under zero / low / medium communication cost."),
    code(POLICY),
    md("## 12. Summary & Business Recommendations"),
    code(SUMMARY),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open("human_baseline_uplift.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written {len(cells)} cells to human_baseline_uplift.ipynb")
