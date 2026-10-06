import os,sys,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ['USEDCAR_DEVICE']='cuda'
sys.path.insert(0,str(ROOT/'app'))
from acceleration import enable_gpu
assert enable_gpu()
import numpy as np
import pandas as pd
import joblib
from cuml.accel import is_proxy
from sklearn.model_selection import RandomizedSearchCV
temp_dir = tempfile.TemporaryDirectory()
for name in ['02_random_forest.ipynb','03_linear_regression.ipynb']:
    nb=json.loads((ROOT/'UsedcarAi'/name).read_text())
    ns={'__name__': '__main__'}
    for i in [1,2,3,5]:exec(''.join(nb['cells'][i]['source']),ns)
    ns['X_train']=ns['X_train'].head(512)
    ns['y_train']=ns['y_train'].head(512)
    source=''.join(nb['cells'][7]['source']).replace('n_estimators=300','n_estimators=10')
    exec(source,ns)
    pipe=ns['pipeline']
    if name.startswith('03'):pipe.fit(ns['X_train'],ns['y_train'])
    assert is_proxy(pipe.named_steps['model'])
    x=ns['X_test'].head(16)
    pred=pipe.predict(x)
    assert np.isfinite(pred).all()
    out=Path(temp_dir.name)/(name+'.joblib')
    joblib.dump(pipe,out)
    np.testing.assert_allclose(pred,joblib.load(out).predict(x),rtol=1e-5,atol=1e-3)
    if name.startswith('02'):
        print('Feature importances:',pipe.named_steps['model'].feature_importances_.shape)
        search=RandomizedSearchCV(pipe,{'model__max_depth':[5,10]},n_iter=1,cv=2,n_jobs=1)
        search.fit(ns['X_train'],ns['y_train'])
    else: print('Coefficients:',pipe.named_steps['model'].coef_.shape)
    print('PASS GPU training / prediction / reload:',name,flush=True)
import prediction_app as app
row=app.CLEANED.iloc[0]
keys=['brand','model','sub_model','model_year','mileage','fuel_type','transmission','engine_size','body_type','color','province','location','seller_type','number_of_seats']
result=app.estimate(*[None if pd.isna(row[k]) else row[k] for k in keys])
assert result[-1]=='คำนวณสำเร็จ',result
print('PASS production app estimate:',result[:3],flush=True)
# Check converted production weights against the original CPU exports.
for key, folder in [('rf_model', 'random_forest'), ('lr_model', 'linear_regression')]:
    cpu_model = joblib.load(ROOT/'dataset/models'/folder/(folder+'_pipeline.joblib'))
    gpu_model = app.PROJECT[key]
    assert type(gpu_model.named_steps['model']).__module__.startswith('cuml.')
    batch = app.CLEANED.head(64)[app.PROJECT[key].feature_names_in_]
    np.testing.assert_allclose(cpu_model.predict(batch), gpu_model.predict(batch), rtol=1e-4, atol=1.0)
print('PASS CPU/GPU production prediction agreement on 64 rows', flush=True)
nb = json.loads((ROOT/'UsedcarAi/04_hybrid.ipynb').read_text())
ns = {'__name__': '__main__', 'display': lambda *args, **kwargs: None}
for cell in nb['cells'][:15]:
    if cell['cell_type'] == 'code':
        exec(''.join(cell['source']), ns)
comp = ns['find_comparable'](ns['train_df'].iloc[0])
assert comp[1] >= 0
print('PASS Hybrid notebook comparable selection', flush=True)

temp_dir.cleanup()
