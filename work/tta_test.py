"""How much does horizontal-flip test-time augmentation buy on this model?"""

import numpy as np
import joblib
from sklearn.metrics import accuracy_score


X_val = np.load("validation_set_to_students/images.npy")
y_val = np.load("validation_set_to_students/labels.npy")

model = joblib.load("reference/hog_trim_seed42.joblib")

base = accuracy_score(y_val, model.predict(X_val))
flip = accuracy_score(y_val, model.predict(X_val[:, :, ::-1]))

p0 = model.predict_proba(X_val)
p1 = model.predict_proba(X_val[:, :, ::-1])
tta = accuracy_score(y_val, model.classes_[np.argmax(p0 + p1, axis=1)])

# geometric mean is often a better way to combine two probabilistic views
geo = accuracy_score(y_val, model.classes_[np.argmax(np.sqrt(p0 * p1), axis=1)])

print(f"baseline      : {base:.4f}")
print(f"hflip only    : {flip:.4f}")
print(f"mean of both  : {tta:.4f}   ({tta - base:+.4f})")
print(f"geom mean     : {geo:.4f}   ({geo - base:+.4f})")

# per-member detail, to see whether the gain is systematic
for i, est in enumerate(model.estimators_):
    a = accuracy_score(y_val, est.predict(X_val))
    b = accuracy_score(y_val, est.predict(X_val[:, :, ::-1]))
    q0 = est.predict_proba(X_val)
    q1 = est.predict_proba(X_val[:, :, ::-1])
    t = accuracy_score(y_val, est.classes_[np.argmax(q0 + q1, axis=1)])
    print(f"  member{i}: plain {a:.4f}  flip {b:.4f}  tta {t:.4f}  ({t - a:+.4f})")
