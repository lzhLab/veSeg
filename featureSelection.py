import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_selection import SelectFromModel
from joblib import Memory
from sklearn.datasets import load_svmlight_file
import pickle

# @mem.cache
inPath="dataset/combinedFeatureData.txt"
modelPath="dataset/model.dat"

def get_data(inPath):
    data = load_svmlight_file(inPath)
    return data[0].tocsr(), data[1]

seed = 12
X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=seed
)

cls = xgb.XGBClassifier(
    **{
        # "base_score": 0.5,
        # "booster": "gbtree",
        # "colsample_bylevel": 0.6,
        # "colsample_bytree": 0.9,
        # "gamma": 0.0,
        # "learning_rate": 0.01,
        # "max_delta_step": 0.0,
        # "max_depth":20,
        # "min_child_weight": 5.0,
        # "missing": None,
        # "n_estimators": 300,
        "n_jobs": -1,
        # "nthread": -1,
        # # "objective": "multi:softprob",
        # "objective": "multi:softmax",
        # 'num_class':2,
        # "random_state": 0,
        # "reg_alpha": 1.4,
        # "reg_lambda": 1.1,
        # "scale_pos_weight": 2,
        # "seed": 10,
        # "silent": True,
        # "subsample": 0.7,
        'scoring': "neg_log_loss",
        'booster': 'gbtree',
        'colsample_bytree': 0.6,
        'learning_rate': 0.01,
        'max_depth': 7,
        'n_estimators': 300,
        'reg_alpha': 1.4,
        'reg_lambda': 1.1,
        'subsample': 0.7
    }
)

model = SelectFromModel(cls)
# model = pickle.load(open("pima.pickle_xg1.dat", "rb"))

model.fit(X_train, y_train)
X_new = model.transform(X_train)
X_test_new = model.transform(X_test)
feature_names = np.linspace(0,8574,8575,dtype=int)

pickle.dump(model, open(modelPath, "wb"))
model = xgb.XGBClassifier()
param_grid = {
    'booster':['gbtree'],
    'n_estimators': [30],
    'learning_rate': [0.01],
    'colsample_bytree': [0.5],
    'max_depth': [4],
    'reg_alpha': [30],
    'reg_lambda': [1],
    'subsample': [0.5],
    # 'scale_pos_weight':[1],
}

kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train, verbose=1)

# save model to file
pickle.dump(grid_search, open(modelPath, "wb"))

# summarize resultscd
params = grid_result.cv_results_['params']
from sklearn.metrics import classification_report,confusion_matrix

y_test_pred = grid_search.predict(X_test)
y_test_proba = grid_search.predict_proba(X_test)

# target_names=['0','1','2']
target_names=['0','1']

####################ROC Curve######################
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt

y_score = y_test_proba[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)  # 
roc_auc = auc(fpr, tpr)  # 
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

#################### KS Line ######################
def GetKS(y_test, y_pred_prob):
    '''
    function: calculate KS value
    input:
    y_pred_prob: one dimensional array, representing model score
    y_test: true value，representing label（{0,1} or {-1,1}）
    '''
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    ks = max(tpr - fpr)
    # draw ks line
    plt.title('XGBoost(ks = %0.2f)' % ks)
    plt.plot(tpr,label='Tpr')
    plt.plot(fpr,label='Fpr')
    plt.plot(tpr - fpr,label='Tpr-Fpr')
    plt.legend()
    plt.show()
    return fpr, tpr, thresholds, ks

print(GetKS(y_test, y_test_proba))

print(classification_report(y_test,y_test_pred,target_names=target_names))
print("confusion matrix:\n")
# print(confusion_matrix(y_test,y_test_pred,labels=[0,1,2]))
print(confusion_matrix(y_test,y_test_pred,labels=[0,1]))

