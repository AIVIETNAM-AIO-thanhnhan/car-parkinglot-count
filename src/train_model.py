"""train_model.py — DT -> RF -> mining -> LightGBM
Chủ sở hữu: P4
"""
import pandas as pd
import glob
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
#hàm gom data
def load_data(file_patten):
    files = glob.glob(file_patten)
    all_data = []
    for i in range(len(files)):
        all_data.append(pd.read_parquet(files[i]))
    data = pd.concat(all_data, ignore_index=True)
    return data


def preprocess_data(data):
    data = data[data['class'] != 'ignore']
    train_dt = data[data['split'] == 'train']
    test_dt = data[data['split'] == 'test']
    val_dt = data[data['split'] == 'val']
    Y_test = (data[data['split'] == 'test']).astype(int)  
    Y_train = (train_dt['class'] == 'car').astype('int')
    Y_val = (val_dt['class'] == 'car').astype('int')
    return train_dt.iloc[:, 8:], Y_train, val_dt.iloc[:, 8:], Y_val, test_dt.iloc[:, 8:],Y_test

def train_model(Xtrain, Ytrain):
    model = DecisionTreeClassifier(
        criterion='entropy', 
        splitter='best', 
        max_depth=10, 
        min_samples_split=500,
        min_samples_leaf=50,
        class_weight='balanced',
        random_state=42
        )
    model.fit(Xtrain, Ytrain)
    return model
def train_model_RF(Xtrain, Ytrain):
    model = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=10,
    min_samples_split=500,
    min_samples_leaf=50,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
    )
    model.fit(X_train, Y_train)
    return model
def train_model_RF(Xtrain, Ytrain):
    model = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=10,
    min_samples_split=500,
    min_samples_leaf=50,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
    )
    model.fit(X_train, Y_train)
    return model
def train_model_hmn(X_new, Y_new):
    """X_new = Xtrain + X_val_predict (False)
       Y_new = ..........
    """
    model_hnm = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=10,
    min_samples_split=500,
    min_samples_leaf=50,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
    model_hnm.fit(X_new, Y_new)
    return model
if __name__ == '__main__':
    file = r'.paquet'
    data = load_data(file)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(data)
    #selec model
    model = train_model #####
    train_acc = model.score(X_train, Y_train)
    test_acc = model.score(X_test, Y_test)
    joblib.dump(model, 'name.pkl')
    print("Train:", train_acc)
    print("Test :", test_acc)
    Y_test_pred = model.predict(X_test)
    print("TEST")
    print(classification_report(Y_test, Y_test_pred))
    print(confusion_matrix(Y_test, Y_test_pred))