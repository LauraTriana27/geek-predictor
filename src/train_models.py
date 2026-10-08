from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from preprocess import load_dataset,FEATURES

RANDOM_STATE=42
MODEL_BUILDERS={
"Logistic Regression":lambda:Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=2000,random_state=RANDOM_STATE))]),
"Decision Tree":lambda:DecisionTreeClassifier(max_depth=8,min_samples_leaf=4,random_state=RANDOM_STATE),
"Random Forest":lambda:RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=2,random_state=RANDOM_STATE,n_jobs=-1),
"KNN":lambda:Pipeline([("scaler",StandardScaler()),("model",KNeighborsClassifier(n_neighbors=17,weights="distance"))])
}
def train_and_evaluate(df):
    X=df[FEATURES]; y=df["profile"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=RANDOM_STATE,stratify=y)
    results=[]; fitted={}
    labels=sorted(y.unique())
    for name,builder in MODEL_BUILDERS.items():
        model=builder(); model.fit(X_train,y_train); pred=model.predict(X_test)
        results.append({"Model":name,"Accuracy":accuracy_score(y_test,pred),"Precision":precision_score(y_test,pred,average="weighted",zero_division=0),"Recall":recall_score(y_test,pred,average="weighted",zero_division=0),"F1":f1_score(y_test,pred,average="weighted",zero_division=0)})
        fitted[name]=(model,confusion_matrix(y_test,pred,labels=labels))
    metrics=pd.DataFrame(results).sort_values(["F1","Accuracy"],ascending=False).reset_index(drop=True)
    best=metrics.iloc[0]["Model"]; model,cm=fitted[best]
    return metrics,best,model,cm,(X_train,X_test,y_train,y_test)
def main():
    root=Path(__file__).resolve().parents[1]; df=load_dataset(root/"data"/"geek_dataset.csv")
    metrics,best,model,cm,_=train_and_evaluate(df); md=root/"models"; md.mkdir(exist_ok=True)
    joblib.dump(model,md/"best_model.pkl")
    joblib.dump({"model":model,"features":FEATURES,"classes":list(model.classes_),"best_model":best},md/"model_bundle.pkl")
    metrics.to_csv(md/"model_metrics.csv",index=False); np.save(md/"confusion_matrix.npy",cm)
    (md/"metadata.json").write_text(json.dumps({"best_model":best,"random_state":RANDOM_STATE},indent=2),encoding="utf-8")
    print(metrics.to_string(index=False)); print("Selected:",best)
if __name__=="__main__": main()
