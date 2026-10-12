import sys, time, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
D=sys.argv[1]
tr=pd.read_csv(f"{D}/train.csv"); va=pd.read_csv(f"{D}/validation.csv")
ytr, yva = tr.condition_label, va.condition_label
def ev(y,p): return dict(Accuracy=accuracy_score(y,p),**{"Macro Precision":precision_score(y,p,average="macro"),
  "Macro Recall":recall_score(y,p,average="macro"),"Macro F1":f1_score(y,p,average="macro"),"Weighted F1":f1_score(y,p,average="weighted")})
feats={"BoW unigram":CountVectorizer(lowercase=True,max_features=30000,ngram_range=(1,1)),
       "BoW uni+bigram":CountVectorizer(lowercase=True,max_features=30000,ngram_range=(1,2)),
       "TF-IDF unigram":TfidfVectorizer(lowercase=True,max_features=30000,ngram_range=(1,1)),
       "TF-IDF uni+bigram":TfidfVectorizer(lowercase=True,max_features=30000,ngram_range=(1,2))}
rows=[]; preds={}
for fname,vec in feats.items():
    Xtr=vec.fit_transform(tr.medical_abstract); Xva=vec.transform(va.medical_abstract)   # fit on train only
    models=[("Naive Bayes",f"alpha={a}",MultinomialNB(alpha=a)) for a in [0.01,0.05,0.1,0.5,1.0]]
    if fname.startswith("TF-IDF"):
        for C in [0.1,0.5,1.0,2.0,5.0]:
            for cw in [None,"balanced"]:
                models.append(("Logistic Regression",f"C={C}, class_weight={cw}",LogisticRegression(C=C,class_weight=cw,max_iter=1000,random_state=42)))
                models.append(("Linear SVM",f"C={C}, class_weight={cw}",LinearSVC(C=C,class_weight=cw,random_state=42)))
    for m,cfg,clf in models:
        t=time.time(); clf.fit(Xtr,ytr); tt=time.time()-t; p=clf.predict(Xva)
        rows.append(dict(Model=m,Features=fname,Config=cfg,**ev(yva,p),**{"Train Time (s)":round(tt,2)}))
        preds[(m,fname,cfg)]=p
res=pd.DataFrame(rows).sort_values("Macro F1",ascending=False)
res.to_csv("classical_validation_all_runs.csv",index=False)
best=res.groupby("Model").head(1)
pd.set_option("display.width",250)
print(best.round(4).to_string(index=False))
print("\nLR unigram rows (to compare with teammate 0.8203):")
print(res[(res.Model=="Logistic Regression")&(res.Features=="TF-IDF unigram")].head(3).round(4).to_string(index=False))
print("\nBest per feature set (Macro F1):"); print(res.groupby(["Model","Features"])["Macro F1"].max().round(4).unstack())
for r in best.itertuples():
    pd.DataFrame({"abstract_id":va.abstract_id,"true_label":yva,"predicted_label":preds[(r.Model,r.Features,r.Config)]}).to_csv(
        r.Model.lower().replace(" ","_")+"_validation_predictions.csv",index=False)
