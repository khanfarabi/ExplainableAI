#Statstical Analysis
class stat_ana:
                @classmethod
                def stat_ana(cls,spps):
                                import statistics
                                from statistics import stdev
                                import math
                                from sklearn.feature_extraction.text import TfidfVectorizer
                                from sklearn.model_selection import train_test_split
                                import math
                                import sklearn
                                from sklearn.feature_extraction.text import TfidfVectorizer
                                from sklearn.model_selection import train_test_split
                                import numpy as np
                                import shap
                                import transformers
                                import shap
                                from sklearn.metrics import (precision_score,recall_score,f1_score)
                                from sklearn import svm
                                from sklearn.svm import SVC
                                from sklearn.svm import LinearSVC
                                shap_exp=spps
                                pred_per_patch={}
                                shap_exp_w={}
                                shap_exp_r={}
                                for v in shap_exp:
                                    gh=[]
                                    gh1=[]
                                    for kk in shap_exp[v]:
                                        if kk.isdigit()==True:
                                            gh.append(kk)
                                        elif kk.isdigit()==False:
                                            gh1.append(kk)
                                    shap_exp_w[v]=gh1
                                    shap_exp_r[v]=gh

                                evid_per_batch_tr={}
                                evid_per_batch_tg={}
                                evid_per_batch_all={}
                                def train_exp(n,shap_exp_w,shap_exp_r,qrat):
                                    train={}
                                    for vv in shap_exp_r:
                                        evid=[]
                                        rl=math.ceil(len(shap_exp_r[vv])*n)
                                        wl=math.ceil(len(shap_exp_w[vv])*n)
                                        if wl==0:
                                            continue
                                        else:
                                                for xz in range(0,rl):
                                                    pass#evid.append(shap_exp_r[vv][xz])
                                                for xz1 in range(0,wl):
                                                    evid.append(shap_exp_w[vv][xz1])
                                                #print(vv,evid)
                                                train[vv]=evid
                                    tr=[]
                                    tg=[]
                                    for gg in train:
                                        s=''
                                        for gg1 in train[gg]:
                                            s=s+str(gg1)+" "
                                            tr.append(gg1)
                                            tg.append(qrat[str(gg)])

                                    return tr,tg,train


                                for dd in range(1,11):
                                    tr,tg,train=train_exp(dd/10,shap_exp_w,shap_exp_r,qrat)
                                    evid_per_batch_tr[dd]=tr
                                    evid_per_batch_tg[dd]=tg
                                    evid_per_batch_all[dd]=train






                                def reverse_accuracy(m,tr11,tg11):
                                            stacc=[]
                                            for zx in range(0,10):
                                                    test=[]
                                                    trg=[]
                                                    corpus_train, corpus_test, y_train, y_test = train_test_split(tr11,tg11, test_size=0.5, random_state=1)
                                                    ltr=math.ceil(len(tr11)*0.5)
                                                    for vb in range(0,ltr):
                                                        test.append(tr11[vb])
                                                    for vb1 in range(0,ltr):
                                                        trg.append(tg11[vb1])
                                                    vectorizer = TfidfVectorizer(min_df=1)
                                                    X_train = vectorizer.fit_transform(corpus_train)
                                                    X_test = vectorizer.transform(corpus_test)
                                                    model =model1=RandomForestClassifier(max_depth=2, random_state=1)
                                                    #svm.LinearSVC(C=10) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                                    model.fit(X_train,y_train)
                                                    p = model.predict(X_test)
                                                    acc=f1_score(y_test, p, average='micro')
                                                    stacc.append(float(acc))
                                            return stacc
                                for vc2 in evid_per_batch_tg:
                                        stacc=reverse_accuracy(vc2,evid_per_batch_tr[vc2],evid_per_batch_tg[vc2])
                                        pred_per_patch[vc2]=stacc
                                       # print(vc,evid_per_batch_tr[vc],evid_per_batch_tg[vc],len(evid_per_batch_tr[vc]),len(evid_per_batch_tg[vc]))
                                       # print("\n\n")
                                for vzz in pred_per_patch:
                                    pass#print(vzz,pred_per_patch[vzz])


                                vvv=[]
                                v1=[]
                                vz=[]
                                c=0
                                for vzz in pred_per_patch:
                                    vvv.append(pred_per_patch[vzz][0])
                                    if c<1:
                                        v1=pred_per_patch[vzz]
                                        c=c+1
                                    m=max(pred_per_patch[vzz])
                                    vz.append(m)
                                print(stdev(vvv))
                                print(statistics.mean(v1))
                                print(max(vz))
                
#stat_ana.stat_ana(spps)
if __name__ == "__main__":
    print("Statistical Analysis")



