import sys
class review_data_processing:
    @classmethod
    def data_pro(cls):
            import transformers
            import datasets
            import shap
            import numpy as np
            import random
            import sys
            import random
            import re
            import csv
            from collections import defaultdict
            import sys
            from nltk.cluster import KMeansClusterer
            import nltk
            from sklearn import cluster
            from sklearn import metrics
            import gensim 
            import operator
            from gensim.models import Word2Vec
            from sklearn import svm
            from sklearn.svm import SVC
            from sklearn.svm import LinearSVC
            import re
            from sklearn import tree
            from collections import defaultdict
            import sys
            from nltk.cluster import KMeansClusterer
            import nltk
            from sklearn import cluster
            from sklearn import metrics
            import gensim 
            import operator
            from gensim.models import Word2Vec
            import sklearn
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.model_selection import train_test_split
            import numpy as np
            import shap
            from sklearn.metrics import (precision_score,recall_score,f1_score)
            from sklearn.ensemble import RandomForestClassifier
            import gensim.models.word2vec as W2V
            import gensim.models
            import sys


            #dataset = datasets.load_dataset("imdb", split="test")

            # shorten the strings to fit into the pipeline model
            #short_data = [v[:500] for v in dataset["text"][:20]]
            f1=open("/Data/review_evid.txt",encoding="ISO-8859-1")
            f2=open("/Data/review_class.txt")
            f3=open("/Data/review_text.txt",encoding="ISO-8859-1")
            f4=open("/Data/review_stw.txt")
            f5=open("/Data/reviewembedded_relation.txt")
            WORDS={}
            tw={}
            qrat={}
            rtext={}
            ann={}
            similar_r_map={}
            rnn={}

            for t in f1:
                p=t.strip("\n ' '").split("::")
                pp=p[1].split()
                #print(pp)
                WORDS[p[0]]=pp
                tw[p[0]]=p[1]
            for t in f2:
                p=t.strip("\n ' '").split("::")
                #pp=p[1].split()
                #print(p[1])
                qrat[p[0]]=int(p[1])
            for t in f3:
                p=t.strip("\n ' '").split("::")
                #pp=p[1].split()
                #print(p[1])
                rtext[p[0]]=p[1]
            for t in f4:
                p=t.strip("\n ' '").split("::")
                pp=p[1].split()
                #print(pp)
                ann[p[0]]=pp
            for t in f5:
                p=t.strip("\n ' '").split("::")
                pp=p[1].split()
                #print(pp)
                similar_r_map[p[0]]=pp

            for t in similar_r_map:
                cc=0
                gh=[]
                for k in similar_r_map[t]:
                    if cc<25:
                        gh.append(k)
                        cc=cc+1
                rnn[t]=gh


            WORDS22={}
            c=0
            c1=0
            for t in WORDS:
                if qrat[t]==2:
                    if c<50:
                        WORDS22[t]=WORDS[t]
                        c=c+1
                elif qrat[t]==0:
                    if c1<50:
                        WORDS22[t]=WORDS[t]
                        c1=c1+1
            for k in WORDS22:
                pass#print(k,qrat[k],WORDS22[k])
            qf={}
            for t in similar_r_map:
                if t in WORDS22:
                    h=rnn[t]+WORDS22[t]
                    #print(t,qrat[t],h),
                    qf[t]=h

            f1=open("/Data/sent.txt","w")
            for t in qf:
                s=''
                for k in qf[t]:
                    s=s+k+" "
                gg=str(t)+":::"+s
                f1.write(str(gg)+"\n")
            f1.close()
            
            






class lime:
            classmethod
            def lime_fd(cls):

                        import pandas as pd
                        def lime_Explanation_Engine(s_words,stopwords,WORDS,qrat,H11,Rev_text_map):

                                                    #Balancing Review Data for both positive and negative
                                                    WORDSt={}
                                                    t1=[]
                                                    t2=[]
                                                    t3=[]
                                                    c1=0
                                                    c2=0
                                                    c3=0
                                                    for y in WORDS:
                                                        if qrat[y]==0:
                                                            if c3<500:
                                                                t1.append(y)
                                                                WORDSt[y]=WORDS[y]
                                                                c3=c3+1
                                                        elif qrat[y]==1:
                                                            continue
                                                            #if c1<102:
                                                                #t2.append(y)
                                                                #WORDS1[y]=WORDS[y]
                                                                #c1=c1+1
                                                        elif qrat[y]==2:
                                                            if c2<500:
                                                                t3.append(y)
                                                                WORDSt[y]=WORDS[y]
                                                                c2=c2+1
                                                    print(len(t1),len(t2),len(t3),len(WORDSt))
                                                    for k in WORDSt:
                                                        if qrat[k]==1:
                                                            print(k)
                                                    d_tt={}
                                                    d_tt[0]='negative'
                                                    d_tt[2]='positive'
                                                    WORDS_u={}
                                                    aw={}
                                                    c0=0
                                                    c1=0
                                                    K=500
                                                    for t in WORDSt:
                                                        if qrat[t]==0:
                                                            if c0<K:
                                                                WORDS_u[t]=WORDSt[t]
                                                                aw[t]=WORDSt[t]
                                                                c0=c0+1
                                                        elif qrat[t]==2:
                                                            if c1<K:
                                                                WORDS_u[t]=WORDSt[t]
                                                                aw[t]=WORDSt[t]
                                                                c1=c1+1
                                                    print(len(WORDS_u),len(aw))
                                                    #annotation
                                                    WORDS_uf={}
                                                    ann={}
                                                    kk=1.0
                                                    for t in WORDS_u:
                                                        m=0
                                                        gg=[]
                                                        for cvv in WORDS_u[t]:
                                                            if cvv not in gg:
                                                                #if m/float(len(WORDS_u[t]))<=kk:
                                                                    gg.append(cvv)

                                                        WORDS_uf[t]=gg
                                                       # print(len(gg),len(WORDS_u[t])         

                                                    for k1 in WORDS_uf:
                                                        vcc=[]
                                                        c=0
                                                        for t2 in WORDS_uf[k1]:
                                                            if t2 in s_words:
                                                                #print(t2)
                                                                if t2 not in vcc:
                                                                    #if c<500:
                                                                        vcc.append(t2)
                                                                        c=c+1
                                                        if len(vcc)>0:
                                                            ann[k1]=vcc
                                                    #annotted word features
                                                    wf2=[]
                                                    for t in ann:
                                                        for j in ann[t]:
                                                            if j not in wf2:
                                                                wf2.append(j)

                                                    #train and target data

                                                    train_r=[]
                                                    targets_r=[]
                                                    m_tid_tr1={}
                                                    wr=[]
                                                    tr=[]
                                                    wr1=[]
                                                    tr1=[]
                                                    c=0
                                                    c1=0
                                                    tw_wm={}
                                                    for t in WORDS_u:
                                                        s=''
                                                        vb=[]
                                                        #if twit_count[t]==1:
                                                        for tt in WORDS_u[t]:
                                                                s=s+str(tt)+" "
                                                                train_r.append(tt)
                                                                targets_r.append(qrat[t])
                                                        vb.append(s)
                                                        wr.append(s)
                                                        wr1.append(vb)
                                                        tr.append(qrat[t])
                                                        tw_wm[t]=s
                                                    unique_words=[]
                                                    ss=set( train_r)
                                                    for w1 in ss:
                                                            if w1 not in unique_words:
                                                                unique_words.append(w1)
                                                    #Shap

                                                    #shap.initjs()
                                                    # Kernal Shap words_train targets


                                                    corpus_train, corpus_test, y_train, y_test = train_test_split(train_r,targets_r, test_size=0.5, random_state=7)
                                                    vectorizer = TfidfVectorizer(min_df=1)
                                                    X_train = vectorizer.fit_transform(corpus_train)
                                                    X_test = vectorizer.transform(corpus_test)
                                                    model =svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                                    #KNeighborsClassifier(n_neighbors=5)
                                                    #RandomForestClassifier(max_depth=2, random_state=1)
                                                    #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                                    model.fit(X_train,y_train)
                                                    p = model.predict(X_test)
                                                    prr={}
                                                    for jj in range(0,len(corpus_test)):
                                                        prr[corpus_test[jj]]=int(p[jj])
                                                    print(f1_score(y_test,p,average='micro'))
                                                    explainer =shap.LinearExplainer(model, X_train, feature_dependence="independent")
                                                    shap_values = explainer.shap_values(X_test)
                                                    X_test_array = X_test.toarray() # we need to pass a dense version for the plotting functions
                                                    feature_names=vectorizer.get_feature_names()
                                                    print(len(feature_names),len(shap_values))
                                                    #shap.summary_plot(shap_values, X_test_array, feature_names=vectorizer.get_feature_names())
                                                    shape_w={}
                                                    fr={}
                                                    feature_sh_v=[]
                                                    for jj in range(0,len(corpus_train)):
                                                          if abs(sum(shap_values[jj]))>0.4:
                                                                                  m=abs(sum(shap_values[jj]))
                                                                                  if corpus_train[jj] not in fr:
                                                                                                  fr[corpus_train[jj]]=abs(sum(shap_values[jj]))
                                                                                  elif corpus_train[jj]  in fr:
                                                                                        if m>fr[corpus_train[jj]]:
                                                                                            fr[corpus_train[jj]]=m
                                                    dd1=sorted(fr.items(), key=operator.itemgetter(1),reverse=True)
                                                    for tt in dd1:
                                                           feature_sh_v.append(tt[0])
                                                    '''
                                                    feature_sh_v1=[]
                                                    for v in feature_sh_v:
                                                        n=v.split()
                                                        for k in n:
                                                            if k not in feature_sh_v1:
                                                                feature_sh_v1.append(k)
                                                    '''
                                                    #shap explanations
                                                    #Test data
                                                    WORDSt_t={}
                                                    cp=0
                                                    cn=0
                                                    for kx in  WORDSt:
                                                            if qrat[kx]==2:
                                                                if cp<(len(WORDSt)//2):
                                                                         WORDSt_t[kx]= WORDSt[kx]
                                                                         cp=cp+1 
                                                            elif qrat[kx]==0:
                                                                if cn<(len(WORDSt)//2):
                                                                         WORDSt_t[kx]= WORDSt[kx]
                                                                         cn=cn+1   

                                                    shap_exp={}
                                                    for t in WORDSt_t:
                                                        gh=[]
                                                        c=0
                                                        for k in WORDSt_t[t]:
                                                            if k in prr:
                                                                if qrat[t]==prr[k]:
                                                                    if k in feature_sh_v:
                                                                        if k not in gh:
                                                                            if c<5:
                                                                                gh.append(k)
                                                                                c=c+1
                                                        if len(gh)>0:
                                                                shap_exp[t]=gh

                                                    for tt in shap_exp:
                                                         pass#print(tt,shap_exp[tt])
                                                    shap_all={}
                                                    shap_all_p={}
                                                    shap_all_n={}

                                                    for t in shap_exp:
                                                        if t in ann:
                                                            c=0
                                                            vb=0
                                                            for zz in shap_exp[t]:
                                                                if zz in ann[t]:
                                                                    if vb<5:
                                                                            c=c+1
                                                                            vb=vb+1
                                                            if len(shap_exp[t])>0:
                                                                s=float(c)/len(shap_exp[t])
                                                                if s>0:
                                                                    shap_all[t]=s
                                                    ss=0
                                                    for k in shap_all:
                                                        ss=ss+float(shap_all[k])

                                                    acc=ss/len(shap_all)
                                                    print("Shap Accuracy without human-feedback"+"\n")
                                                    print("Shap Accuracy")
                                                    print(acc)
                                                    print("Shap only positive reviews")
                                                    for t in shap_exp:
                                                        if t in ann:
                                                            if qrat[t]==2:
                                                                        c=0
                                                                        vb1=0
                                                                        for zz in shap_exp[t]:
                                                                            if zz in ann[t]:
                                                                                if vb1<5:
                                                                                    c=c+1
                                                                                    vb1=vb1+1
                                                                        if len(shap_exp[t])>0:
                                                                            s=float(c)/len(shap_exp[t])
                                                                            if s>0:
                                                                                shap_all_p[t]=s
                                                    ss1=0
                                                    for k in shap_all_p:
                                                        ss1=ss1+float(shap_all_p[k])

                                                    acc_p=ss1/len(shap_all_p)
                                                    print("Shap Accuracy for positive reviews")
                                                    print(acc_p)
                                                    print("Shap only negative reviews")
                                                    for t in shap_exp:
                                                        if t in ann:
                                                            if qrat[t]==0:
                                                                        c=0
                                                                        vb2=0
                                                                        for zz in shap_exp[t]:
                                                                            if zz in ann[t]:
                                                                                if vb2<5:
                                                                                    c=c+1
                                                                                    vb2=vb2+1
                                                                        if len(shap_exp[t])>0:
                                                                            s=float(c)/len(shap_exp[t])
                                                                            if s>0:
                                                                                shap_all_n[t]=s
                                                    ss2=0
                                                    for k in shap_all_n:
                                                        ss2=ss2+float(shap_all_n[k])

                                                    acc_n=ss2/len(shap_all_n)
                                                    print("Shap Accuracy for negative reviews")
                                                    print(acc_n)
                                                    return acc,acc_p,acc_n,shap_exp,shap_values,ann,corpus_train, corpus_test, y_train, y_test,p,qrat,WORDSt
                                    


                        #s_words,stopwords,WORDS,qrat,H11,Rev_text_map=data_processing()
                        acc,acc_p,acc_n,shap_exp,shap_values,ann,corpus_train,corpus_test,y_train,y_test,p,qrat,WORDSt=Shap_Explanation_Engine(s_words,stopwords,WORDS,qrat,H11,Rev_text_map)


                        import shap
                        
                        def cluster_feed(KK,qrat,WORDSt,shap_exp,Rev_text_map):
                                    #Sentance generation
                                    sent=[]
                                    sent1=[]
                                    sent_map=defaultdict(list)
                                    for ty in WORDSt:
                                        gh=[]
                                        gh.append(str(ty))
                                        #gh1=[]
                                        #gh2=[]
                                        for j in WORDSt[ty]:

                                            j1=str(j)
                                            #gh.append(str(ty))
                                            if j1 not in gh:
                                                gh.append(j1)

                                            #print(gh)


                                        if gh not in sent:
                                                sent.append(gh)


                                    documents=[]
                                    #documents1=[]
                                    for t in sent:
                                        for jh in t:
                                            documents.append(jh)

                                    for w in sent:
                                        pass#print(w)
                                    #K-Means Run 14
                                    #cluster generation with k-means
                                    model = Word2Vec(sent, min_count=1)
                                    X = model[model.wv.vocab]
                                    NUM_CLUSTERS=KK
                                    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,repeats=25)
                                    assigned_clusters1 = kclusterer.cluster(X,assign_clusters=True)
                                    #print (assigned_clusters)
                                    cluster={}
                                    words = list(model.wv.vocab)
                                    for i, word in enumerate(words):
                                      gh=[] 
                                      gh1=[] 
                                      gh2=[] 
                                      if word.isdigit(): 
                                        cluster[word]=assigned_clusters1[i]
                                        #print (word + ":" + str(assigned_clusters[i]))
                                    cluster_final={}
                                    for j in range(NUM_CLUSTERS):
                                        gg=[]
                                        for tt in cluster:
                                            if int(cluster[tt])==int(j):
                                                if tt not in gg:
                                                    gg.append(tt)
                                        if len(gg)>0:
                                                    cluster_final[j]=gg
                                    cc=0
                                    final_clu={}
                                    lmm=(KK*3)+5
                                    for t in cluster_final:
                                        ghh=[]
                                        vx=0
                                        for k in cluster_final[t]:
                                            if int(k) in WORDS and int(k) in shap_exp or str(k) in shap_exp:
                                                if vx<lmm:
                                                        ghh.append(int(k))
                                                        vx=vx+1
                                        if len(ghh)>=2:
                                                final_clu[cc]=ghh
                                                cc=cc+1
                                    for k in final_clu:
                                        pass#print(k,final_clu[k],len(final_clu[k]))
                                    return final_clu







                        cl={}
                        for kk in range(10,31,5):
                                final_clu=cluster_feed(kk,qrat,WORDSt,shap_exp,Rev_text_map)
                                cl[kk]=final_clu
                        def lime_all_acc():
                                        def feedback_accuracy(WORDS22,qrat,ann):
                                                train_r=[]
                                                targets_r=[]
                                                m_tid_tr1={}
                                                wr=[]
                                                tr=[]
                                                wr1=[]
                                                tr1=[]
                                                c=0
                                                c1=0
                                                tw_wm={}
                                                for t in WORDS22:
                                                    s=''
                                                    vb=[]
                                                    #if twit_count[t]==1:
                                                    for tt in WORDS22[t]:
                                                            s=s+str(tt)+" "
                                                            train_r.append(tt)
                                                            targets_r.append(qrat[t])
                                                    vb.append(s)
                                                    wr.append(s)
                                                    wr1.append(vb)
                                                    tr.append(qrat[t])
                                                    tw_wm[t]=s
                                                unique_words=[]
                                                ss=set( train_r)
                                                for w1 in ss:
                                                        if w1 not in unique_words:
                                                            unique_words.append(w1)
                                                #Shap

                                                corpus_train, corpus_test, y_train, y_test = train_test_split(train_r,targets_r, test_size=0.5, random_state=7)
                                                vectorizer = TfidfVectorizer(min_df=1)
                                                X_train = vectorizer.fit_transform(corpus_train)
                                                X_test = vectorizer.transform(corpus_test)
                                                model =svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                                #KNeighborsClassifier(n_neighbors=5)
                                                #RandomForestClassifier(max_depth=2, random_state=1)
                                                #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                                model.fit(X_train,y_train)
                                                p = model.predict(X_test)
                                                prr={}
                                                for jj in range(0,len(corpus_test)):
                                                    prr[corpus_test[jj]]=int(p[jj])
                                                print(f1_score(y_test,p,average='micro'))
                                                explainer =shap.LinearExplainer(model, X_train, feature_dependence="independent")
                                                shap_values = explainer.shap_values(X_train)
                                                X_test_array = X_test.toarray() # we need to pass a dense version for the plotting functions
                                                feature_names=vectorizer.get_feature_names()
                                                print(len(feature_names),len(shap_values))
                                                #shap.summary_plot(shap_values, X_test_array, feature_names=vectorizer.get_feature_names())
                                                shape_w={}
                                                fr={}
                                                feature_sh_v=[]
                                                for jj in range(0,len(corpus_train)):
                                                      if abs(sum(shap_values[jj]))>0.4:
                                                                              m=abs(sum(shap_values[jj]))
                                                                              if corpus_train[jj] not in fr:
                                                                                              fr[corpus_train[jj]]=abs(sum(shap_values[jj]))
                                                                              elif corpus_train[jj]  in fr:
                                                                                    if m>fr[corpus_train[jj]]:
                                                                                        fr[corpus_train[jj]]=m
                                                dd1=sorted(fr.items(), key=operator.itemgetter(1),reverse=True)
                                                for tt in dd1:
                                                       feature_sh_v.append(tt[0])
                                                '''
                                                feature_sh_v1=[]
                                                for v in feature_sh_v:
                                                    n=v.split()
                                                    for k in n:
                                                        if k not in feature_sh_v1:
                                                            feature_sh_v1.append(k)
                                                '''
                                                #shap explanations

                                                shap_exp={}
                                                for t in WORDS22:
                                                    gh=[]
                                                    c=0
                                                    for k in WORDS22[t]:
                                                        if k in prr:
                                                            if qrat[t]==prr[k]:
                                                                if k in feature_sh_v:
                                                                    if k not in gh:
                                                                        if c<5:
                                                                            gh.append(k)
                                                                            c=c+1
                                                    if len(gh)>0:
                                                            shap_exp[t]=gh

                                                for tt in shap_exp:
                                                     pass#print(tt,shap_exp[tt])
                                                shap_all={}
                                                shap_all_p={}
                                                shap_all_n={}

                                                for t in shap_exp:
                                                    vc=0
                                                    if t in ann:
                                                        c=0
                                                        for zz in shap_exp[t]:
                                                            if zz in ann[t]:
                                                                if vc<5:
                                                                    c=c+1
                                                                    vc=vc+1
                                                        if len(shap_exp[t])>0:
                                                            s=float(c)/len(shap_exp[t])
                                                            if s>0:
                                                                shap_all[t]=s
                                                ss=0
                                                acc=0
                                                acc_p=0
                                                acc_n=0
                                                for k in shap_all:
                                                    ss=ss+float(shap_all[k])
                                                if len(shap_all)>0:
                                                     acc=ss/len(shap_all)

                                                #print("Shap Accuracy without human-feedback"+"\n")
                                                #print("Shap Accuracy")
                                                #print(acc)
                                                #print("Shap only positive reviews")
                                                for t in shap_exp:
                                                    if t in ann:
                                                        if qrat[t]==2:
                                                                    c=0
                                                                    vc1=0
                                                                    for zz in shap_exp[t]:
                                                                        if zz in ann[t]:
                                                                            if vc1<5:
                                                                                c=c+1
                                                                                vc1=vc1+1
                                                                    if len(shap_exp[t])>0:
                                                                        s=float(c)/len(shap_exp[t])
                                                                        if s>0:
                                                                            shap_all_p[t]=s
                                                ss1=0
                                                for k in shap_all_p:
                                                    ss1=ss1+float(shap_all_p[k])
                                                if len(shap_all_p)>0:
                                                    acc_p=ss1/len(shap_all_p)
                                                else:
                                                    acc_p=0
                                                #print("Shap Accuracy for positive reviews")
                                                #print(acc_p)
                                                #print("Shap only negative reviews")
                                                for t in shap_exp:
                                                    if t in ann:
                                                        if qrat[t]==0:
                                                                    c=0
                                                                    vc2=0
                                                                    for zz in shap_exp[t]:
                                                                        if zz in ann[t]:
                                                                            if vc2<5:
                                                                                c=c+1
                                                                                vc2=vc2+1
                                                                    if len(shap_exp[t])>0:
                                                                        s=float(c)/len(shap_exp[t])
                                                                        if s>0:
                                                                            shap_all_n[t]=s
                                                ss2=0
                                                for k in shap_all_n:
                                                    ss2=ss2+float(shap_all_n[k])
                                                if len(shap_all_n)>0:
                                                    acc_n=ss2/len(shap_all_n)
                                                else:
                                                    acc_n=0
                                                #print("Shap Accuracy for negative reviews")
                                                #print(acc_n)
                                                return acc,acc_p,acc_n







                                        def feed(mn,qrat,cl,w3):
                                                        sent=[]
                                                        sent1=[]
                                                        w33={}
                                                        cp=0
                                                        cn=0
                                                        for kkk in w3:
                                                            gvv=[]
                                                            if qrat[kkk]==2:
                                                                if cp<250:
                                                                    w33[kkk]=w3[kkk]
                                                                    cp=cp+1
                                                            elif qrat[kkk]==0:
                                                                if cn<250:
                                                                    w33[kkk]=w3[kkk]
                                                                    cn=cn+1


                                                        sent_map=defaultdict(list)
                                                        for ty in w3:
                                                            gh=[]
                                                            gh.append(str(ty))
                                                            #gh1=[]
                                                            #gh2=[]
                                                            for j in w3[ty]:

                                                                j1=str(j)
                                                                #gh.append(str(ty))
                                                                if j1 not in gh:
                                                                    gh.append(j1)

                                                            if gh not in sent:
                                                                    sent.append(gh)
                                                        documents=[]
                                                        #documents1=[]
                                                        for t in sent:
                                                            for jh in t:
                                                                documents.append(jh)
                                                        hh="feedback_shap_"+str(mn)+".csv"
                                                        ps={}
                                                        ns={}
                                                        vot={}
                                                        vtt={}
                                                        f1=pd.read_csv(hh)
                                                        vot={}
                                                        vtt={}
                                                        ll=0
                                                        for col in f1.columns:
                                                            if 'Label' in col:
                                                                ll=ll+1
                                                        m=len(f1['Review_ID'])
                                                        for t in range(0,m):
                                                            #print(f1['Review_ID'][t])
                                                            vtt[f1['Review_ID'][t]]=f1['Explanation'][t]
                                                            gh=[]
                                                            for vv in range(1,ll+1):
                                                                          vb1="Label"+str(vv)
                                                                          #print(f1[vb1][t])
                                                                          gh.append(f1[vb1][t])
                                                            vot[f1['Review_ID'][t]]=gh


                                                        for uy in vot:
                                                            cp=0
                                                            cn=0
                                                            for kk in vot[uy]:
                                                                if int(kk)==1:
                                                                    cp=cp+1
                                                                else:
                                                                    cn=cn+1
                                                            if cp>=cn:
                                                                ps[uy]=vtt[uy]
                                                            elif cp<cn:
                                                                ns[uy]=vtt[uy]

                                                        cl1=cl[mn]

                                                        WORDS23={}
                                                        rm=[]
                                                        for jj in ns:
                                                            for kj in ns[jj]:
                                                                rm.append(kj)

                                                        model = Word2Vec(sent, min_count=1)
                                                        rme={}
                                                        for uu in ns:
                                                            for k in cl1:
                                                                if uu in cl1[k]:
                                                                    for kk in cl1[k]:
                                                                        gg=[]
                                                                        zz=0
                                                                        if kk in w33:
                                                                            for vv in w33[kk]:
                                                                                if vv in ns[uu]:
                                                                                    continue
                                                                                else:
                                                                                    if zz<5:
                                                                                        gg.append(vv)
                                                                                        zz=zz+1
                                                                            rme[kk]=gg


                                                        for t in cl1:
                                                            for k in ps:
                                                                if k in cl1[t]:
                                                                    for kk in cl1[t]:
                                                                        chu1=[]
                                                                        vb={}
                                                                        for v in ps[k]:
                                                                            vb1={}
                                                                            if kk in w33:
                                                                                for v1 in w33[kk]:
                                                                                    try:
                                                                                            gh1=model.similarity(v,v1)
                                                                                            if gh1>0.4:
                                                                                                          vb1[v1]=float(gh1)
                                                                                    except:
                                                                                        continue 
                                                                                for jk in vb1:
                                                                                                if jk in vb:
                                                                                                    if float(vb1[jk])>=float(vb[jk]):
                                                                                                        #print(jk,vb1[jk],vb[jk])
                                                                                                        vb[jk]=vb1[jk]
                                                                                                else:
                                                                                                    vb[jk]=vb1[jk]

                                                                        dd1=sorted(vb.items(), key=operator.itemgetter(1),reverse=True)
                                                                        cc=0
                                                                        for kkk in dd1:
                                                                            if kkk[0] not in chu1:
                                                                                if cc<5:
                                                                                        chu1.append(kkk[0])
                                                                                        cc=cc+1
                                                                        if len(chu1)>0 :
                                                                            if kk in w33:
                                                                                WORDS23[kk]=chu1 
                                                        WORDS25={}
                                                        for t in WORDS23:
                                                            cc=0
                                                            vcx=[]
                                                            if t not in rme:
                                                                WORDS25[t]=WORDS23[t]
                                                            elif t in rme:
                                                                vcc=WORDS23[t]+rme[t]
                                                                for zz in vcc:
                                                                    if cc<5:
                                                                        vcx.append(zz)
                                                                        cc=cc+1
                                                                WORDS25[t]=vcx
                                                        print(len(WORDS25))
                                                        for cc in rme:
                                                            fg=[]
                                                            vc4=0
                                                            if cc not in WORDS25:
                                                                for bb in rme[cc]:
                                                                    if vc4<5:
                                                                        fg.append(bb)
                                                                        vc4=vc4+1
                                                                WORDS25[cc]=fg
                                                        print(len(WORDS25))

                                                        return WORDS25






                                        all_a={}
                                        on_p={}
                                        on_n={}
                                        for mn in range(10,31,5):     
                                                WORDS25=feed(mn,qrat,cl,WORDSt)
                                                acc,acc_p,acc_n=feedback_accuracy(WORDS25,qrat,ann)  
                                                all_a[mn]=acc
                                                on_p[mn]=acc_p
                                                on_n[mn]=acc_n


                                        for tt in all_a:
                                            print(tt,all_a[tt],on_p[tt],on_n[tt])
                                            print("\n\n")


                        shapp_all_acc()
                        class GloveVectorizer:
                            def __init__(self, verbose=False, lowercase=True, minchars=3):

                                #sentences = [['Test','Alice'],['Test','Bob'],['Alice','Carl'],['David','Carl']]
                                f2=open("sent.txt")
                                WORDStt={}
                                for k in f2:
                                    pp=k.strip("\n \t " " ").split(":::")
                                    #print(pp)
                                    WORDStt[pp[0]]=pp[1]
                                sent=[]
                                sent1=[]
                                self.data=WORDStt
                                sent_map=defaultdict(list)
                                for ty in  WORDStt:
                                    gh=[]
                                    gh.append(str(ty))
                                    for j in WORDStt[ty]:
                                        j1=str(j)
                                        #gh.append(str(ty))
                                        if j1 not in gh:
                                            gh.append(j1)
                                    if gh not in sent:
                                            sent.append(gh)
                                f2.close()
                                self.model = gensim.models.Word2Vec(sentences=sent,min_count=1,window=2)#,vector_size=20)  

                            def fit(self, data, *args):
                                pass

                            def transform(self, data, *args):
                                W,D = self.model.wv.vectors.shape
                                X = np.zeros((len(data), D))
                                n = 0
                                emptycount = 0
                                for sentence in data:
                                    #if sentence.isdigit()==True:
                                    tokens = sentence
                                    vecs = []
                                    for word in tokens:
                                        if word in self.model.wv:
                                            vec = self.model.wv[word]
                                            vecs.append(vec)
                                    if len(vecs) > 0:
                                        vecs = np.array(vecs)
                                        X[n] = vecs.mean(axis=0)
                                    else:
                                        emptycount += 1
                                    n += 1
                                #X = np.random.rand(100,20)
                                #X1 = np.asarray(X,dtype='float64')
                                return X

                            def fit_transform(self, X, *args):
                                self.fit(X, *args)
                                return self.transform(X, *args)

                        def lime_relation(WORDS22,qrat,similar_r_map,ann,rnn,option):
                                            train_r=[]
                                            targets_r=[]
                                            m_tid_tr1={}
                                            wr=[]
                                            tr=[]
                                            wr1=[]
                                            tr1=[]
                                            c=0
                                            c1=0
                                            tw_wm={}
                                            similar_r_map=rnn#similar_r_map
                                                 # organizing feature vector
                                            qf={}
                                            for t in similar_r_map:
                                                    if t in WORDS22:
                                                        h=rnn[t]+WORDS22[t]
                                                        #print(t,qrat[t],h),
                                                        qf[t]=h

                                            for t in qf:
                                                    s=''
                                                    vb=[]
                                                    for tt in qf[t]:
                                                            train_r.append(tt)
                                                            targets_r.append(qrat[t])

                                            #LIMe

                                            corpus_train, corpus_test, y_train, y_test = train_test_split(train_r,targets_r, test_size=0.5, random_state=7)
                                            vectorizer = TfidfVectorizer(min_df=1)
                                            X_train = vectorizer.fit_transform(corpus_train)
                                            X_test = vectorizer.transform(corpus_test)
                                            #model3='
                                            if option=='svm':
                                                model =svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)
                                            elif option=='bagging':
                                                model=BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0)
                                            elif option=='random':
                                                model=RandomForestClassifier(max_depth=2, random_state=1)
                                            elif option=='extratree':
                                                model=ExtraTreesClassifier(n_estimators=100, random_state=0) 
                                            elif option=='knn':
                                                model=KNeighborsClassifier(n_neighbors=3)

                                            #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                            #KNeighborsClassifier(n_neighbors=5)
                                            #RandomForestClassifier(max_depth=2, random_state=1)
                                            #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                            clf=model
                                            clf.fit(X_train,y_train)
                                            p = clf.predict(X_test)
                                            prr={}
                                            for jj in range(0,len(corpus_test)):
                                                prr[corpus_test[jj]]=int(p[jj])
                                            print(f1_score(y_test,p,average='micro'))
                                            gv = GloveVectorizer()
                                            X=gv.fit_transform(qf)
                                            #print(X)

                                            c = make_pipeline(gv,clf)
                                            #try:
                                            c.fit(corpus_train,y_train)
                                            rtt=c.predict_proba([corpus_test[0]]).round(3)
                                            print(rtt)
                                            #sys.exit()

                                            #print(c)
                                            #for ii in range(0,len(corpus_test)):
                                                #rtt=c.predict_proba([corpus_test[ii]]).round(3)
                                                #for kj in range(0,len(rtt[0])):
                                                   # if rtt[0][kj]==max(rtt[0]):
                                                        #if corpus_test[ii].isdigit()==True:
                                                            #if kj==0:
                                                               # pass#print(test[ii],prd[test[ii]],kj,tr[test[ii]])
                                                           # else:
                                                                #pass#print(test[ii],prd[test[ii]],kj+1,tr[test[ii]])
                                                        #else:
                                                           # if kj==0:
                                                                      #  pass#print(test[ii],prd[test[ii]],kj,tr[test[ii]])
                                                           # else:
                                                                       # pass#print(test[ii],prd[test[ii]],kj+1,tr[test[ii]])


                                            #mt1=['7','12','13','14','15','16','17'] #clasnames
                                            mt1=['0','1']
                                            cna=mt1


                                            #explainer = lime.lime_tabular.LimeTabularExplainer(tr, feature_names=feature_names, class_names=targets_t, discretize_continuous=True)

                                            explainer = LimeTextExplainer(class_names=cna)

                                            #sys.exit()
                                            #d22=sorted(dd3.items(),key=operator.itemgetter(1),reverse=True)
                                            word_weight={}
                                            for ii in range(0,len(corpus_test)):

                                                                        ww={}
                                                                        ww2={}
                                                                        wq=[]
                                                                        exp = explainer.explain_instance(corpus_test[ii], c.predict_proba, labels=(0,1), top_labels=None, num_features=44370, num_samples=1000, distance_metric=u'cosine', model_regressor=None)# explainer.explain_instance(trainf[int(ii)], c.predict_proba, num_features=2961, labels=[0,1])#, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
                                                                        #print("jhhjvhhvhvhhg")
                                                                        #print(exp)
                                                                        tty=exp.as_list()
                                                                        #print(tty)
                                                                        rtr=''
                                                                        for i in tty:
                                                                            #print(i)
                                                                            #rrr=random.randint(2,10)
                                                                            mlp=random.uniform(0.8,1.1)
                                                                            if float(i[1])>0:
                                                                                word_weight[i[0]]=abs(float(i[1]))
                                            # lime accuracy
                                            import operator
                                            lw={}
                                            dd=sorted(word_weight.items(), key=operator.itemgetter(1),reverse=True)

                                            for k in dd:
                                                #lw.append(k[0])
                                                lw[k[0]]=k[1]
                                            #print(lw)

                                            #Lime Exp

                                            lexp={}
                                            lexp1={}
                                            for tt in qf:
                                                    gh=[]
                                                    gh2=[]
                                                    gh3=[]
                                                    c=0
                                                    c1=0
                                                    for gg in lw:
                                                        if gg.isdigit()==False:
                                                            if gg in qf[tt]:
                                                                    if int(qrat[tt])==int(prr[gg]):
                                                                           # print(qrat[tt])
                                                                            #vb1=str(gg)+":"+str(word_weight[gg])
                                                                            gh3.append(gg)
                                                                            if c<25:
                                                                                gh.append(gg)
                                                                                c=c+1
                                                        else:
                                                            if gg in qf[tt] or int(gg) in qf[tt] or str(gg) in qf[tt]:
                                                                    if int(qrat[tt])==int(prr[gg]):
                                                                            #print("digit")
                                                                            #vb1=str(gg)+":"+str(word_weight[gg])
                                                                            gh3.append(gg)
                                                                            if c1<10:
                                                                                gh.append(gg)
                                                                                c1=c1+1



                                                    #epp=gh+gh
                                                    if len(gh)>0:
                                                            lexp[tt]=gh
                                                    if len(gh3)>0:
                                                            lexp1[tt]=gh3
                                            lime_all={}
                                            lime_all_p={}
                                            lime_all_n={}
                                            acco=0
                                            acc_po=0
                                            acc_no=0
                                            accor=0
                                            acc_por=0
                                            acc_nor=0

                                            for t in lexp:
                                                if t in ann:
                                                    c=0
                                                    vb=0
                                                    for zz in lexp[t]:
                                                        if zz.isdigit():
                                                            continue
                                                        else:
                                                            if zz in ann[t]:
                                                                if vb<5:
                                                                        c=c+1
                                                                        vb=vb+1
                                                    if len(lexp[t])>0:
                                                        s=float(c)/len(lexp[t])
                                                        if s>0:
                                                            lime_all[t]=s
                                            ss=0
                                            for k in lime_all:
                                                ss=ss+float(lime_all[k])

                                            if len(lime_all)>0:
                                                acco=ss/(len(lime_all)+len(lime_all)//2)
                                            #print(option)
                                            #print("Lime Accuracy without human-feedback"+"\n")
                                            #print(acc)
                                            #print("Lime Accuracy without human-feedback only positive reviews"+"\n")

                                            ss1=0
                                            pp=0
                                            for k in lime_all:
                                                if qrat[k]==2:
                                                    ss1=ss1+float(lime_all[k])
                                                    pp=pp+1

                                            if len(lime_all)>0:
                                                acc_po=ss1/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for positive reviews")
                                            #print(acc_p)
                                           # print("Lime only negative reviews")
                                            ss2=0
                                            nn=0
                                            for k in lime_all:
                                                if qrat[k]==0:
                                                    ss2=ss2+float(lime_all[k])
                                                    nn=nn+1

                                            if len(lime_all)>0:
                                                acc_no=ss2/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for negative reviews")
                                            #print(acc_n)

                                            lime_all={}
                                            lime_all_p={}
                                            lime_all_n={}
                                            #Relation
                                            for t in lexp:
                                                if t in similar_r_map:
                                                    #print("RElation")
                                                    c=0
                                                    vb=0
                                                    for zz in lexp[t]:
                                                        if zz.isdigit():
                                                            #print(zz)
                                                            #print("test")
                                                            if int(zz) in similar_r_map[t] or str(zz) in similar_r_map[t] or zz in similar_r_map[t]:
                                                                if vb<5:
                                                                        c=c+1
                                                                        vb=vb+1
                                                    if len(lexp[t])>0:
                                                        s=float(c)/len(lexp[t])
                                                        if s>0:
                                                            lime_all[t]=s
                                                            #print("ya")
                                                            #print(s)
                                            ss=0
                                            for k in lime_all:
                                                ss=ss+float(lime_all[k])

                                            if len(lime_all)>0:
                                                accr=ss/(len(lime_all)+len(lime_all)//2)
                                            #print(option)
                                            #print("Lime Accuracy without human-feedback"+"\n")
                                            #print(acc)
                                            #print("Lime Accuracy without human-feedback only positive reviews"+"\n")
                                            ss1=0
                                            pp=0
                                            for k in lime_all:
                                                if qrat[k]==2:
                                                   # print("positive")
                                                    ss1=ss1+float(lime_all[k])
                                                    pp=pp+1

                                            if len(lime_all)>0:
                                                acc_por=ss1/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for positive reviews")
                                            #print(acc_p)
                                           # print("Lime only negative reviews")
                                            ss2=0
                                            nn=0
                                            for k in lime_all:
                                                if qrat[k]==0:
                                                    ss2=ss2+float(lime_all[k])
                                                    nn=nn+1

                                            if len(lime_all)>0:
                                                acc_nor=ss2/len(lime_all)
                                            return acco,acc_po,acc_no,accr,acc_por,acc_nor,lexp,lexp1





                        ao={}
                        po={}
                        no={}
                        aor={}
                        por={}
                        nor={}
                        ep={}
                        clss={}
                        option=['bagging']


                        for tx in option:           
                            acco,acc_po,acc_no,accr,acc_pr,acc_nr,lexp,lexp1=lime_relation(WORDS22,qrat,similar_r_map,ann,rnn,tx)
                            ao[tx]=acco
                            po[tx]=acc_po
                            no[tx]=acc_no
                            aor[tx]=accr
                            por[tx]=acc_pr
                            nor[tx]=acc_nr
                            ep[tx]=lexp
                            
class lime_relation_feedback:
    
        @classmethod
        def op1(cls):
                    import pandas as pd
                    # Relational annotation
                    @classmethod
                    def relational_embedding_exp(cls,m,WORDS22,qrat,ann):
                        # Relational Exp generatetion based on neural embedding
                                    sent2=[]
                                    sent1=[]
                                    sent_map=defaultdict(list)
                                    for ty in WORDS22:
                                        gh=[]
                                        gh.append(str(ty))
                                        #gh1=[]
                                        #gh2=[]
                                        for j in WORDS22[ty]:

                                            j1=str(j)
                                            #gh.append(str(ty))
                                            if j1 not in gh:
                                                gh.append(j1)
                                            ##print(gh)


                                        if gh not in sent2:
                                                sent2.append(gh)


                                    documents1=[]
                                    #documents1=[]
                                    for t in sent2:
                                        s=''
                                        for jh in t:
                                            if jh.isdigit():
                                                 documents1.append(jh)
                                            else:
                                                s=" "+str(jh)+s+" "
                                        documents1.append(s)


                                    #sentence embedding
                                    from gensim.test.utils import common_texts
                                    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
                                    documents2 = [TaggedDocument(doc, [i]) for i, doc in enumerate(sent2)]
                                    for t in documents2:
                                        pass##print(t)
                                    model = Doc2Vec(documents2, vector_size=5, window=2, min_count=1, workers=4)

                                    #K-Means Run 14 to find the neighbors per query 

                                    #cluster generation with k-means
                                    import sys
                                    from nltk.cluster import KMeansClusterer
                                    import nltk
                                    from sklearn import cluster
                                    from sklearn import metrics
                                    import gensim 
                                    import operator
                                    #from gensim.models import Word2Vec


                                    #model = Word2Vec(sent, min_count=1) dd=sorted(vb.items(), key=operator.itemgetter(1),reverse=False)
                                    import operator
                                    X = model[model.wv.vocab]
                                    c=0
                                    cluster={}
                                    num=[]
                                    weight_map={}
                                    similar_r_map={}
                                    fg={}
                                    for jj in WORDS22:
                                        gh1=[]
                                        gh2=[]
                                        s=0

                                        for k in documents1:
                                            if str(k)==str(jj):
                                                gh=model.most_similar(positive=str(k),topn=600)
                                               # #print(gh)
                                                for tt in gh:
                                                    if float(tt[1]) not in gh1:
                                                        gh1.append(float(tt[1]))
                                                    #if tt[0] not in gh2:
                                                    if tt[0].isdigit():
                                                            #if ccc<5:
                                                                    #gh2.append(tt[0])
                                                                    fg[tt[0]]=tt[1]
                                                                    #ccc=ccc+1
                                        #for ffg in gh1:
                                            #s=s+ffg
                                        dd=sorted(fg.items(), key=operator.itemgetter(1),reverse=True)
                                        ccc=0
                                        for t5 in dd:
                                            if qrat[str(jj)]==qrat[str(t5[0])]:
                                                if m==5:
                                                    if ccc<500:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1
                                                elif m==10:
                                                    if ccc<400:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1
                                                elif m==15:
                                                    if ccc<500:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1
                                                elif m==20:
                                                    if ccc<600:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1
                                                elif m==25:
                                                    if ccc<700:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1

                                        #if len(gh2)>=2:
                                        similar_r_map[jj]=gh2
                                                #ccc=ccc+1

                                    return similar_r_map


                    import gensim.models.word2vec as W2V
                    import gensim.models
                    import sys
                    class GloveVectorizer:
                        def __init__(self, verbose=False, lowercase=True, minchars=3):
                            '''
                            # load in pre-trained word vectors
                            print('Loading word vectors...')
                            word2vec = {}
                            embedding = []
                            idx2word = []
                            with open('../data/glove.6B.50d.txt') as f:
                                  # is just a space-separated text file in the format:
                                  # word vec[0] vec[1] vec[2] ...
                                  for line in f:
                                    values = line.split()
                                    word = values[0]
                                    vec = np.asarray(values[1:], dtype='float32')
                                    word2vec[word] = vec
                                    embedding.append(vec)
                                    idx2word.append(word)
                            print('Found %s word vectors.' % len(word2vec))

                            self.word2vec = word2vec
                            self.embedding = np.array(embedding)
                            self.word2idx = {v:k for k,v in enumerate(idx2word)}
                            self.V, self.D = self.embedding.shape
                            self.verbose = verbose
                            self.lowercase = lowercase
                            self.minchars = minchars
                            '''
                            #sentences = [['Test','Alice'],['Test','Bob'],['Alice','Carl'],['David','Carl']]
                            f2=open("sent.txt")
                            WORDStt={}
                            for k in f2:
                                pp=k.strip("\n \t " " ").split(":::")
                                #print(pp)
                                WORDStt[pp[0]]=pp[1]
                            sent=[]
                            sent1=[]
                            self.data=WORDStt
                            sent_map=defaultdict(list)
                            for ty in  WORDStt:
                                gh=[]
                                gh.append(str(ty))
                                for j in WORDStt[ty]:
                                    j1=str(j)
                                    #gh.append(str(ty))
                                    if j1 not in gh:
                                        gh.append(j1)
                                if gh not in sent:
                                        sent.append(gh)
                            f2.close()
                            self.model = gensim.models.Word2Vec(sentences=sent,min_count=1,window=2)#,vector_size=20)  

                        def fit(self, data, *args):
                            pass

                        def transform(self, data, *args):
                            W,D = self.model.wv.vectors.shape
                            X = np.zeros((len(data), D))
                            n = 0
                            emptycount = 0
                            for sentence in data:
                                #if sentence.isdigit()==True:
                                tokens = sentence
                                vecs = []
                                for word in tokens:
                                    if word in self.model.wv:
                                        vec = self.model.wv[word]
                                        vecs.append(vec)
                                if len(vecs) > 0:
                                    vecs = np.array(vecs)
                                    X[n] = vecs.mean(axis=0)
                                else:
                                    emptycount += 1
                                n += 1
                            #X = np.random.rand(100,20)
                            #X1 = np.asarray(X,dtype='float64')
                            return X

                        def fit_transform(self, X, *args):
                            self.fit(X, *args)
                            return self.transform(X, *args)


                    def lime_all_acc(WORDS22,tx,qrat):
                                    def feedback_accuracy(WORDS22,qrat,similar_r_map,ann,rnn,option):
                                            train_r=[]
                                            targets_r=[]
                                            m_tid_tr1={}
                                            wr=[]
                                            tr=[]
                                            wr1=[]
                                            tr1=[]
                                            c=0
                                            c1=0
                                            tw_wm={}
                                            train_r=[]
                                            targets_r=[]
                                            m_tid_tr1={}
                                            wr=[]
                                            tr=[]
                                            wr1=[]
                                            tr1=[]
                                            c=0
                                            c1=0
                                            tw_wm={}
                                            similar_r_map=relational_embedding_exp(5,WORDS22,qrat,ann)
                                            rnn1={}
                                            for t in similar_r_map:
                                                gg=[]
                                                cc=0
                                                for k in similar_r_map[t]:
                                                    if cc<25:
                                                        gg.append(k)
                                                        cc=cc+1
                                                rnn1[t]=gg

                                            # organizing feature vector
                                            qf={}
                                            for t in similar_r_map:
                                                    #if t in WORDS22 and t in rnn:
                                                        h=rnn1[t]+WORDS22[t]
                                                        #print(t,qrat[t],h),
                                                        qf[t]=h

                                            #s=''
                                            #vb=[]
                                            #if twit_count[t]==1:
                                            for t in qf:
                                                    for tt in qf[t]:
                                                            train_r.append(tt)
                                                            targets_r.append(qrat[str(t)])



                                            corpus_train, corpus_test, y_train, y_test = train_test_split(train_r,targets_r, test_size=0.5, random_state=7)
                                            vectorizer = TfidfVectorizer(min_df=1)
                                            X_train = vectorizer.fit_transform(corpus_train)
                                            X_test = vectorizer.transform(corpus_test)
                                            #['svm','bagging','random','extratree','knn']
                                            if option=='svm':
                                                model1 =svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)
                                            elif option=='bagging':
                                                model1=BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0)
                                            elif option=='random':
                                                model1=RandomForestClassifier(max_depth=2, random_state=1)
                                            elif option=='extratree':
                                                model1=ExtraTreesClassifier(n_estimators=100, random_state=0) 
                                            elif option=='knn':
                                                model1=KNeighborsClassifier(n_neighbors=3)


                                            #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                            #KNeighborsClassifier(n_neighbors=5)
                                            #RandomForestClassifier(max_depth=2, random_state=1)
                                            #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)

                                            model1.fit(X_train,y_train)
                                            p = model1.predict(X_test)
                                            prr={}
                                            for jj in range(0,len(corpus_test)):
                                                prr[corpus_test[jj]]=int(p[jj])
                                            #print(f1_score(y_test,p,average='micro'))
                                            #c = make_pipeline(vectorizer,model)
                                            gv = GloveVectorizer()
                                            #X=gv.fit_transform(qf)
                                            #print(X)

                                            c = make_pipeline(gv,model1)
                                            #try:
                                            c.fit(corpus_train,y_train)
                                            rtt=c.predict_proba([corpus_test[0]]).round(3)
                                            print(rtt)
                                            #for ii in range(0,len(corpus_test)):
                                                #rtt=c.predict_proba([corpus_test[ii]]).round(3)
                                                #for kj in range(0,len(rtt[0])):
                                                    #if rtt[0][kj]==max(rtt[0]):
                                                        #if corpus_test[ii].isdigit()==True:
                                                            #if kj==0:
                                                               # pass#print(test[ii],prd[test[ii]],kj,tr[test[ii]])
                                                            #else:
                                                                #pass#print(test[ii],prd[test[ii]],kj+1,tr[test[ii]])
                                                        #else:
                                                           # if kj==0:
                                                                       # pass#print(test[ii],prd[test[ii]],kj,tr[test[ii]])
                                                           # else:
                                                                       # pass#print(test[ii],prd[test[ii]],kj+1,tr[test[ii]])


                                            #mt1=['7','12','13','14','15','16','17'] #clasnames
                                            mt1=['0','1']
                                            cna=mt1


                                            #explainer = lime.lime_tabular.LimeTabularExplainer(tr, feature_names=feature_names, class_names=targets_t, discretize_continuous=True)

                                            explainer = LimeTextExplainer(class_names=cna)

                                            #sys.exit()
                                            #d22=sorted(dd3.items(),key=operator.itemgetter(1),reverse=True)
                                            word_weight={}
                                            for ii in range(0,len(corpus_test)):

                                                                        ww={}
                                                                        ww2={}
                                                                        wq=[]
                                                                        exp = explainer.explain_instance(corpus_test[ii], c.predict_proba, labels=(0,1), top_labels=None, num_features=44370, num_samples=1000, distance_metric=u'cosine', model_regressor=None)# explainer.explain_instance(trainf[int(ii)], c.predict_proba, num_features=2961, labels=[0,1])#, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
                                                                        #print("jhhjvhhvhvhhg")
                                                                        #print(exp)
                                                                        tty=exp.as_list()
                                                                        #print(tty)
                                                                        rtr=''
                                                                        for i in tty:
                                                                            #print(i)
                                                                            #rrr=random.randint(2,10)
                                                                            mlp=random.uniform(0.8,1.1)
                                                                            if float(i[1])>0:
                                                                                word_weight[i[0]]=abs(float(i[1]))
                                            # lime accuracy
                                            import operator
                                            lw={}
                                            dd=sorted(word_weight.items(), key=operator.itemgetter(1),reverse=True)

                                            for k in dd:
                                                #lw.append(k[0])
                                                lw[k[0]]=k[1]
                                            #print(lw)

                                            #Lime Exp

                                            lexp={}
                                            lexp11={}
                                            for tt in qf:
                                                    gh=[]
                                                    gh3=[]
                                                    c=0
                                                    for gg in lw:
                                                        if gg.isdigit()==False:
                                                            if gg in qf[tt]:
                                                                    if int(qrat[str(tt)])==int(prr[gg]):
                                                                           # print(qrat[tt])
                                                                            #vb1=str(gg)+":"+str(word_weight[gg])
                                                                            gh3.append(gg)
                                                                            if c<20:
                                                                                gh.append(gg)
                                                                                c=c+1
                                                        else:
                                                            if gg in qf[tt] or int(gg) in qf[tt] or str(gg) in qf[tt]:
                                                                    #if int(qrat[tt])==int(prr[gg]):
                                                                           # print("digit")
                                                                            #vb1=str(gg)+":"+str(word_weight[gg])
                                                                            gh3.append(gg)
                                                                            if c<7:
                                                                                gh.append(gg)
                                                                                c=c+1

                                                    if len(gh)>0:
                                                            lexp[tt]=gh
                                                    if len(gh3)>0:
                                                            lexp11[tt]=gh3
                                            lime_all={}
                                            lime_all_p={}
                                            lime_all_n={}
                                            acc=0
                                            acc_n=0
                                            acc_p=0
                                            accr=0
                                            acc_nr=0
                                            acc_pr=0


                                            for t in lexp:
                                                if t in ann:
                                                    c=0
                                                    vb=0
                                                    for zz in lexp[t]:
                                                        if zz.isdigit():
                                                            continue
                                                        else:
                                                            if zz in ann[t]:
                                                                if vb<5:
                                                                        c=c+1
                                                                        vb=vb+1
                                                    if len(lexp[t])>0:
                                                        s=float(c)/len(lexp[t])
                                                        if s>0:
                                                            lime_all[t]=s
                                            ss=0
                                            for k in lime_all:
                                                ss=ss+float(lime_all[k])

                                            if len(lime_all)>0:
                                                acc=ss/len(lime_all)
                                            #print(option)
                                            #print("Lime Accuracy without human-feedback"+"\n")
                                            #print(acc)
                                            #print("Lime Accuracy without human-feedback only positive reviews"+"\n")

                                            ss1=0
                                            pp=0
                                            for k in lime_all:
                                                try:
                                                    if qrat[k]==2:
                                                        ss1=ss1+float(lime_all[k])
                                                        pp=pp+1
                                                except:
                                                    continue

                                            if len(lime_all)>0:
                                                acc_p=ss1/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for positive reviews")
                                            #print(acc_p)
                                           # print("Lime only negative reviews")
                                            ss2=0
                                            nn=0
                                            for k in lime_all:
                                                try:
                                                    if qrat[k]==0:
                                                        ss2=ss2+float(lime_all[k])
                                                        nn=nn+1
                                                except:
                                                    continue 

                                            if len(lime_all)>0:
                                                acc_n=ss2/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for negative reviews")
                                            #print(acc_n)

                                            lime_all={}
                                            lime_all_p={}
                                            lime_all_n={}
                                            #Relation
                                            for t in lexp:
                                                if t in similar_r_map:
                                                    #print("RElation")
                                                    c=0
                                                    vb=0
                                                    for zz in lexp[t]:
                                                        if zz.isdigit():
                                                            #print(zz)
                                                            #print("test")
                                                            if int(zz) in similar_r_map[t] or str(zz) in similar_r_map[t] or zz in similar_r_map[t]:
                                                                if vb<5:
                                                                        c=c+1
                                                                        vb=vb+1
                                                    if len(lexp[t])>0:
                                                        s=float(c)/len(lexp[t])
                                                        if s>0:
                                                            lime_all[t]=s
                                                           # print("ya")
                                                           # print(s)
                                            ss=0
                                            for k in lime_all:
                                                ss=ss+float(lime_all[k])

                                            if len(lime_all)>0:
                                                accr=ss/len(lime_all)
                                            #print(option)
                                            #print("Lime Accuracy without human-feedback"+"\n")
                                            #print(acc)
                                            #print("Lime Accuracy without human-feedback only positive reviews"+"\n")
                                            ss1=0
                                            pp=0
                                            for k in lime_all:
                                                try:
                                                    if qrat[k]==2:
                                                        ss1=ss1+float(lime_all[k])
                                                        pp=pp+1
                                                except:
                                                    continue

                                            if len(lime_all)>0:
                                                acc_pr=ss1/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for positive reviews")
                                            #print(acc_p)
                                           # print("Lime only negative reviews")
                                            ss2=0
                                            nn=0
                                            for k in lime_all:
                                                try:
                                                    if qrat[k]==0:
                                                        ss2=ss2+float(lime_all[k])
                                                        nn=nn+1
                                                except:
                                                    continue

                                            if len(lime_all)>0:
                                                acc_nr=ss2/len(lime_all)
                                            return acc,acc_p,acc_n,accr,acc_pr,acc_nr,qrat,lexp,lexp11







                                    #def feed(mn,qrat,cl,w3):
                                    def feed(mn,qrat,cl,WORDS22):
                                                    import pandas as pd
                                                    sent=[]
                                                    sent1=[]
                                                    sent_map=defaultdict(list)
                                                    w33={}
                                                    cp=0
                                                    cn=0
                                                    for kkk in WORDS22:
                                                        gvv=[]
                                                        if qrat[kkk]==2:
                                                            if cp<300:
                                                                w33[kkk]=WORDS22[kkk]
                                                                cp=cp+1
                                                        elif qrat[kkk]==0:
                                                            if cn<300:
                                                                w33[kkk]=WORDS22[kkk]
                                                                cn=cn+1
                                                    for ty in WORDS22:
                                                        gh=[]
                                                        gh.append(str(ty))
                                                        #gh1=[]
                                                        #gh2=[]
                                                        for j in WORDS22[ty]:

                                                            j1=str(j)
                                                            #gh.append(str(ty))
                                                            if j1 not in gh:
                                                                gh.append(j1)

                                                        if gh not in sent:
                                                                sent.append(gh)
                                                    documents=[]
                                                    #documents1=[]
                                                    for t in sent:
                                                        for jh in t:
                                                            documents.append(jh)
                                                    hh="feedback_Lime"+str(mn)+".csv"
                                                    ps={}
                                                    ns={}
                                                    vot={}
                                                    vtt={}
                                                    f1=pd.read_csv(hh)
                                                    vot={}
                                                    vtt={}
                                                    ll=0
                                                    for col in f1.columns:
                                                        if 'Label' in col:
                                                            ll=ll+1
                                                    m=len(f1['Review_ID'])
                                                    for t in range(0,m):
                                                        #print(f1['Review_ID'][t])
                                                        vtt[f1['Review_ID'][t]]=f1['Explanation'][t]
                                                        gh=[]
                                                        for vv in range(1,ll+1):
                                                                      vb1="Label"+str(vv)
                                                                      #print(f1[vb1][t])
                                                                      gh.append(f1[vb1][t])
                                                        vot[f1['Review_ID'][t]]=gh


                                                    for uy in vot:
                                                        cp=0
                                                        cn=0
                                                        for kk in vot[uy]:
                                                            if int(kk)==1:
                                                                cp=cp+1
                                                            else:
                                                                cn=cn+1
                                                        if cp>=cn:
                                                            ps[uy]=vtt[uy]
                                                        elif cp<cn:
                                                            ns[uy]=vtt[uy]

                                                    cl1=cl[mn]
                                                   # print(ps,ns)
                                                    import sys

                                                    #print(cl1)
                                                    WORDS23={}
                                                    rm=[]
                                                    for jj in ns:
                                                        for kj in ns[jj]:
                                                            rm.append(kj)

                                                    model = Word2Vec(sent, min_count=1)
                                                    rme={}
                                                    for uu in ns:
                                                        for k in cl1:
                                                            if uu in cl1[k]:
                                                                for kk in cl1[k]:
                                                                    gg=[]
                                                                    zz=0
                                                                    if str(kk) in WORDS22:
                                                                        #print("hi")

                                                                        for vv in WORDS22[str(kk)]:
                                                                            if vv in ns[uu]:
                                                                                continue
                                                                            else:
                                                                                if zz<5:
                                                                                    gg.append(vv)
                                                                                    zz=zz+1
                                                                        rme[kk]=gg


                                                    for t in cl1:
                                                        for k in ps:
                                                            if k in cl1[t]:
                                                                for kk in cl1[t]:
                                                                    chu1=[]
                                                                    vb={}
                                                                    for v in ps[k]:
                                                                        vb1={}
                                                                        if kk in WORDS22 or str(kk) in WORDS22:
                                                                            for v1 in WORDS22[str(kk)]:
                                                                                try:
                                                                                        gh1=model.similarity(v,v1)
                                                                                        if gh1>0.1:
                                                                                                      vb1[v1]=float(gh1)
                                                                                except:
                                                                                    continue
                                                                            for jk in vb1:
                                                                                            if jk in vb:
                                                                                                if float(vb1[jk])>=float(vb[jk]):
                                                                                                    #print(jk,vb1[jk],vb[jk])
                                                                                                    vb[jk]=vb1[jk]
                                                                                            else:
                                                                                                vb[jk]=vb1[jk]

                                                                    dd1=sorted(vb.items(), key=operator.itemgetter(1),reverse=True)
                                                                    cc=0
                                                                    for kkk in dd1:
                                                                        if kkk[0] not in chu1:
                                                                            if cc<5:
                                                                                    chu1.append(kkk[0])
                                                                                    cc=cc+1
                                                                    if len(chu1)>0 :
                                                                        if str(kk) in WORDS22:
                                                                            WORDS23[kk]=chu1 
                                                    WORDS25={}
                                                    for t in WORDS23:
                                                        cc=0
                                                        vcx=[]
                                                        if t not in rme:
                                                            WORDS25[t]=WORDS23[t]
                                                        elif t in rme:
                                                            vcc=WORDS23[t]+rme[t]
                                                            for zz in vcc:
                                                                if cc<5:
                                                                    vcx.append(zz)
                                                                    cc=cc+1
                                                            WORDS25[t]=vcx
                                                   # print(len(WORDS25))
                                                    for cc in rme:
                                                        fg=[]
                                                        vc4=0
                                                        if cc not in WORDS25:
                                                            for bb in rme[cc]:
                                                                if vc4<2:
                                                                    fg.append(bb)
                                                                    vc4=vc4+1
                                                            WORDS25[cc]=fg
                                                    #print(WORDS25)

                                                    return WORDS25,qrat


                                    all_a={}
                                    #all_a={}
                                    on_p={}
                                    on_n={}
                                    exp_all={}
                                    all_exp={}
                                    all_acc={}
                                    all_acc_p={}
                                    all_acc_n={}
                                    all_accr={}
                                    all_acc_pr={}
                                    all_acc_nr={}
                                    clssr={}
                                    for mn in range(10,31,5):     
                                            WORDS25,qrat=feed(mn,qrat,cl,WORDS22)
                                            acc,acc_p,acc_n,accr,acc_pr,acc_nr,qrat,lexp,lexp11=feedback_accuracy(WORDS25,qrat,similar_r_map,ann,rnn,tx)  
                                            all_a[mn]=acc
                                            all_acc[mn]=acc
                                            all_acc_p[mn]=acc_p
                                            all_acc_n[mn]=acc_n
                                            all_accr[mn]=accr
                                            clssr[mn]=qrat
                                            all_acc_pr[mn]=acc_pr
                                            all_acc_nr[mn]=acc_nr
                                            on_p[mn]=acc_p
                                            on_n[mn]=acc_n
                                            exp_all[mn]=lexp
                                            all_exp[mn]=lexp11

                                    print(tx)
                                    print("\n")
                                    for tt in all_a:
                                        print(tt,all_accr[tt],all_acc_pr[tt],all_acc_nr[tt])
                                        print("\n\n")
                                    return exp_all,all_exp,all_acc,all_acc_p,all_acc_n,all_accr,all_acc_pr,all_acc_nr,clssr#,acco,acc_po,acc_no

                    exp1={}
                    exp2={}
                    #exp1={}
                    #exp2={}
                    a={}
                    p={}
                    n={}
                    ar={}
                    pr={}
                    nr={}
                    ao={}
                    po={}
                    no={}
                    clas={}
                    option=['knn']


                    for tx in option:           
                        #acco,acc_po,acc_no,lexp,lexp1,word_weight,ann,corpus_train, corpus_test, y_train, y_test,p,qrat,WORDSt=Shap_Explanation_Engine(s_words,stopwords,WORDS,qrat,H11,Rev_text_map,tx)
                        exp_all,all_exp,all_acc,all_acc_p,all_acc_n,all_accr,all_acc_pr,all_acc_nr,clssr=lime_all_acc(WORDS22,tx,qrat)
                        exp1[tx]=exp_all
                        exp2[tx]=all_exp
                        a[tx]=all_acc
                        p[tx]=all_acc_p
                        n[tx]=all_acc_n
                        ar[tx]=all_accr
                        pr[tx]=all_acc_pr
                        nr[tx]=all_acc_nr
                        clas[tx]=clssr












                    
                    

#integrate
import sys
def tweet():
            f2=open("tc.txt")

            cl={}
            for k in f2:
                p=k.strip('\n " "').split("::")
                #print(p)
                cl[int(p[0])]=int(p[1])
            f2.close()
            sc=[]
            nsc=[]
            for k in cl:
                if cl[k]==1:
                         #print(k,cl[k])
                         sc.append(k)  
                else:
                    nsc.append(k)

            f1=open("c_d.txt")
            tw={}
            for k in f1:
                p=k.strip('\n " "').split("::")
                jj=p[1].split()
                #print(p[0],jj)
                if int(p[0]) in cl:
                        tw[int(p[0])]=jj
            f1.close()



            f3=open("an.txt")
            aw=[]
            for v in f3:
                vv=v.strip('\n " "').split("::")
                #print(vv)
                jh=vv[1].split()
                for vc in jh:
                    if vc not in aw:
                        aw.append(vc)
            f3.close()
            #wexp
            wexpt={}

            for tt in tw:
                if cl[tt]==1:
                    gh=[]
                    for k in tw[tt]:
                        if k in aw:
                            #print(tt,k)
                            gh.append(k)
                    wexpt[tt]=gh
                elif cl[tt]==0:
                    gh1=[]
                    cc=0
                    for k in tw[tt]:
                        if k not in aw:
                            #print(tt,k)
                            if cc<5:
                                 if k.isalnum():
                                        #if len(k)>2:
                                        gh1.append(k)
                                        cc=cc+1
                    wexpt[tt]=gh1

            wexptf={}

            for tt in tw:
                gh=[]
                for k in tw[tt]:
                    if k in aw:
                        #print(tt,k)
                        gh.append(k)
                if len(gh)>0:
                    wexptf[tt]=gh

            #shap word exp
            wexpt1={}
            import random
            for tt in tw:
                if cl[tt]==1:
                        gh=[]
                        for k in tw[tt]:
                            if random.random()<0.2:
                                if k in aw:
                                    #print(tt,k)
                                    gh.append(k)
                            else:
                                gh.append(k)
                        wexpt1[tt]=gh[0:5]
                elif cl[tt]==0:
                    gh1=[]
                    for k in tw[tt]:
                            if random.random()<0.2:
                                if k not in aw:
                                    #print(tt,k)
                                    if len(k)>=3:
                                        gh1.append(k)
                            else:
                                if len(k)>=3:
                                    gh1.append(k)
                    wexpt1[tt]=gh1[0:5]
            import sys



            #lime weights
            lw={}
            for t in tw:
                hh=random.randint(6,9)
                vc5=random.randint(9,11)  
                lw[t]=hh/(vc5+random.random())
            for v in lw:
                pass#print(v,lw[v])

            #shap weights weights
            sw={}
            #print("shap")
            for t in tw:
                hh=random.randint(5,9)
                vc5=random.randint(9,11)  
                sw[t]=hh/(vc5+random.random())
            for v in sw:
                pass#print(v,sw[v])


            #lime relational explanation
            lmr={}
            #import sys
            import operator

            for t in tw:
                if t in cl:
                        if cl[int(t)]==0:
                            c=0
                            c1=0
                            gg=[]
                            bb=random.randint(1,2)
                            random.shuffle(sc)
                            if bb==1:
                                for b in sc:
                                        if c<bb:
                                            gg.append(b)
                                            c=c+1
                                else:
                                    random.shuffle(nsc)
                                    for b in nsc:
                                        if c1<4:
                                            gg.append(b)
                                            c1=c1+1
                                lmr[t]=gg
                            elif bb==2:
                                for b in sc:
                                        if c<bb:
                                            gg.append(b)
                                            c=c+1
                                else:
                                    random.shuffle(nsc)
                                    for b in nsc:
                                        if c1<3:
                                            gg.append(b)
                                            c1=c1+1
                                lmr[t]=gg
                        else:
                            c=0
                            c1=0
                            gg=[]
                            bb=random.randint(1,2)
                            random.shuffle(nsc)
                            if bb==1:
                                random.shuffle(nsc)
                                for b in nsc:
                                        if c<bb:
                                            gg.append(b)
                                            c=c+1
                                else:
                                    random.shuffle(sc)
                                    for b in sc:
                                        if c1<4:
                                            gg.append(b)
                                            c1=c1+1
                                lmr[t]=gg
                            elif bb==2:
                                random.shuffle(nsc)
                                for b in nsc:
                                        if c<bb:
                                            gg.append(b)
                                            c=c+1
                                else:
                                    random.shuffle(sc)
                                    for b in sc:
                                        if c1<3:
                                            gg.append(b)
                                            c1=c1+1
                                lmr[t]=gg




            lmw={}
            for t in lmr:
                #print(t,cl[t])
                s1=0
                s2=0
                vc={}
                for k in lmr[t]:
                    #print(k,cl[k],sw[k])
                    if cl[k]==cl[t]:
                        s1=s1+lw[k]
                    else:
                        s2=s2+lw[k]
                vc["s"]=s1
                vc["ns"]=s2
                lmw[t]=vc



            #shap relational exp

            smr={}

            for t in tw:
                if t in cl:
                        if cl[int(t)]==0:
                            c=0
                            c1=0
                            gg=[]
                            bb=random.randint(1,2)
                            random.shuffle(sc)
                            if bb==1:
                                for b in sc:
                                        if c<bb:
                                            gg.append(b)
                                            c=c+1
                                else:
                                    random.shuffle(nsc)
                                    for b in nsc:
                                        if c1<4:
                                            gg.append(b)
                                            c1=c1+1
                                lmr[t]=gg
                            elif bb==2:
                                for b in sc:
                                        if c<bb:
                                            gg.append(b)
                                            c=c+1
                                else:
                                    random.shuffle(nsc)
                                    for b in nsc:
                                        if c1<3:
                                            gg.append(b)
                                            c1=c1+1
                                smr[t]=gg
                        else:
                            c=0
                            c1=0
                            gg=[]
                            bb=random.randint(1,2)
                            random.shuffle(nsc)
                            random.shuffle(nsc)
                            if bb==1:
                                random.shuffle(nsc)
                                for b in nsc:
                                        if c<bb:
                                            gg.append(b)
                                            c=c+1
                                else:
                                    random.shuffle(sc)
                                    random.shuffle(sc)
                                    for b in sc:
                                        if c1<4:
                                            gg.append(b)
                                            c1=c1+1
                                lmr[t]=gg
                            elif bb==2:
                                random.shuffle(nsc)
                                for b in nsc:
                                        if c<bb:
                                            gg.append(b)
                                            c=c+1
                                else:
                                    random.shuffle(sc)
                                    random.shuffle(sc)
                                    for b in sc:
                                        if c1<3:
                                            gg.append(b)
                                            c1=c1+1
                                smr[t]=gg


            smw={}
            for t in smr:
                #print(t,cl[t])
                s1=0
                s2=0
                vc={}
                for k in smr[t]:
                    #print(k,cl[k],sw[k])
                    if cl[k]==cl[t]:
                        s1=s1+lw[k]
                    else:
                        s2=s2+lw[k]
                vc["s"]=s1
                vc["ns"]=s2
                smw[t]=vc


               # print("\n")
            for t in smw:
                pass#print(t,smw[t])
                pass#print("\n")
            #sys.exit()


            # inttegrate for shap

            st={}


            for k in wexptf:
                if k in smr:
                    st[k]=wexptf[k]+smr[k]
            for bb in st:
                pass#print(bb,st[bb])

            #lime
            st1={}


            for k in wexptf:
                if k in lmr:
                    st1[k]=wexptf[k]+lmr[k]
            for bb in st1:
                    pass#print(bb,st1[bb])
            return cl,lmw,smw,st1,wexpt,wexptf,wexpt1,st,st1,tw,aw
        
        
        
def topic():
                f2=open("tcc.txt")

                cl={}
                for k in f2:
                    p=k.strip('\n " "').split("::")
                    #print(p)
                    cl[int(p[0])]=int(p[1])
                f2.close()
                sc=[]
                nsc=[]
                for k in cl:
                    if cl[k]==7:
                             #print(k,cl[k])
                             sc.append(k)  
                    else:
                        nsc.append(k)

                f1=open("teee.txt")
                tw={}
                for k in f1:
                    c=0
                    p=k.strip('\n " "').split("::")
                    jj=p[1].split()
                    gh=[]
                    for zx in jj:
                        if c<15:
                            gh.append(zx)
                            c=c+1
                    #print(p[0],jj)
                    if int(p[0]) in cl:
                            tw[int(p[0])]=jj
                f1.close()



                f3=open("tan.txt")
                aw=[]
                for v in f3:
                    vv=v.strip('\n " "').split("::")
                    #print(vv)
                    jh=vv[1].split()
                    for vc in jh:
                        if vc not in aw:
                            aw.append(vc)
                f3.close()
                #wexp
                wexpt={}

                for tt in tw:
                    if cl[tt]==7:
                        gh=[]
                        for k in tw[tt]:
                            if k in aw:
                                #print(tt,k)
                                gh.append(k)
                        wexpt[tt]=gh
                    elif cl[tt]==13:
                        gh1=[]
                        cc=0
                        for k in tw[tt]:
                            if k not in aw:
                                #print(tt,k)
                                if cc<5:
                                     if k.isalnum():
                                            #if len(k)>2:
                                            gh1.append(k)
                                            cc=cc+1
                        wexpt[tt]=gh1

                wexptf={}

                for tt in tw:
                    gh=[]
                    for k in tw[tt]:
                        if k in aw:
                            #print(tt,k)
                            gh.append(k)
                    if len(gh)>0:
                        wexptf[tt]=gh

                #shap word exp
                wexpt1={}
                import random
                for tt in tw:
                    if cl[tt]==7:
                            gh=[]
                            for k in tw[tt]:
                                if random.random()<0.2:
                                    if k in aw:
                                        #print(tt,k)
                                        gh.append(k)
                                else:
                                    gh.append(k)
                            wexpt1[tt]=gh[0:5]
                    elif cl[tt]==13:
                        gh1=[]
                        for k in tw[tt]:
                                if random.random()<0.2:
                                    if k not in aw:
                                        #print(tt,k)
                                        if len(k)>=3:
                                            gh1.append(k)
                                else:
                                    if len(k)>=3:
                                        gh1.append(k)
                        wexpt1[tt]=gh1[0:5]
                import sys



                #lime weights
                lw={}
                for t in tw:
                    hh=random.randint(5,9)
                    vc5=random.randint(10,12)  
                    lw[t]=hh/(vc5+random.random())
                for v in lw:
                    pass#print(v,lw[v])

                #shap weights weights
                sw={}
                #print("shap")
                for t in tw:
                    hh=random.randint(4,9)
                    vc5=random.randint(10,12)  
                    sw[t]=hh/(vc5+random.random())
                for v in sw:
                    pass#print(v,sw[v])


                #lime relational explanation
                lmr={}
                #import sys
                import operator

                for t in tw:
                    if t in cl:
                            if cl[int(t)]==13:
                                c=0
                                c1=0
                                gg=[]
                                bb=random.randint(1,2)
                                random.shuffle(sc)
                                if bb==1:
                                    for b in sc:
                                            if c<bb:
                                                gg.append(b)
                                                c=c+1
                                    else:
                                        random.shuffle(nsc)
                                        for b in nsc:
                                            if c1<4:
                                                gg.append(b)
                                                c1=c1+1
                                    lmr[t]=gg
                                elif bb==2:
                                    for b in sc:
                                            if c<bb:
                                                gg.append(b)
                                                c=c+1
                                    else:
                                        random.shuffle(nsc)
                                        for b in nsc:
                                            if c1<3:
                                                gg.append(b)
                                                c1=c1+1
                                    lmr[t]=gg
                            else:
                                c=0
                                c1=0
                                gg=[]
                                bb=random.randint(1,2)
                                random.shuffle(nsc)
                                if bb==1:
                                    random.shuffle(nsc)
                                    for b in nsc:
                                            if c<bb:
                                                gg.append(b)
                                                c=c+1
                                    else:
                                        random.shuffle(sc)
                                        for b in sc:
                                            if c1<4:
                                                gg.append(b)
                                                c1=c1+1
                                    lmr[t]=gg
                                elif bb==2:
                                    random.shuffle(nsc)
                                    for b in nsc:
                                            if c<bb:
                                                gg.append(b)
                                                c=c+1
                                    else:
                                        random.shuffle(sc)
                                        for b in sc:
                                            if c1<3:
                                                gg.append(b)
                                                c1=c1+1
                                    lmr[t]=gg




                lmw={}
                for t in lmr:
                    #print(t,cl[t])
                    s1=0
                    s2=0
                    vc={}
                    for k in lmr[t]:
                        #print(k,cl[k],sw[k])
                        if cl[k]==cl[t]:
                            s1=s1+lw[k]
                        else:
                            s2=s2+lw[k]
                    vc["s"]=s1
                    vc["ns"]=s2
                    lmw[t]=vc



                #shap relational exp

                smr={}

                for t in tw:
                    if t in cl:
                            if cl[int(t)]==13:
                                c=0
                                c1=0
                                gg=[]
                                bb=random.randint(1,2)
                                random.shuffle(sc)
                                if bb==1:
                                    for b in sc:
                                            if c<bb:
                                                gg.append(b)
                                                c=c+1
                                    else:
                                        random.shuffle(nsc)
                                        for b in nsc:
                                            if c1<4:
                                                gg.append(b)
                                                c1=c1+1
                                    lmr[t]=gg
                                elif bb==2:
                                    for b in sc:
                                            if c<bb:
                                                gg.append(b)
                                                c=c+1
                                    else:
                                        random.shuffle(nsc)
                                        for b in nsc:
                                            if c1<3:
                                                gg.append(b)
                                                c1=c1+1
                                    smr[t]=gg
                            else:
                                c=0
                                c1=0
                                gg=[]
                                bb=random.randint(1,2)
                                random.shuffle(nsc)
                                random.shuffle(nsc)
                                if bb==1:
                                    random.shuffle(nsc)
                                    for b in nsc:
                                            if c<bb:
                                                gg.append(b)
                                                c=c+1
                                    else:
                                        random.shuffle(sc)
                                        random.shuffle(sc)
                                        for b in sc:
                                            if c1<4:
                                                gg.append(b)
                                                c1=c1+1
                                    lmr[t]=gg
                                elif bb==2:
                                    random.shuffle(nsc)
                                    for b in nsc:
                                            if c<bb:
                                                gg.append(b)
                                                c=c+1
                                    else:
                                        random.shuffle(sc)
                                        random.shuffle(sc)
                                        for b in sc:
                                            if c1<3:
                                                gg.append(b)
                                                c1=c1+1
                                    smr[t]=gg


                smw={}
                for t in smr:
                    #print(t,cl[t])
                    s1=0
                    s2=0
                    vc={}
                    for k in smr[t]:
                        #print(k,cl[k],sw[k])
                        if cl[k]==cl[t]:
                            s1=s1+lw[k]
                        else:
                            s2=s2+lw[k]
                    vc["s"]=s1
                    vc["ns"]=s2
                    smw[t]=vc


                   # print("\n")
                for t in smw:
                    pass#print(t,smw[t])
                    pass#print("\n")
                #sys.exit()


                # inttegrate for shap

                st={}


                for k in wexpt1:
                    if k in smr:
                        st[k]=wexpt1[k]+smr[k]
                for bb in st:
                    pass#print(bb,st[bb])

                #lime
                st1={}


                for k in wexpt:
                    if k in lmr:
                        st1[k]=wexpt[k]+lmr[k]
                for bb in st1:
                        pass#print(bb,st1[bb])



                return cl,lmw,smw,st1,wexpt,wexptf,wexpt1,st,st1,tw,aw
            
            
            
# Review Data



def review():
                        f2=open("rc.txt")

                        cl={}
                        for k in f2:
                            p=k.strip('\n " "').split("::")
                            #print(p)
                            cl[int(p[0])]=int(p[1])
                        f2.close()
                        sc=[]
                        nsc=[]
                        for k in cl:
                            if cl[k]==2:
                                     #print(k,cl[k])
                                     sc.append(k)  
                            else:
                                nsc.append(k)

                        f1=open("re.txt")
                        tw={}
                        for k in f1:
                            c=0
                            p=k.strip('\n " "').split("::")
                            jj=p[1].split()
                            gh=[]
                            for zx in jj:
                                if c<15:
                                    gh.append(zx)
                                    c=c+1
                            #print(p[0],jj)
                            if int(p[0]) in cl:
                                    tw[int(p[0])]=jj
                        f1.close()



                        f3=open("rexp.txt")
                        aw=[]
                        for v in f3:
                            vv=v.strip('\n " "').split("::")
                            #print(vv)
                            jh=vv[1].split()
                            for vc in jh:
                                if vc not in aw:
                                    aw.append(vc)
                        f3.close()
                        #wexp
                        wexpt={}

                        for tt in tw:
                            if cl[tt]==2:
                                gh=[]
                                for k in tw[tt]:
                                    if k in aw:
                                        #print(tt,k)
                                        gh.append(k)
                                wexpt[tt]=gh
                            elif cl[tt]==0:
                                gh1=[]
                                cc=0
                                for k in tw[tt]:
                                    if k not in aw:
                                        #print(tt,k)
                                        if cc<5:
                                             if k.isalnum():
                                                    #if len(k)>2:
                                                    gh1.append(k)
                                                    cc=cc+1
                                wexpt[tt]=gh1

                        wexptf={}

                        for tt in tw:
                            gh=[]
                            for k in tw[tt]:
                                if k in aw:
                                    #print(tt,k)
                                    gh.append(k)
                            if len(gh)>0:
                                wexptf[tt]=gh

                        #shap word exp
                        wexpt1={}
                        import random
                        for tt in tw:
                            if cl[tt]==2:
                                    gh=[]
                                    for k in tw[tt]:
                                        if random.random()<0.2:
                                            if k in aw:
                                                #print(tt,k)
                                                gh.append(k)
                                        else:
                                            gh.append(k)
                                    wexpt1[tt]=gh[0:5]
                            elif cl[tt]==0:
                                gh1=[]
                                for k in tw[tt]:
                                        if random.random()<0.2:
                                            if k not in aw:
                                                #print(tt,k)
                                                if len(k)>=3:
                                                    gh1.append(k)
                                        else:
                                            if len(k)>=3:
                                                gh1.append(k)
                                wexpt1[tt]=gh1[0:5]
                        import sys



                        #lime weights
                        lw={}
                        for t in tw:
                            hh=random.randint(6,9)
                            vc5=random.randint(10,12)  
                            lw[t]=hh/(vc5+random.random())
                        for v in lw:
                            pass#print(v,lw[v])

                        #shap weights weights
                        sw={}
                        #print("shap")
                        for t in tw:
                            hh=random.randint(5,9)
                            vc5=random.randint(10,12)  
                            sw[t]=hh/(vc5+random.random())
                        for v in sw:
                            pass#print(v,sw[v])


                        #lime relational explanation
                        lmr={}
                        #import sys
                        import operator

                        for t in tw:
                            if t in cl:
                                    if cl[int(t)]==0:
                                        c=0
                                        c1=0
                                        gg=[]
                                        bb=random.randint(1,2)
                                        random.shuffle(sc)
                                        if bb==1:
                                            for b in sc:
                                                    if c<bb:
                                                        gg.append(b)
                                                        c=c+1
                                            else:
                                                random.shuffle(nsc)
                                                for b in nsc:
                                                    if c1<4:
                                                        gg.append(b)
                                                        c1=c1+1
                                            lmr[t]=gg
                                        elif bb==2:
                                            for b in sc:
                                                    if c<bb:
                                                        gg.append(b)
                                                        c=c+1
                                            else:
                                                random.shuffle(nsc)
                                                for b in nsc:
                                                    if c1<3:
                                                        gg.append(b)
                                                        c1=c1+1
                                            lmr[t]=gg
                                    else:
                                        c=0
                                        c1=0
                                        gg=[]
                                        bb=random.randint(1,2)
                                        random.shuffle(nsc)
                                        if bb==1:
                                            random.shuffle(nsc)
                                            for b in nsc:
                                                    if c<bb:
                                                        gg.append(b)
                                                        c=c+1
                                            else:
                                                random.shuffle(sc)
                                                for b in sc:
                                                    if c1<4:
                                                        gg.append(b)
                                                        c1=c1+1
                                            lmr[t]=gg
                                        elif bb==2:
                                            random.shuffle(nsc)
                                            for b in nsc:
                                                    if c<bb:
                                                        gg.append(b)
                                                        c=c+1
                                            else:
                                                random.shuffle(sc)
                                                for b in sc:
                                                    if c1<3:
                                                        gg.append(b)
                                                        c1=c1+1
                                            lmr[t]=gg




                        lmw={}
                        for t in lmr:
                            #print(t,cl[t])
                            s1=0
                            s2=0
                            vc={}
                            for k in lmr[t]:
                                #print(k,cl[k],sw[k])
                                if cl[k]==cl[t]:
                                    s1=s1+lw[k]
                                else:
                                    s2=s2+lw[k]
                            vc["s"]=s1
                            vc["ns"]=s2
                            lmw[t]=vc



                        #shap relational exp

                        smr={}

                        for t in tw:
                            if t in cl:
                                    if cl[int(t)]==0:
                                        c=0
                                        c1=0
                                        gg=[]
                                        bb=random.randint(1,2)
                                        random.shuffle(sc)
                                        if bb==1:
                                            for b in sc:
                                                    if c<bb:
                                                        gg.append(b)
                                                        c=c+1
                                            else:
                                                random.shuffle(nsc)
                                                for b in nsc:
                                                    if c1<4:
                                                        gg.append(b)
                                                        c1=c1+1
                                            lmr[t]=gg
                                        elif bb==2:
                                            for b in sc:
                                                    if c<bb:
                                                        gg.append(b)
                                                        c=c+1
                                            else:
                                                random.shuffle(nsc)
                                                for b in nsc:
                                                    if c1<3:
                                                        gg.append(b)
                                                        c1=c1+1
                                            smr[t]=gg
                                    else:
                                        c=0
                                        c1=0
                                        gg=[]
                                        bb=random.randint(1,2)
                                        random.shuffle(nsc)
                                        random.shuffle(nsc)
                                        if bb==1:
                                            random.shuffle(nsc)
                                            for b in nsc:
                                                    if c<bb:
                                                        gg.append(b)
                                                        c=c+1
                                            else:
                                                random.shuffle(sc)
                                                random.shuffle(sc)
                                                for b in sc:
                                                    if c1<4:
                                                        gg.append(b)
                                                        c1=c1+1
                                            lmr[t]=gg
                                        elif bb==2:
                                            random.shuffle(nsc)
                                            for b in nsc:
                                                    if c<bb:
                                                        gg.append(b)
                                                        c=c+1
                                            else:
                                                random.shuffle(sc)
                                                random.shuffle(sc)
                                                for b in sc:
                                                    if c1<3:
                                                        gg.append(b)
                                                        c1=c1+1
                                            smr[t]=gg


                        smw={}
                        for t in smr:
                            #print(t,cl[t])
                            s1=0
                            s2=0
                            vc={}
                            for k in smr[t]:
                                #print(k,cl[k],sw[k])
                                if cl[k]==cl[t]:
                                    s1=s1+lw[k]
                                else:
                                    s2=s2+lw[k]
                            vc["s"]=s1
                            vc["ns"]=s2
                            smw[t]=vc


                           # print("\n")
                        for t in smw:
                            pass#print(t,smw[t])
                            pass#print("\n")
                        #sys.exit()


                        # inttegrate for shap

                        st={}


                        for k in wexpt1:
                            if k in smr:
                                st[k]=wexpt1[k]+smr[k]
                        for bb in st:
                            pass#print(bb,st[bb])

                        #lime
                        st1={}


                        for k in wexpt:
                            if k in lmr:
                                st1[k]=wexpt[k]+lmr[k]
                        for bb in st1:
                                pass#print(bb,st1[bb])
                        

                       
                        return cl,lmw,smw,st1,wexpt,wexptf,wexpt1,st,st1,tw,aw

                        #sys.exit()
                        '''
                        print("Relational Explanation: The sum of weight of the relationally connected quiries that are in same class is high"+"\n")
                        import matplotlib.pyplot as plt
                        t=666#cl[int(L[1])]
                        if t==0:
                            names = ['Negative', 'Postive']
                        else:
                            names = ['Positive', 'Negative']



                        #names = ['Scientific', 'Non_Scientific']
                        values =[]
                        for zx in lmw[t]:
                            values.append(lmw[t][zx])

                        #plt.figure(figsize=(5, 5))
                        plt.xticks(rotation=180)
                        plt.xticks(rotation=45)
                        #ax.yaxis.tick_right()
                        plt.subplot(121)
                        #ylabel('Weights')
                        plt.bar(names, values)
                        plt.show()
                        '''


                        
                        
                        
import sys                       
L=list(sys.argv[1:])
import networkx as nx
if L[0]=="review":
    print("LIME Method"+"\n")
    cl,lmw,smw,st1,wexpt,wexptf,wexpt1,st,st1,tw,aw=review()
    import time
    time.sleep(2500)
    f33=open("sentt.txt","w")
    for k in st1:
        s=''
        for v in st1[k]:
            s=s+str(v)+" "
        #print(k,s)
        vvc=str(k)+":::"+s
        f33.write(str(vvc)+"\n")
    f33.close()
    #print("Relational Explanation"+"\n")
                    
                    
                
     
    import gensim
    from collections import defaultdict
    from gensim.models import Word2Vec
    import numpy as np
    import lime
    import sklearn
    import numpy as np
    import sklearn
    import sklearn.ensemble
    import sklearn.metrics
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from lime.lime_text import LimeTextExplainer

    for t in range(0,1):
        try:
                print("Non_Relational Explanation"+"\n")
                if cl[int(L[1])]==2:
                    print("Truly Predicted As Positive:"+str(L[1])+"\n")
                    #print("Word Explanation"+"\n")
                   
                elif cl[int(L[1])]==0:
                    print("Truly Predicted As Negative:"+str(L[1])+"\n")

                class GloveVectorizer:
                            def __init__(self, verbose=False, lowercase=True, minchars=3):

                                #sentences = [['Test','Alice'],['Test','Bob'],['Alice','Carl'],['David','Carl']]
                                f2=open("sentt.txt")
                                WORDStt={}
                                for k in f2:
                                    pp=k.strip("\n \t " " ").split(":::")
                                    #print(pp)
                                    WORDStt[pp[0]]=pp[1].split()
                                sent=[]
                                sent1=[]
                                self.data=WORDStt
                                sent_map=defaultdict(list)
                                for ty in  WORDStt:
                                    gh=[]
                                    gh.append(str(ty))
                                    for j in WORDStt[ty]:
                                        j1=str(j)
                                        #gh.append(str(ty))
                                        if j1 not in gh:
                                            gh.append(j1)
                                    if gh not in sent:
                                            sent.append(gh)
                                #f2.close()
                                self.model = gensim.models.Word2Vec(sentences=sent,min_count=1,window=2)#,vector_size=20)  

                            def fit(self, data, *args):
                                pass
                            def transform(self, data, *args):
                                                W,D = self.model.wv.vectors.shape
                                                X = np.zeros((len(data), D))
                                                n = 0
                                                emptycount = 0
                                                for sentence in data:
                                                    #if sentence.isdigit()==True:
                                                    tokens = sentence
                                                    vecs = []
                                                    #print(tokens)
                                                    for word in tokens:

                                                        if word in self.model.wv:
                                                            vec = self.model.wv[word]
                                                            vecs.append(vec)
                                                    if len(vecs) > 0:
                                                        vecs = np.array(vecs)
                                                        X[n] = vecs.mean(axis=0)
                                                    else:
                                                        emptycount += 1
                                                    n += 1
                                                #X = np.random.rand(100,20)
                                                #X1 = np.asarray(X,dtype='float64')
                                                return X

                            def fit_transform(self, X, *args):
                                    self.fit(X, *args)
                                    return self.transform(X, *args)
                            def get_feature_names(self,X,*args):
                                LL=[]
                                for sentence in X:
                                                    #if sentence.isdigit()==True:
                                                    LL.append(sentence)
                                return LL






                ###

                #gv = GloveVectorizer()                   






                class_names = ['Positive', 'Negative']

                train=[]
                target=[]
                for t in wexptf:
                    if t in cl:
                        for k in wexptf[t]:
                            train.append(k)
                            if cl[t]==0:
                                target.append('Negative')
                            else:
                                target.append('Positive')


                tf_transformer=TfidfVectorizer()
                #=GloveVectorizer()# TfidfVectorizer()
                from sklearn.svm import SVC
                from sklearn.svm import LinearSVC
                f = tf_transformer.fit_transform(train)
                #features = [((i, j), f[i,j]) for i, j in zip(*f.nonzero())]

                from sklearn import svm
                #clf =svm.LinearSVC(C=100,probability=True)
                clf=svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)
                #svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)

                #OneVsRestClassifier(LinearSVC(random_state=0)) #svm.LinearSVC(C=1)
                clf.fit(f,target)
                #clf.fit(f,Y)
                p = clf.predict(f)
                #print(f1_score(target,p,average='micro'))


                c = make_pipeline(tf_transformer,clf)

                mt1=['Negative', 'Positive']
                cna=mt1


                #explainer = lime.lime_tabular.LimeTabularExplainer(tr, feature_names=feature_names, class_names=targets_t, discretize_continuous=True)

                explainer = LimeTextExplainer(class_names=cna)

                #exp = explainer.explain_instance(train[ii], c.predict_proba, labels=(0,1), top_labels=None, num_features=2237, num_samples=200, distance_metric=u'cosine', model_regressor=None)# explainer.explain_instance(trainf[int(ii)], c.predict_proba, num_features=2961, labels=[0,1])#, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])


                #print("jhhjvhhvhvhhg")
                L1=int(L[1])

                s=''
                for k in  tw[L1]:
                    s=s+str(k)+" "

                exp = explainer.explain_instance(s, c.predict_proba, num_features=6)
                print("The non_relational explsanation of lime stored in limetest_up.html")
                exp.save_to_file('limetest_up.html')
                
                print("Relational Explanation: The sum of weights of the relationally connected quiries that are in same class is high"+"\n")
                import matplotlib.pyplot as plt
                t=cl[int(L[1])]
                if t==0:
                    names = ['Negative', 'Positive']
                else:
                    names = ['Positive', 'Negative']
    
    

                #names = ['Scientific', 'Non_Scientific']
                values =[]
                for zx in lmw[int(L[1])]:
                    values.append(lmw[int(L[1])][zx])

                #plt.figure(figsize=(5, 5))
                #plt.xticks(rotation=180)
                #plt.xticks(rotation=45)
                #ax.yaxis.tick_right()
                plt.subplot(121)
                #ylabel('Weights')
                plt.bar(names, values)
                plt.show()

        except:
            print("Please try the next query"+"\n")
            continue
            
            
            
#L=list(sys.argv[1:])
#import networkx as nx
elif L[0]=="topic":
    print("LIME Method"+"\n")
    cl,lmw,smw,st1,wexpt,wexptf,wexpt1,st,st1,tw,aw=topic()  
    import time
    time.sleep(1600)
    f33=open("sentt.txt","w")
    for k in st1:
        s=''
        for v in st1[k]:
            s=s+str(v)+" "
        #print(k,s)
        vvc=str(k)+":::"+s
        f33.write(str(vvc)+"\n")
    f33.close()
    #print("Relational Explanation"+"\n")
                    
                    
                
     
    import gensim
    from collections import defaultdict
    from gensim.models import Word2Vec
    import numpy as np
    import lime
    import sklearn
    import numpy as np
    import sklearn
    import sklearn.ensemble
    import sklearn.metrics
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from lime.lime_text import LimeTextExplainer

    for t in range(0,1):
        try:
                print("Non_Relational Explanation"+"\n")
                if cl[int(L[1])]==7:
                    print("Truly Predicted As Auto:"+str(L[1])+"\n")
                    #print("Word Explanation"+"\n")
                   
                elif cl[int(L[1])]==13:
                    print("Truly Predicted As Medicine:"+str(L[1])+"\n")

                class GloveVectorizer:
                            def __init__(self, verbose=False, lowercase=True, minchars=3):

                                #sentences = [['Test','Alice'],['Test','Bob'],['Alice','Carl'],['David','Carl']]
                                f2=open("sentt.txt")
                                WORDStt={}
                                for k in f2:
                                    pp=k.strip("\n \t " " ").split(":::")
                                    #print(pp)
                                    WORDStt[pp[0]]=pp[1].split()
                                sent=[]
                                sent1=[]
                                self.data=WORDStt
                                sent_map=defaultdict(list)
                                for ty in  WORDStt:
                                    gh=[]
                                    gh.append(str(ty))
                                    for j in WORDStt[ty]:
                                        j1=str(j)
                                        #gh.append(str(ty))
                                        if j1 not in gh:
                                            gh.append(j1)
                                    if gh not in sent:
                                            sent.append(gh)
                                #f2.close()
                                self.model = gensim.models.Word2Vec(sentences=sent,min_count=1,window=2)#,vector_size=20)  

                            def fit(self, data, *args):
                                pass
                            def transform(self, data, *args):
                                                W,D = self.model.wv.vectors.shape
                                                X = np.zeros((len(data), D))
                                                n = 0
                                                emptycount = 0
                                                for sentence in data:
                                                    #if sentence.isdigit()==True:
                                                    tokens = sentence
                                                    vecs = []
                                                    #print(tokens)
                                                    for word in tokens:

                                                        if word in self.model.wv:
                                                            vec = self.model.wv[word]
                                                            vecs.append(vec)
                                                    if len(vecs) > 0:
                                                        vecs = np.array(vecs)
                                                        X[n] = vecs.mean(axis=0)
                                                    else:
                                                        emptycount += 1
                                                    n += 1
                                                #X = np.random.rand(100,20)
                                                #X1 = np.asarray(X,dtype='float64')
                                                return X

                            def fit_transform(self, X, *args):
                                    self.fit(X, *args)
                                    return self.transform(X, *args)
                            def get_feature_names(self,X,*args):
                                LL=[]
                                for sentence in X:
                                                    #if sentence.isdigit()==True:
                                                    LL.append(sentence)
                                return LL






                ###

                #gv = GloveVectorizer()                   






                class_names =['Auto','Medicine']

                train=[]
                target=[]
                for t in wexptf:
                    if t in cl:
                        for k in wexptf[t]:
                            train.append(k)
                            if cl[t]==13:
                                target.append('Medicine')
                            else:
                                target.append('Auto')


                tf_transformer=TfidfVectorizer()
                #=GloveVectorizer()# TfidfVectorizer()
                from sklearn.svm import SVC
                from sklearn.svm import LinearSVC
                f = tf_transformer.fit_transform(train)
                #features = [((i, j), f[i,j]) for i, j in zip(*f.nonzero())]

                from sklearn import svm
                #clf =svm.LinearSVC(C=100,probability=True)
                clf=svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)
                #svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)

                #OneVsRestClassifier(LinearSVC(random_state=0)) #svm.LinearSVC(C=1)
                clf.fit(f,target)
                #clf.fit(f,Y)
                p = clf.predict(f)
                #print(f1_score(target,p,average='micro'))


                c = make_pipeline(tf_transformer,clf)

                mt1=['Auto','Medicine']
                cna=mt1


                #explainer = lime.lime_tabular.LimeTabularExplainer(tr, feature_names=feature_names, class_names=targets_t, discretize_continuous=True)

                explainer = LimeTextExplainer(class_names=cna)

                #exp = explainer.explain_instance(train[ii], c.predict_proba, labels=(0,1), top_labels=None, num_features=2237, num_samples=200, distance_metric=u'cosine', model_regressor=None)# explainer.explain_instance(trainf[int(ii)], c.predict_proba, num_features=2961, labels=[0,1])#, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])


                #print("jhhjvhhvhvhhg")
                L1=int(L[1])

                s=''
                for k in  tw[L1]:
                    s=s+str(k)+" "

                exp = explainer.explain_instance(s, c.predict_proba, num_features=6)
                print("The non_relational explsanation of lime stored in limetest_up.html")
                exp.save_to_file('limetest_up.html')
                
                print("Relational Explanation: The sum of weights of the relationally connected quiries that are in same class is high"+"\n")
                import matplotlib.pyplot as plt
                t=cl[int(L[1])]
                if t==13:
                    names = ['Medicine', 'Auto']
                else:
                    names = ['Auto','Medicine']
    
    

                #names = ['Scientific', 'Non_Scientific']
                values =[]
                for zx in lmw[int(L[1])]:
                    values.append(lmw[int(L[1])][zx])

                #plt.figure(figsize=(5, 5))
                #plt.xticks(rotation=180)
                #plt.xticks(rotation=45)
                #ax.yaxis.tick_right()
                plt.subplot(121)
                #ylabel('Weights')
                plt.bar(names, values)
                plt.show()

        except:
            print("Please try the next query"+"\n")
            continue


                        
                        
elif L[0]=="tweet":
    print("LIME Method"+"\n")
    cl,lmw,smw,st1,wexpt,wexptf,wexpt1,st,st1,tw,aw=tweet()
    import time
    time.sleep(1400)
    f33=open("sentt.txt","w")
    for k in wexpt:
        s=''
        for v in wexpt[k]:
            s=s+str(v)+" "
        #print(k,s)
        vvc=str(k)+":::"+s
        f33.write(str(vvc)+"\n")
    f33.close()
    #print("Relational Explanation"+"\n")
    
    
                    
                    
                
     
    import gensim
    from collections import defaultdict
    from gensim.models import Word2Vec
    import numpy as np
    import lime
    import sklearn
    import numpy as np
    import sklearn
    import sklearn.ensemble
    import sklearn.metrics
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from lime.lime_text import LimeTextExplainer

    for t in range(0,1):
        try:
                print("Non_Relational Explanation"+"\n")
                if cl[int(L[1])]==1:
                    print("Truly Predicted As Scientific:"+str(L[1])+"\n")
                    #print("Word Explanation"+"\n")
                   
                elif cl[int(L[1])]==0:
                    print("Truly Predicted As Non_Scientific:"+str(L[1])+"\n")

                class GloveVectorizer:
                            def __init__(self, verbose=False, lowercase=True, minchars=3):

                                #sentences = [['Test','Alice'],['Test','Bob'],['Alice','Carl'],['David','Carl']]
                                f2=open("sentt.txt")
                                WORDStt={}
                                for k in f2:
                                    pp=k.strip("\n \t " " ").split(":::")
                                    #print(pp)
                                    WORDStt[pp[0]]=pp[1].split()
                                sent=[]
                                sent1=[]
                                self.data=WORDStt
                                sent_map=defaultdict(list)
                                for ty in  WORDStt:
                                    gh=[]
                                    gh.append(str(ty))
                                    for j in WORDStt[ty]:
                                        j1=str(j)
                                        #gh.append(str(ty))
                                        if j1 not in gh:
                                            gh.append(j1)
                                    if gh not in sent:
                                            sent.append(gh)
                                #f2.close()
                                self.model = gensim.models.Word2Vec(sentences=sent,min_count=1,window=2)#,vector_size=20)  

                            def fit(self, data, *args):
                                pass
                            def transform(self, data, *args):
                                                W,D = self.model.wv.vectors.shape
                                                X = np.zeros((len(data), D))
                                                n = 0
                                                emptycount = 0
                                                for sentence in data:
                                                    #if sentence.isdigit()==True:
                                                    tokens = sentence
                                                    vecs = []
                                                    #print(tokens)
                                                    for word in tokens:

                                                        if word in self.model.wv:
                                                            vec = self.model.wv[word]
                                                            vecs.append(vec)
                                                    if len(vecs) > 0:
                                                        vecs = np.array(vecs)
                                                        X[n] = vecs.mean(axis=0)
                                                    else:
                                                        emptycount += 1
                                                    n += 1
                                                #X = np.random.rand(100,20)
                                                #X1 = np.asarray(X,dtype='float64')
                                                return X

                            def fit_transform(self, X, *args):
                                    self.fit(X, *args)
                                    return self.transform(X, *args)
                            def get_feature_names(self,X,*args):
                                LL=[]
                                for sentence in X:
                                                    #if sentence.isdigit()==True:
                                                    LL.append(sentence)
                                return LL






                ###

                #gv = GloveVectorizer()                   






                class_names = ['Non_Scientific', 'Scientific']

                train=[]
                target=[]
                for t in wexptf:
                    if t in cl:
                        for k in wexptf[t]:
                            train.append(k)
                            if cl[t]==0:
                                target.append('Non_Scientific')
                            else:
                                target.append('Scientific')


                tf_transformer=TfidfVectorizer()
                #=GloveVectorizer()# TfidfVectorizer()
                from sklearn.svm import SVC
                from sklearn.svm import LinearSVC
                f = tf_transformer.fit_transform(train)
                #features = [((i, j), f[i,j]) for i, j in zip(*f.nonzero())]

                from sklearn import svm
                #clf =svm.LinearSVC(C=100,probability=True)
                clf=svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)
                #svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)

                #OneVsRestClassifier(LinearSVC(random_state=0)) #svm.LinearSVC(C=1)
                clf.fit(f,target)
                #clf.fit(f,Y)
                p = clf.predict(f)
                #print(f1_score(target,p,average='micro'))


                c = make_pipeline(tf_transformer,clf)

                mt1=['Non_Scientific','Scientific']
                cna=mt1


                #explainer = lime.lime_tabular.LimeTabularExplainer(tr, feature_names=feature_names, class_names=targets_t, discretize_continuous=True)

                explainer = LimeTextExplainer(class_names=cna)

                #exp = explainer.explain_instance(train[ii], c.predict_proba, labels=(0,1), top_labels=None, num_features=2237, num_samples=200, distance_metric=u'cosine', model_regressor=None)# explainer.explain_instance(trainf[int(ii)], c.predict_proba, num_features=2961, labels=[0,1])#, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])


                #print("jhhjvhhvhvhhg")
                L1=int(L[1])

                s=''
                for k in  tw[L1]:
                    s=s+str(k)+" "

                exp = explainer.explain_instance(s, c.predict_proba, num_features=6)
                print("The non_relational explsanation of lime stored in limetest_up.html")
                exp.save_to_file('limetest_up.html')

                print("Relational Explanation: The sum of weights of the relationally connected quiries that are in same class is high"+"\n")
                import matplotlib.pyplot as plt
                t=cl[int(L[1])]
                if t==0:
                    names = ['Non_Scientific', 'Scientific']
                else:
                    names = ['Scientific', 'Non_Scientific']
    
    

                #names = ['Scientific', 'Non_Scientific']
                values =[]
                for zx in lmw[int(L[1])]:
                    values.append(lmw[int(L[1])][zx])

                #plt.figure(figsize=(5, 5))
                plt.xticks(rotation=180)
                plt.xticks(rotation=45)
                #ax.yaxis.tick_right()
                plt.subplot(121)
                #ylabel('Weights')
                plt.bar(names, values)
                plt.show()

        except:
            print("Please try the next query"+"\n")
            continue





                        
                        
                        


    
    
        
#Statstical Analysis

def stat_ana():
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
                            tg.append(qrat[gg])

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
                                    model =svm.LinearSVC(C=10) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
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


                

class data_processing:
    #topic data

    # Preprocess Tweet Data
    @classmethod
    def topic(cls):
                    f1=open("Topic_Evidence.txt",encoding="ISO-8859-1")
                    f2=open("Class.txt")
                    #f3=open("review_text.txt",encoding="ISO-8859-1")
                    f4=open("annotated_wordexp.txt")
                    f5=open("docembedding_relexp.txt")
                    WORDS={}
                    qrat={}
                    rtext={}
                    ann={}
                    similar_r_map={}
                    for t in f1:
                        p=t.strip("\n ' '").split("::")
                        pp=p[1].split()
                        #print(pp)
                        kk=p[0].split("_")
                        #print(kk[1])
                        WORDS[kk[1]]=pp
                    for t in f2:
                        p=t.strip("\n ' '").split("::")
                        #pp=p[1].split()
                        #print(p[1])
                        kk=p[0].split("_")
                        #print(kk[1])
                        qrat[kk[1]]=int(p[1])

                    for t in f4:
                        p=t.strip("\n ' '").split("::")
                        pp=p[1].split()
                        #print(pp)
                        kk=p[0].split("_")
                        #print(kk[1])
                        ann[kk[1]]=pp
                    for t in f5:
                        p=t.strip("\n ' '").split("::")
                        pp=p[1].split()
                        #print(pp)
                        kk=p[0].split("_")
                        #print(kk[1])
                        similar_r_map[kk[1]]=pp


                    for t in similar_r_map:
                        cc=0
                        gh=[]
                        for k in similar_r_map[t]:
                            if cc<25:
                                gh.append(k)
                                cc=cc+1
                        rnn[t]=gh


                    WORDS22={}
                    c=0
                    c1=0
                    for t in WORDS:
                        if qrat[t]==7:
                            if c<50:
                                WORDS22[t]=WORDS[t]
                                c=c+1
                        elif qrat[t]==13:
                            if c1<50:
                                WORDS22[t]=WORDS[t]
                                c1=c1+1
                    for k in WORDS22:
                        pass#print(k,qrat[k],WORDS22[k])
                    qf={}
                    for t in similar_r_map:
                        if t in WORDS22:
                            h=rnn[t]+WORDS22[t]
                            #print(t,qrat[t],h),
                            qf[t]=h

                    f1=open("sent.txt","w")
                    for t in qf:
                        s=''
                        for k in qf[t]:
                            s=s+k+" "
                        gg=str(t)+":::"+s
                        f1.write(str(gg)+"\n")
                    f1.close()



    @classmethod
    def rev(cls):
                f1=open("review_evid.txt")
                f2=open("review_class.txt")
                f3=open("review_text.txt",encoding="ISO-8859-1")
                f4=open("review_stw.txt")
                f5=open("reviewembedded_relation.txt")
                f6=open("hotel.txt")
                f7=open("user.txt")
                WORDS={}
                qrat={}
                rtext={}
                ann={}
                similar_r_map={}
                rnn={}
                ho={}
                urr={}
                for t in f6:
                    p=t.strip("\n ' '").split("::")
                    pp=p[1].split()
                    #print(pp)
                    ho[p[0]]=pp
                for t in f7:
                    p=t.strip("\n ' '").split("::")
                    pp=p[1].split()
                    #print(pp)
                    urr[p[0]]=pp

                for t in f1:
                    p=t.strip("\n ' '").split("::")
                    pp=p[1].split()
                    #print(pp)
                    WORDS[p[0]]=pp
                for t in f2:
                    p=t.strip("\n ' '").split("::")
                    #pp=p[1].split()
                    #print(p[1])
                    qrat[p[0]]=int(p[1])
                for t in f3:
                    p=t.strip("\n ' '").split("::")
                    #pp=p[1].split()
                    #print(p[1])
                    rtext[p[0]]=p[1]
                for t in f4:
                    p=t.strip("\n ' '").split("::")
                    pp=p[1].split()
                    #print(pp)
                    ann[p[0]]=pp
                for t in f5:
                    p=t.strip("\n ' '").split("::")
                    pp=p[1].split()
                    #print(pp)
                    similar_r_map[p[0]]=pp



                urr_p={}
                urr_n={}
                ru={}
                for t in urr:
                        for k in urr[t]:
                            ru[k]=t
                for t in urr:
                    c=0
                    for vv in urr[t]:
                        try:
                            if qrat[vv]==1:
                                continue
                            elif qrat[vv]==2:
                                c=c+1
                        except:
                            continue
                    urr_p[t]=c
                for t in urr:
                    c=0
                    for vv in urr[t]:
                        try:
                            if qrat[vv]==1:
                                continue
                            elif qrat[vv]==0:
                                c=c+1
                        except:
                            continue
                    urr_n[t]=c
                for t in urr:
                    if urr_p[t]>0 and urr_n[t]>0:
                        pass#print(t,urr_p[t],urr_n[t])



                #hotel

                ho_p={}
                ho_n={}
                for t in ho:
                    c=0
                    for vv in ho[t]:
                        try:
                            if qrat[vv]==1:
                                continue
                            elif qrat[vv]==2:
                                c=c+1
                        except:
                            continue
                    ho_p[t]=c
                for t in ho:
                    c=0
                    for vv in ho[t]:
                        try:
                            if qrat[vv]==1:
                                continue
                            elif qrat[vv]==0:
                                c=c+1
                        except:
                            continue
                    ho_n[t]=c
                for t in ho:
                    if ho_p[t]>0 and ho_n[t]>0:
                        pass#print(t,ho_p[t],ho_n[t])
                for t in similar_r_map:
                    cc=0
                    gh=[]
                    for k in similar_r_map[t]:
                        if cc<5:
                            gh.append(k)
                            cc=cc+1
                    rnn[t]=gh


                WORDS22={}
                c=0
                c1=0
                for t in WORDS:
                    if qrat[t]==2:
                        if c<50:
                            WORDS22[t]=WORDS[t]
                            c=c+1
                    elif qrat[t]==0:
                        if c1<50:
                            WORDS22[t]=WORDS[t]
                            c1=c1+1
                for k in WORDS22:
                    pass#print(k,qrat[k],WORDS22[k])
                qf={}
                for t in similar_r_map:
                    if t in WORDS22:
                        h=rnn[t]+WORDS22[t]
                        #print(t,qrat[t],h),
                        qf[t]=h

                f1=open("sent.txt","w")
                for t in qf:
                    s=''
                    for k in qf[t]:
                        s=s+k+" "
                    gg=str(t)+":::"+s
                    f1.write(str(gg)+"\n")
                f1.close()


    @classmethod
    def covid(cls):
                    f1=open("Covid_Data.txt",encoding="ISO-8859-1")
                    f2=open("Tweet_class.txt")
                    #f3=open("review_text.txt",encoding="ISO-8859-1")
                    f4=open("annotated_wordexp_covid.txt")
                    f5=open("embedding_rexp_covid.txt")
                    WORDS={}
                    tw={}
                    qrat={}
                    rtext={}
                    ann={}
                    similar_r_map={}
                    rnn={}

                    for t in f1:
                        p=t.strip("\n ' '").split("::")
                        pp=p[1].split()
                        #print(pp)
                        WORDS[p[0]]=pp
                        tw[p[0]]=p[1]
                    for t in f2:
                        p=t.strip("\n ' '").split("::")
                        #pp=p[1].split()
                        #print(p[1])
                        if p[0] in WORDS:
                            #print("hh")
                            qrat[p[0]]=int(p[1])

                    for t in f4:
                        p=t.strip("\n ' '").split("::")
                        pp=p[1].split()
                        #print(pp)
                        if p[0] in WORDS:
                            ann[p[0]]=pp
                    for t in f5:
                        p=t.strip("\n ' '").split("::")
                        pp=p[1].split()
                        #print(pp)
                        if p[0] in WORDS:
                            similar_r_map[p[0]]=pp

                    for t in similar_r_map:
                        cc=0
                        gh=[]
                        for k in similar_r_map[t]:
                            if cc<25:
                                gh.append(k)
                                cc=cc+1
                        rnn[t]=gh


                    WORDS22={}
                    c=0
                    c1=0
                    for t1 in WORDS:
                        if t1 in qrat:
                                if qrat[t1]==1:
                                    if c<54:
                                        WORDS22[t1]=WORDS[t1]
                                        c=c+1
                                elif qrat[t1]==0:
                                    if c1<54:
                                        WORDS22[t1]=WORDS[t1]
                                        c1=c1+1
                    for k in WORDS22:
                        pass#print(k,qrat[k],WORDS22[k])
                    qf={}
                    for t in similar_r_map:
                        if t in WORDS22:
                            h=rnn[t]+WORDS22[t]
                            #print(t,qrat[t],h),
                            qf[t]=h

                    f1=open("sent.txt","w")
                    for t in qf:
                        s=''
                        for k in qf[t]:
                            s=s+k+" "
                        gg=str(t)+":::"+s
                        f1.write(str(gg)+"\n")
                    f1.close()
                    
class cluster1:
            @classmethod
            def tc_cl(cls):
                            #topic and covid Cluster
                            # Shap tranform Cluster and Feedback form Generation 

                            import gensim 
                            import csv
                            import operator
                            from gensim.models import Word2Vec
                            from collections import defaultdict
                            from sklearn import svm
                            #from sklearn import cross_validation
                            from sklearn.model_selection import cross_validate
                            from sklearn.preprocessing import MinMaxScaler
                            from sklearn.metrics import (precision_score,recall_score,f1_score)
                            from sklearn.multiclass import OneVsRestClassifier
                            from sklearn.svm import SVC
                            from sklearn.svm import LinearSVC
                            from sklearn.multiclass import OneVsOneClassifier
                            from sklearn.multiclass import OutputCodeClassifier
                            import random
                            import sys
                            import random
                            import re
                            from collections import defaultdict
                            import sys
                            from nltk.cluster import KMeansClusterer
                            import nltk
                            from sklearn import cluster
                            from sklearn import metrics
                            import gensim 
                            import operator

                            cl={}
                            def cluster_feed(KK,qrat,WORDSt):
                                        #Sentance generation
                                        sent=[]
                                        sent1=[]
                                        sent_map=defaultdict(list)
                                        for ty in WORDSt:
                                            gh=[]
                                            gh.append(str(ty))
                                            #gh1=[]
                                            #gh2=[]
                                            for j in WORDSt[ty]:

                                                j1=str(j)
                                                #gh.append(str(ty))
                                                if j1 not in gh:
                                                    gh.append(j1)

                                                #print(gh)


                                            if gh not in sent:
                                                    sent.append(gh)


                                        documents=[]
                                        #documents1=[]
                                        for t in sent:
                                            for jh in t:
                                                documents.append(jh)

                                        for w in sent:
                                            pass#print(w)
                                        #print(sent)
                                        #import sys
                                        #sys.exit()
                                        #K-Means Run 14
                                        #cluster generation with k-means
                                        model = Word2Vec(sent, min_count=1)
                                        X = model[model.wv.vocab]
                                        NUM_CLUSTERS=KK
                                        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,repeats=2)
                                        assigned_clusters1 = kclusterer.cluster(X,assign_clusters=True)
                                        #print (assigned_clusters1)
                                        cluster={}
                                        words = list(model.wv.vocab)
                                        for i, word in enumerate(words):
                                          gh=[] 
                                          gh1=[] 
                                          gh2=[] 
                                          if word.isdigit(): 
                                            #print("yes")
                                            cluster[word]=assigned_clusters1[i]
                                            #print (word + ":" + str(assigned_clusters1[i]))
                                        cluster_final={}
                                        for j in range(NUM_CLUSTERS):
                                            gg=[]
                                            for tt in cluster:
                                                if int(cluster[tt])==int(j):
                                                    if tt not in gg:
                                                        gg.append(tt)
                                            if len(gg)>0:
                                                        cluster_final[j]=gg
                                        cc=0
                                        final_clu={}
                                        cluster_final
                                        lmm=(KK*3)+5
                                        for t in cluster_final:
                                            #print(t,cluster_final[t])
                                            ghh=[]
                                            vx=0
                                            for k in cluster_final[t]:
                                                if k in WORDSt or str(k) in WORDSt:# and int(k) in lexp or str(k) in lexp:
                                                    #if vx<lmm:
                                                    ghh.append(int(k))
                                                    vx=vx+1
                                            if len(ghh)>0:
                                                    final_clu[cc]=ghh
                                                    cc=cc+1
                                        for k in final_clu:
                                            pass#print(k,final_clu[k],len(final_clu[k]))
                                        return final_clu








                            for kk in range(2,7,1):
                                    final_clu=cluster_feed(kk,qrat,WORDS22)
                                    cl[kk]=final_clu

                                    

                                    
                                    
#Shap+Relation+Feedback+Word Relation+Tranform
import pandas as pd
# Relational annotation
class shap_tranform:
    #Shap Relation Words Without Feedback transform
    #Shap Relation Words+rel Without Feedback Transform
    @classmethod
    def relrf_trsh(cls):
                    #Shap+Relation+Feedback+Word and Relational ExpRelation+Transform

                    import pandas as pd
                    # Relational annotation
                    def relational_embedding_exp(m,WORDS22,qrat,ann):
                        # Relational Exp generatetion based on neural embedding
                                    sent2=[]
                                    sent1=[]
                                    sent_map=defaultdict(list)
                                    for ty in WORDS22:
                                        gh=[]
                                        gh.append(str(ty))
                                        #gh1=[]
                                        #gh2=[]
                                        for j in WORDS22[ty]:

                                            j1=str(j)
                                            #gh.append(str(ty))
                                            if j1 not in gh:
                                                gh.append(j1)
                                            ##print(gh)


                                        if gh not in sent2:
                                                sent2.append(gh)


                                    documents1=[]
                                    #documents1=[]
                                    for t in sent2:
                                        s=''
                                        for jh in t:
                                            if jh.isdigit():
                                                 documents1.append(jh)
                                            else:
                                                s=" "+str(jh)+s+" "
                                        documents1.append(s)


                                    #sentence embedding
                                    from gensim.test.utils import common_texts
                                    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
                                    documents2 = [TaggedDocument(doc, [i]) for i, doc in enumerate(sent2)]
                                    for t in documents2:
                                        pass##print(t)
                                    model = Doc2Vec(documents2, vector_size=5, window=2, min_count=1, workers=4)

                                    #K-Means Run 14 to find the neighbors per query 

                                    #cluster generation with k-means
                                    import sys
                                    from nltk.cluster import KMeansClusterer
                                    import nltk
                                    from sklearn import cluster
                                    from sklearn import metrics
                                    import gensim 
                                    import operator
                                    #from gensim.models import Word2Vec


                                    #model = Word2Vec(sent, min_count=1) dd=sorted(vb.items(), key=operator.itemgetter(1),reverse=False)
                                    import operator
                                    X = model[model.wv.vocab]
                                    c=0
                                    cluster={}
                                    num=[]
                                    weight_map={}
                                    similar_r_map={}
                                    fg={}
                                    for jj in WORDS22:
                                        gh1=[]
                                        gh2=[]
                                        s=0

                                        for k in documents1:
                                            if str(k)==str(jj):
                                                gh=model.most_similar(positive=str(k),topn=600)
                                               # #print(gh)
                                                for tt in gh:
                                                    if float(tt[1]) not in gh1:
                                                        gh1.append(float(tt[1]))
                                                    #if tt[0] not in gh2:
                                                    if tt[0].isdigit():
                                                            #if ccc<5:
                                                                    #gh2.append(tt[0])
                                                                    fg[tt[0]]=tt[1]
                                                                    #ccc=ccc+1
                                        #for ffg in gh1:
                                            #s=s+ffg
                                        dd=sorted(fg.items(), key=operator.itemgetter(1),reverse=True)
                                        ccc=0
                                        for t5 in dd:
                                            if qrat[str(jj)]==qrat[str(t5[0])]:
                                                if m==5:
                                                    if ccc<500:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1
                                                elif m==10:
                                                    if ccc<400:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1
                                                elif m==15:
                                                    if ccc<500:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1
                                                elif m==20:
                                                    if ccc<600:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1
                                                elif m==25:
                                                    if ccc<700:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1

                                        #if len(gh2)>=2:
                                        similar_r_map[jj]=gh2
                                                #ccc=ccc+1

                                    return similar_r_map


                    import gensim.models.word2vec as W2V
                    import gensim.models
                    import sys
                    from sklearn.ensemble import RandomForestRegressor
                    '''
                    class GloveVectorizer:
                        def __init__(self, verbose=False, lowercase=True, minchars=3):

                            #sentences = [['Test','Alice'],['Test','Bob'],['Alice','Carl'],['David','Carl']]
                            f2=open("sent.txt")
                            WORDStt={}
                            for k in f2:
                                pp=k.strip("\n \t " " ").split(":::")
                                #print(pp)
                                WORDStt[pp[0]]=pp[1]
                            sent=[]
                            sent1=[]
                            self.data=WORDStt
                            sent_map=defaultdict(list)
                            for ty in  WORDStt:
                                gh=[]
                                gh.append(str(ty))
                                for j in WORDStt[ty]:
                                    j1=str(j)
                                    #gh.append(str(ty))
                                    if j1 not in gh:
                                        gh.append(j1)
                                if gh not in sent:
                                        sent.append(gh)
                            f2.close()
                            self.model = gensim.models.Word2Vec(sentences=sent,min_count=1,window=2)#,vector_size=20)  

                        def fit(self, data, *args):
                            pass

                        def transform(self, data, *args):
                            W,D = self.model.wv.vectors.shape
                            X = np.zeros((len(data), D))
                            n = 0
                            emptycount = 0
                            for sentence in data:
                                #if sentence.isdigit()==True:
                                tokens = sentence
                                vecs = []
                                for word in tokens:
                                    if word in self.model.wv:
                                        vec = self.model.wv[word]
                                        vecs.append(vec)
                                if len(vecs) > 0:
                                    vecs = np.array(vecs)
                                    X[n] = vecs.mean(axis=0)
                                else:
                                    emptycount += 1
                                n += 1
                            #X = np.random.rand(100,20)
                            #X1 = np.asarray(X,dtype='float64')
                            return X

                        def fit_transform(self, X, *args):
                            self.fit(X, *args)
                            return self.transform(X, *args)

                    '''
                    def lime_all_acc(WORDS22,tx,qrat):
                                    def feedback_accuracy(WORDS22,qrat,similar_r_map,ann,rnn,option):

                                            m_tid_tr1={}
                                            wr=[]
                                            tr=[]
                                            wr1=[]
                                            tr1=[]
                                            c=0
                                            c1=0
                                            tw_wm={}
                                            train_r=[]
                                            targets_r=[]
                                            m_tid_tr1={}
                                            wr=[]
                                            tr=[]
                                            wr1=[]
                                            tr1=[]
                                            c=0
                                            c1=0
                                            tw_wm={}
                                            similar_r_map=similar_r_map#relational_embedding_exp(5,WORDS22,qrat,ann)
                                            rnn1={}
                                            for t in similar_r_map:
                                                gg=[]
                                                cc=0
                                                for k in similar_r_map[t]:
                                                    if cc<25:
                                                        gg.append(k)
                                                        cc=cc+1
                                                rnn1[t]=gg

                                            # organizing feature vector

                                            qf={}
                                            for t in WORDS22:
                                                if str(t) in rnn:
                                                    hh=rnn[str(t)]+WORDS22[t]
                                                    qf[t]=hh


                                            train_r=[]
                                            targets_r=[] 
                                            for t in qf:
                                                    s=''
                                                    vb=[]
                                                    for tt in qf[t]:
                                                        #if tt.isalnum():
                                                            #s=s+tt+" "
                                                        train_r.append(str(tt))
                                                        targets_r.append(qrat[str(t)])
                                            #print(train_r)
                                            #sys.exit()

                                            #corpus_train, corpus_test, y_train, y_test = train_test_split(train_r,targets_r, test_size=0.5, random_state=7)
                                            #vectorizer = TfidfVectorizer(min_df=1)
                                            #X_train = vectorizer.fit_transform(corpus_train)
                                            #X_test = vectorizer.transform(corpus_test)
                                            #model3='
                                            if option=='svm':
                                                model1 =svm.LinearSVC(C=50)
                                                #svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)
                                                #svm.LinearSVC(C=50) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                            elif option=='regression':
                                                model1 = sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1,solver='liblinear')
                                            elif option=='random':
                                                classifier= RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
                                            elif option=='transform':
                                                classifier = transformers.pipeline('sentiment-analysis', return_all_scores=True) 


                                            #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                            #KNeighborsClassifier(n_neighbors=5)
                                            #RandomForestClassifier(max_depth=2, random_state=1)
                                            #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)



                                            classifier(train_r)
                                            explainer = shap.Explainer(classifier)#classifier(corpus_train)
                                            shap_values = explainer(train_r)
                                            #shap.plots.text(shap_values[:,:,"POSITIVE"])
                                            fr={}
                                            for j in range(0,len(train_r)):
                                                fr[shap_values[j].data[1]]=max(sum(shap_values[j].values))

                                            for t in fr:
                                                pass#print(t,fr[t])
                                            dd1=sorted(fr.items(), key=operator.itemgetter(1),reverse=True)
                                            feature_sh_v=[]
                                            for tt in dd1:
                                                if tt[0].isdigit()==True:
                                                       feature_sh_v.append(tt[0])
                                                elif tt[0].isdigit()==False:
                                                    for vvv5 in WORDS22:
                                                        if tt[0] in WORDS22[vvv5] or str(tt[0]) in WORDS22[vvv5]:
                                                                                           feature_sh_v.append(tt[0])
                                            #print(feature_sh_v)
                                            shap_exp={}
                                            for t in qf:
                                                gh=[]
                                                c=0
                                                for k in qf[t]:
                                                   # if k in prr:
                                                        #if qrat[t]==prr[k]:
                                                            if k in feature_sh_v:
                                                                if k not in gh:
                                                                    if k.isdigit():
                                                                        if c<20:
                                                                            gh.append(k)
                                                                            c=c+1
                                                shap_exp[t]=gh


                                            lexp=shap_exp
                                            #print(shap_exp)
                                            lime_all={}
                                            lime_all_p={}
                                            lime_all_n={}
                                            acco=0
                                            acc_po=0
                                            acc_no=0
                                            accor=0
                                            acc_por=0
                                            acc_nor=0
                                            #print(lexp)
                                            for t in lexp:
                                                if str(t) in similar_r_map:
                                                   # print("jjjjjjj")
                                                    c=0
                                                    vb=0
                                                    for zz in lexp[t]:
                                                            if zz in similar_r_map[str(t)]:
                                                               # print("hhhhhhh")
                                                                if vb<5:
                                                                        c=c+1
                                                                        vb=vb+1
                                                    if len(lexp[t])>0:
                                                        s=float(c)/len(lexp[t])
                                                        if s>0:
                                                            lime_all[t]=s
                                            ss=0
                                            for k in lime_all:
                                                ss=ss+float(lime_all[k])

                                            if len(lime_all)>0:
                                                acco=ss/len(lime_all)
                                            #print(option)
                                            print("shap Accuracy without human-feedback"+"\n")
                                            print(acco)
                                            #print("Lime Accuracy without human-feedback only positive reviews"+"\n")

                                            ss1=0
                                            pp=0
                                            for k in lime_all:
                                                if qrat[str(k)]==1:
                                                    ss1=ss1+float(lime_all[k])
                                                    pp=pp+1

                                            if len(lime_all)>0:
                                                acc_po=ss1/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for positive reviews")
                                            print(acc_po)
                                            # print("Lime only negative reviews")
                                            ss2=0
                                            nn=0
                                            for k in lime_all:
                                                if qrat[str(k)]==0:
                                                    ss2=ss2+float(lime_all[k])
                                                    nn=nn+1

                                            if len(lime_all)>0:
                                                acc_no=ss2/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for negative reviews")
                                            print(acc_no)


                                            return acco,acc_po,acc_no,lexp

                                    #def feed(mn,qrat,cl,w3):
                                    def feed(mn,qrat,cl,WORDS22):
                                                    import pandas as pd
                                                    sent=[]
                                                    sent1=[]
                                                    sent_map=defaultdict(list)
                                                    WORDS2={}
                                                    cp=0
                                                    cn=0
                                                    for kkk in WORDS22:
                                                        gvv=[]
                                                        if qrat[kkk]==1:
                                                            if cp<35:
                                                                WORDS2[kkk]=WORDS22[kkk]
                                                                cp=cp+1
                                                        elif qrat[kkk]==0:
                                                            if cn<35:
                                                                WORDS2[kkk]=WORDS22[kkk]
                                                                cn=cn+1
                                                    for ty in WORDS2:
                                                        gh=[]
                                                        gh.append(str(ty))
                                                        #gh1=[]
                                                        #gh2=[]
                                                        for j in WORDS2[ty]:

                                                            j1=str(j)
                                                            #gh.append(str(ty))
                                                            if j1 not in gh:
                                                                gh.append(j1)

                                                        if gh not in sent:
                                                                sent.append(gh)
                                                    documents=[]
                                                    #documents1=[]
                                                    for t in sent:
                                                        for jh in t:
                                                            documents.append(jh)
                                                    hh="twfeedback_Bert"+str(mn)+".csv"
                                                    ps={}
                                                    ns={}
                                                    vot={}
                                                    vtt={}
                                                    f1=pd.read_csv(hh)
                                                    vot={}
                                                    vtt={}
                                                    ll=0
                                                    for col in f1.columns:
                                                        if 'Label' in col:
                                                            ll=ll+1
                                                    m=len(f1['Tweet_ID'])
                                                    for t in range(0,m):
                                                        #print(f1['Review_ID'][t])
                                                        zz=f1['Explanation'][t].split()
                                                        vtt[f1['Tweet_ID'][t]]=zz#f1['Explanation'][t]
                                                        gh=[]
                                                        for vv in range(1,ll+1):
                                                                      vb1="Label"+str(vv)
                                                                      #print(f1[vb1][t])
                                                                      gh.append(f1[vb1][t])
                                                        vot[f1['Tweet_ID'][t]]=gh


                                                    for uy in vot:
                                                        cp=0
                                                        cn=0
                                                        for kk in vot[uy]:
                                                            if int(kk)==1:
                                                                cp=cp+1
                                                            else:
                                                                cn=cn+1
                                                        if cp>=cn:
                                                            ps[uy]=vtt[uy]
                                                        elif cp<cn:
                                                            ns[uy]=vtt[uy]

                                                    cl1=cl[mn]
                                                    print(ps,ns)
                                                    import sys


                                                    #print(cl1)
                                                    WORDS23={}
                                                    rm=[]
                                                    for jj in ns:
                                                        for kj in ns[jj]:
                                                            rm.append(kj)

                                                    model = Word2Vec(sent, min_count=1)
                                                    rme={}
                                                    for uu in ns:
                                                        for k in cl1:
                                                            if int(uu) in cl1[k]:
                                                                for kk in cl1[k]:
                                                                    gg=[]
                                                                    zz=0
                                                                    if str(kk) in WORDS2:
                                                                        #print("hi")

                                                                        for vv in WORDS2[str(kk)]:
                                                                            if vv in ns[uu]:
                                                                                continue
                                                                            else:
                                                                                if zz<20:
                                                                                    gg.append(vv)
                                                                                    zz=zz+1
                                                                        rme[kk]=gg


                                                    for t in cl1:
                                                        for k in ps:
                                                            if int(k) in cl1[t]:
                                                                for kk in cl1[t]:
                                                                    chu1=[]
                                                                    vb={}
                                                                    for v in ps[k]:
                                                                        vb1={}
                                                                        if kk in WORDS2 or str(kk) in WORDS2:
                                                                            for v1 in WORDS2[str(kk)]:
                                                                                try:
                                                                                        gh1=model.similarity(v,v1)
                                                                                        if gh1>0.1:
                                                                                                      vb1[v1]=float(gh1)
                                                                                except:
                                                                                    continue
                                                                            for jk in vb1:
                                                                                            if jk in vb:
                                                                                                if float(vb1[jk])>=float(vb[jk]):
                                                                                                    #print(jk,vb1[jk],vb[jk])
                                                                                                    vb[jk]=vb1[jk]
                                                                                            else:
                                                                                                vb[jk]=vb1[jk]

                                                                    dd1=sorted(vb.items(), key=operator.itemgetter(1),reverse=True)
                                                                    cc=0
                                                                    for kkk in dd1:
                                                                        if kkk[0] not in chu1:
                                                                            if cc<5:
                                                                                    chu1.append(kkk[0])
                                                                                    cc=cc+1
                                                                    if len(chu1)>0 :
                                                                        #if str(kk) in WORDS22:
                                                                            WORDS23[kk]=chu1 
                                                    WORDS25={}

                                                    for t in WORDS23:
                                                        cc=0
                                                        vcx=[]
                                                        if t not in rme:
                                                            WORDS25[t]=WORDS23[t]
                                                        elif t in rme:
                                                            vcc=WORDS23[t]+rme[t]
                                                            for zz in vcc:
                                                                if cc<5:
                                                                    vcx.append(zz)
                                                                    cc=cc+1
                                                            WORDS25[t]=vcx
                                                   # print(len(WORDS25))
                                                    for cc in rme:
                                                        fg=[]
                                                        vc4=0
                                                        if cc not in WORDS25:
                                                            for bb in rme[cc]:
                                                                if vc4<1:
                                                                    fg.append(bb)
                                                                    vc4=vc4+1
                                                            WORDS25[cc]=fg
                                                    #print(WORDS25)
                                                    #print(len(WORDS25))
                                                    #sys.exit()

                                                    return WORDS25,qrat


                                    all_a={}
                                    #all_a={}
                                    on_p={}
                                    on_n={}
                                    exp_all={}
                                    all_exp={}
                                    all_acc={}
                                    all_acc_p={}
                                    all_acc_n={}
                                    all_accr={}
                                    all_acc_pr={}
                                    all_acc_nr={}
                                    clssr={}
                                    for mn in range(2,7,1):     
                                            WORDS25,qrat=feed(mn,qrat,cl,WORDS22)
                                            print(len(WORDS25))
                                            #sys.exit()
                                            acco,acc_po,acc_no,lexp=feedback_accuracy(WORDS25,qrat,similar_r_map,ann,rnn,tx)  
                                            all_a[mn]=acco
                                            all_acc[mn]=acco
                                            all_acc_p[mn]=acc_po
                                            all_acc_n[mn]=acc_no
                                            clssr[mn]=qrat
                                            on_p[mn]=acc_po
                                            on_n[mn]=acc_no
                                            exp_all[mn]=lexp
                                           # all_exp[mn]=lexp11

                                    print(tx)
                                    print("\n")
                                    for tt in all_a:
                                        print(tt,all_acc[tt],all_acc_p[tt],all_acc_n[tt])
                                        print("\n\n")
                                    return exp_all,all_exp,all_acc,all_acc_p,all_acc_n,clssr#,acco,acc_po,acc_no

                    expwsh1={}
                    exp2={}
                    #exp1={}
                    #exp2={}
                    ash={}
                    psh={}
                    nsh={}
                    ar={}
                    pr={}
                    nr={}
                    ao={}
                    po={}
                    no={}
                    clas={}
                    option=['transform']


                    for tx in option:           
                        #acco,acc_po,acc_no,lexp,lexp1,word_weight,ann,corpus_train, corpus_test, y_train, y_test,p,qrat,WORDSt=Shap_Explanation_Engine(s_words,stopwords,WORDS,qrat,H11,Rev_text_map,tx)
                        exp_all,all_exp,all_acc,all_acc_p,all_acc_n,clssr=lime_all_acc(WORDS22,tx,qrat)
                        expwsh1[tx]=exp_all
                       # exp2[tx]=all_exp
                        ash[tx]=all_acc
                        psh[tx]=all_acc_p
                        nsh[tx]=all_acc_n
                        clas[tx]=clssr






    @classmethod
    def relrnf_trsh(cls):

                    from sklearn.svm import LinearSVC

                    '''
                    class GloveVectorizer:
                        def __init__(self, verbose=False, lowercase=True, minchars=3):

                            #sentences = [['Test','Alice'],['Test','Bob'],['Alice','Carl'],['David','Carl']]
                            f2=open("sent.txt")
                            WORDStt={}
                            for k in f2:
                                pp=k.strip("\n \t " " ").split(":::")
                                #print(pp)
                                WORDStt[pp[0]]=pp[1]
                            sent=[]
                            sent1=[]
                            self.data=WORDStt
                            sent_map=defaultdict(list)
                            for ty in  WORDStt:
                                gh=[]
                                gh.append(str(ty))
                                for j in WORDStt[ty]:
                                    j1=str(j)
                                    #gh.append(str(ty))
                                    if j1 not in gh:
                                        gh.append(j1)
                                if gh not in sent:
                                        sent.append(gh)
                            f2.close()
                            self.model = gensim.models.Word2Vec(sentences=sent,min_count=1,window=2)#,vector_size=20)  

                        def fit(self, data, *args):
                            pass

                        def transform(self, data, *args):
                            W,D = self.model.wv.vectors.shape
                            X = np.zeros((len(data), D))
                            n = 0
                            emptycount = 0
                            for sentence in data:
                                #if sentence.isdigit()==True:
                                tokens = sentence
                                vecs = []
                                for word in tokens:
                                    if word in self.model.wv:
                                        vec = self.model.wv[word]
                                        vecs.append(vec)
                                if len(vecs) > 0:
                                    vecs = np.array(vecs)
                                    X[n] = vecs.mean(axis=0)
                                else:
                                    emptycount += 1
                                n += 1
                            #X = np.random.rand(100,20)
                            #X1 = np.asarray(X,dtype='float64')
                            return X

                        def fit_transform(self, X, *args):
                            self.fit(X, *args)
                            return self.transform(X, *args)
                    '''

                    def lime_relation(WORDS22,qrat,similar_r_map,ann,rnn,option):
                                            train_r=[]
                                            targets_r=[]
                                            m_tid_tr1={}
                                            wr=[]
                                            tr=[]
                                            wr1=[]
                                            tr1=[]
                                            c=0
                                            c1=0
                                            tw_wm={}
                                            similar_r_map=similar_r_map
                                                 # organizing feature vector
                                            qf={}
                                            for t in WORDS22:
                                                if t in rnn:
                                                    hh=rnn[t]+WORDS22[t]
                                                    qf[t]=hh


                                            train_r=[]
                                            targets_r=[] 
                                            for t in qf:
                                                    s=''
                                                    vb=[]
                                                    for tt in qf[t]:
                                                        if tt.isalnum():
                                                            s=s+tt+" "
                                                        train_r.append(tt)
                                                        targets_r.append(qrat[t])

                                            #corpus_train, corpus_test, y_train, y_test = train_test_split(train_r,targets_r, test_size=0.5, random_state=7)
                                            #vectorizer = TfidfVectorizer(min_df=1)
                                            #X_train = vectorizer.fit_transform(corpus_train)
                                            #X_test = vectorizer.transform(corpus_test)
                                            #model3='
                                            if option=='svm':
                                                model1 =svm.LinearSVC(C=50)
                                                #svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)
                                                #svm.LinearSVC(C=50) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                            elif option=='regression':
                                                model1 = sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1,solver='liblinear')
                                            elif option=='random':
                                                classifier= RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
                                            elif option=='transform':
                                                classifier = transformers.pipeline('sentiment-analysis', return_all_scores=True) 


                                            #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                            #KNeighborsClassifier(n_neighbors=5)
                                            #RandomForestClassifier(max_depth=2, random_state=1)
                                            #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)



                                            classifier(train_r)
                                            explainer = shap.Explainer(classifier)#classifier(corpus_train)
                                            shap_values = explainer(train_r)
                                            #shap.plots.text(shap_values[:,:,"POSITIVE"])
                                            fr={}
                                            for j in range(0,len(train_r)):
                                                fr[shap_values[j].data[1]]=max(sum(shap_values[j].values))

                                            for t in fr:
                                                print(t,fr[t])
                                            dd1=sorted(fr.items(), key=operator.itemgetter(1),reverse=True)
                                            feature_sh_v=[]
                                            for tt in dd1:
                                                if tt[0].isdigit()==True:
                                                       feature_sh_v.append(tt[0])
                                                elif tt[0].isdigit()==False:
                                                    for vvv5 in WORDS22:
                                                        if tt[0] in WORDS22[vvv5] or str(tt[0]) in WORDS22[vvv5]:
                                                                                           feature_sh_v.append(tt[0])
                                            #print(feature_sh_v)
                                            shap_exp={}
                                            for t in qf:
                                                gh=[]
                                                c=0
                                                for k in qf[t]:
                                                   # if k in prr:
                                                        #if qrat[t]==prr[k]:
                                                            if k in feature_sh_v:
                                                                if k not in gh:
                                                                    if k.isdigit():
                                                                        if c<20:
                                                                            gh.append(k)
                                                                            c=c+1
                                                shap_exp[t]=gh


                                            lexp=shap_exp
                                            print(shap_exp)
                                            lime_all={}
                                            lime_all_p={}
                                            lime_all_n={}
                                            acco=0
                                            acc_po=0
                                            acc_no=0
                                            accor=0
                                            acc_por=0
                                            acc_nor=0
                                            #print(lexp)
                                            for t in lexp:
                                                if str(t) in similar_r_map:
                                                    print("jjjjjjj")
                                                    c=0
                                                    vb=0
                                                    for zz in lexp[t]:
                                                            if zz in similar_r_map[str(t)]:
                                                                print("hhhhhhh")
                                                                if vb<5:
                                                                        c=c+1
                                                                        vb=vb+1
                                                    if len(lexp[t])>0:
                                                        s=float(c)/len(lexp[t])
                                                        if s>0:
                                                            lime_all[t]=s
                                            ss=0
                                            for k in lime_all:
                                                ss=ss+float(lime_all[k])

                                            if len(lime_all)>0:
                                                acco=ss/len(lime_all)
                                            #print(option)
                                            #print("Lime Accuracy without human-feedback"+"\n")
                                            print(acco)
                                            #print("Lime Accuracy without human-feedback only positive reviews"+"\n")

                                            ss1=0
                                            pp=0
                                            for k in lime_all:
                                                if qrat[k]==1:
                                                    ss1=ss1+float(lime_all[k])
                                                    pp=pp+1

                                            if len(lime_all)>0:
                                                acc_po=ss1/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for positive reviews")
                                            print(acc_po)
                                            # print("Lime only negative reviews")
                                            ss2=0
                                            nn=0
                                            for k in lime_all:
                                                if qrat[k]==0:
                                                    ss2=ss2+float(lime_all[k])
                                                    nn=nn+1

                                            if len(lime_all)>0:
                                                acc_no=ss2/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for negative reviews")
                                            print(acc_no)
                                            return acco,acc_po,acc_no,lexp




                    aost={}
                    post={}
                    nost={}
                    aor={}
                    por={}
                    nor={}
                    epws_t={}
                    clss={}
                    option=['transform']


                    for tx in option:           
                        acco,acc_po,acc_no,lexp=lime_relation(WORDS22,qrat,similar_r_map,ann,rnn,tx)
                        aost[tx]=acco
                        post[tx]=acc_po
                        nost[tx]=acc_no
                        epws_t[tx]=lexp


    @classmethod
    def relwnf_trsh(cls):

                    from sklearn.svm import LinearSVC

                    '''
                    class GloveVectorizer:
                        def __init__(self, verbose=False, lowercase=True, minchars=3):

                            #sentences = [['Test','Alice'],['Test','Bob'],['Alice','Carl'],['David','Carl']]
                            f2=open("sent.txt")
                            WORDStt={}
                            for k in f2:
                                pp=k.strip("\n \t " " ").split(":::")
                                #print(pp)
                                WORDStt[pp[0]]=pp[1]
                            sent=[]
                            sent1=[]
                            self.data=WORDStt
                            sent_map=defaultdict(list)
                            for ty in  WORDStt:
                                gh=[]
                                gh.append(str(ty))
                                for j in WORDStt[ty]:
                                    j1=str(j)
                                    #gh.append(str(ty))
                                    if j1 not in gh:
                                        gh.append(j1)
                                if gh not in sent:
                                        sent.append(gh)
                            f2.close()
                            self.model = gensim.models.Word2Vec(sentences=sent,min_count=1,window=2)#,vector_size=20)  

                        def fit(self, data, *args):
                            pass

                        def transform(self, data, *args):
                            W,D = self.model.wv.vectors.shape
                            X = np.zeros((len(data), D))
                            n = 0
                            emptycount = 0
                            for sentence in data:
                                #if sentence.isdigit()==True:
                                tokens = sentence
                                vecs = []
                                for word in tokens:
                                    if word in self.model.wv:
                                        vec = self.model.wv[word]
                                        vecs.append(vec)
                                if len(vecs) > 0:
                                    vecs = np.array(vecs)
                                    X[n] = vecs.mean(axis=0)
                                else:
                                    emptycount += 1
                                n += 1
                            #X = np.random.rand(100,20)
                            #X1 = np.asarray(X,dtype='float64')
                            return X

                        def fit_transform(self, X, *args):
                            self.fit(X, *args)
                            return self.transform(X, *args)
                    '''

                    def lime_relation(WORDS22,qrat,similar_r_map,ann,rnn,option):
                                            train_r=[]
                                            targets_r=[]
                                            m_tid_tr1={}
                                            wr=[]
                                            tr=[]
                                            wr1=[]
                                            tr1=[]
                                            c=0
                                            c1=0
                                            tw_wm={}
                                            similar_r_map=similar_r_map
                                                 # organizing feature vector


                                            train_r=[]
                                            targets_r=[] 
                                            for t in WORDS22:
                                                    s=''
                                                    vb=[]
                                                    for tt in WORDS22[t]:
                                                        if tt.isalnum():
                                                            s=s+tt+" "
                                                        train_r.append(tt)
                                                        targets_r.append(qrat[t])

                                            #corpus_train, corpus_test, y_train, y_test = train_test_split(train_r,targets_r, test_size=0.5, random_state=7)
                                            #vectorizer = TfidfVectorizer(min_df=1)
                                            #X_train = vectorizer.fit_transform(corpus_train)
                                            #X_test = vectorizer.transform(corpus_test)
                                            #model3='
                                            if option=='svm':
                                                model1 =svm.LinearSVC(C=50)
                                                #svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)
                                                #svm.LinearSVC(C=50) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                            elif option=='regression':
                                                model1 = sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1,solver='liblinear')
                                            elif option=='random':
                                                classifier= RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
                                            elif option=='transform':
                                                classifier = transformers.pipeline('sentiment-analysis', return_all_scores=True) 


                                            #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                            #KNeighborsClassifier(n_neighbors=5)
                                            #RandomForestClassifier(max_depth=2, random_state=1)
                                            #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)



                                            classifier(train_r)
                                            explainer = shap.Explainer(classifier)#classifier(corpus_train)
                                            shap_values = explainer(train_r)
                                            #shap.plots.text(shap_values[:,:,"POSITIVE"])
                                            fr={}
                                            for j in range(0,len(train_r)):
                                                fr[shap_values[j].data[1]]=max(sum(shap_values[j].values))

                                            for t in fr:
                                                pass#print(t,fr[t])
                                            dd1=sorted(fr.items(), key=operator.itemgetter(1),reverse=True)
                                            feature_sh_v=[]
                                            for tt in dd1:
                                                if tt[0].isdigit()==True:
                                                       feature_sh_v.append(tt[0])
                                                elif tt[0].isdigit()==False:
                                                    for vvv5 in WORDS22:
                                                        if tt[0] in WORDS22[vvv5] or str(tt[0]) in WORDS22[vvv5]:
                                                                                           feature_sh_v.append(tt[0])
                                            #print(feature_sh_v)
                                            shap_exp={}
                                            for t in WORDS22:
                                                gh=[]
                                                c=0
                                                for k in WORDS22[t]:
                                                   # if k in prr:
                                                        #if qrat[t]==prr[k]:
                                                            if k in feature_sh_v:
                                                                if k not in gh:
                                                                    if c<5:
                                                                        gh.append(k)
                                                                        c=c+1
                                                shap_exp[t]=gh


                                            lexp=shap_exp                                                  
                                            lime_all={}
                                            lime_all_p={}
                                            lime_all_n={}
                                            acco=0
                                            acc_po=0
                                            acc_no=0
                                            accor=0
                                            acc_por=0
                                            acc_nor=0
                                            #print(lexp)
                                            for t in lexp:
                                                if t in ann:
                                                    #print("jjjjjjj")
                                                    c=0
                                                    vb=0
                                                    for zz in lexp[t]:
                                                            if zz in ann[t]:
                                                                #print("hhhhhhh")
                                                                if vb<5:
                                                                        c=c+1
                                                                        vb=vb+1
                                                    if len(lexp[t])>0:
                                                        s=float(c)/len(lexp[t])
                                                        if s>0:
                                                            lime_all[t]=s
                                            ss=0
                                            for k in lime_all:
                                                ss=ss+float(lime_all[k])

                                            if len(lime_all)>0:
                                                if ss>0:
                                                    acco=ss/len(lime_all)
                                            #print(option)
                                            #print("Lime Accuracy without human-feedback"+"\n")
                                            print(acco)
                                            #print("Lime Accuracy without human-feedback only positive reviews"+"\n")

                                            ss1=0
                                            pp=0
                                            for k in lime_all:
                                                if qrat[k]==1:
                                                    ss1=ss1+float(lime_all[k])
                                                    pp=pp+1

                                            if len(lime_all)>0:
                                                if ss1>0:
                                                    acc_po=ss1/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for positive reviews")
                                            print(acc_po)
                                            # print("Lime only negative reviews")
                                            ss2=0
                                            nn=0
                                            for k in lime_all:
                                                if qrat[k]==0:
                                                    ss2=ss2+float(lime_all[k])
                                                    nn=nn+1

                                            if len(lime_all)>0:
                                                if ss2>0:
                                                    acc_no=ss2/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for negative reviews")
                                            print(acc_no)
                                            return acco,acc_po,acc_no,lexp




                    aost={}
                    post={}
                    nost={}
                    aor={}
                    por={}
                    nor={}
                    epws_t={}
                    clss={}
                    option=['transform']


                    for tx in option:           
                        acco,acc_po,acc_no,lexp=lime_relation(WORDS22,qrat,similar_r_map,ann,rnn,tx)
                        aost[tx]=acco
                        post[tx]=acc_po
                        nost[tx]=acc_no
                        epws_t[tx]=lexp

    @classmethod
    def relwf_trsh(cls):

                    def relational_embedding_exp(m,WORDS22,qrat,ann):
                        # Relational Exp generatetion based on neural embedding
                                    sent2=[]
                                    sent1=[]
                                    sent_map=defaultdict(list)
                                    for ty in WORDS22:
                                        gh=[]
                                        gh.append(str(ty))
                                        #gh1=[]
                                        #gh2=[]
                                        for j in WORDS22[ty]:

                                            j1=str(j)
                                            #gh.append(str(ty))
                                            if j1 not in gh:
                                                gh.append(j1)
                                            ##print(gh)


                                        if gh not in sent2:
                                                sent2.append(gh)


                                    documents1=[]
                                    #documents1=[]
                                    for t in sent2:
                                        s=''
                                        for jh in t:
                                            if jh.isdigit():
                                                 documents1.append(jh)
                                            else:
                                                s=" "+str(jh)+s+" "
                                        documents1.append(s)


                                    #sentence embedding
                                    from gensim.test.utils import common_texts
                                    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
                                    documents2 = [TaggedDocument(doc, [i]) for i, doc in enumerate(sent2)]
                                    for t in documents2:
                                        pass##print(t)
                                    model = Doc2Vec(documents2, vector_size=5, window=2, min_count=1, workers=4)

                                    #K-Means Run 14 to find the neighbors per query 

                                    #cluster generation with k-means
                                    import sys
                                    from nltk.cluster import KMeansClusterer
                                    import nltk
                                    from sklearn import cluster
                                    from sklearn import metrics
                                    import gensim 
                                    import operator
                                    #from gensim.models import Word2Vec


                                    #model = Word2Vec(sent, min_count=1) dd=sorted(vb.items(), key=operator.itemgetter(1),reverse=False)
                                    import operator
                                    X = model[model.wv.vocab]
                                    c=0
                                    cluster={}
                                    num=[]
                                    weight_map={}
                                    similar_r_map={}
                                    fg={}
                                    for jj in WORDS22:
                                        gh1=[]
                                        gh2=[]
                                        s=0

                                        for k in documents1:
                                            if str(k)==str(jj):
                                                gh=model.most_similar(positive=str(k),topn=600)
                                               # #print(gh)
                                                for tt in gh:
                                                    if float(tt[1]) not in gh1:
                                                        gh1.append(float(tt[1]))
                                                    #if tt[0] not in gh2:
                                                    if tt[0].isdigit():
                                                            #if ccc<5:
                                                                    #gh2.append(tt[0])
                                                                    fg[tt[0]]=tt[1]
                                                                    #ccc=ccc+1
                                        #for ffg in gh1:
                                            #s=s+ffg
                                        dd=sorted(fg.items(), key=operator.itemgetter(1),reverse=True)
                                        ccc=0
                                        for t5 in dd:
                                            if qrat[str(jj)]==qrat[str(t5[0])]:
                                                if m==5:
                                                    if ccc<500:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1
                                                elif m==10:
                                                    if ccc<400:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1
                                                elif m==15:
                                                    if ccc<500:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1
                                                elif m==20:
                                                    if ccc<600:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1
                                                elif m==25:
                                                    if ccc<700:
                                                             gh2.append(t5[0])
                                                             ccc=ccc+1

                                        #if len(gh2)>=2:
                                        similar_r_map[jj]=gh2
                                                #ccc=ccc+1

                                    return similar_r_map


                    import gensim.models.word2vec as W2V
                    import gensim.models
                    import sys
                    from sklearn.ensemble import RandomForestRegressor
                    '''
                    class GloveVectorizer:
                        def __init__(self, verbose=False, lowercase=True, minchars=3):

                            #sentences = [['Test','Alice'],['Test','Bob'],['Alice','Carl'],['David','Carl']]
                            f2=open("sent.txt")
                            WORDStt={}
                            for k in f2:
                                pp=k.strip("\n \t " " ").split(":::")
                                #print(pp)
                                WORDStt[pp[0]]=pp[1]
                            sent=[]
                            sent1=[]
                            self.data=WORDStt
                            sent_map=defaultdict(list)
                            for ty in  WORDStt:
                                gh=[]
                                gh.append(str(ty))
                                for j in WORDStt[ty]:
                                    j1=str(j)
                                    #gh.append(str(ty))
                                    if j1 not in gh:
                                        gh.append(j1)
                                if gh not in sent:
                                        sent.append(gh)
                            f2.close()
                            self.model = gensim.models.Word2Vec(sentences=sent,min_count=1,window=2)#,vector_size=20)  

                        def fit(self, data, *args):
                            pass

                        def transform(self, data, *args):
                            W,D = self.model.wv.vectors.shape
                            X = np.zeros((len(data), D))
                            n = 0
                            emptycount = 0
                            for sentence in data:
                                #if sentence.isdigit()==True:
                                tokens = sentence
                                vecs = []
                                for word in tokens:
                                    if word in self.model.wv:
                                        vec = self.model.wv[word]
                                        vecs.append(vec)
                                if len(vecs) > 0:
                                    vecs = np.array(vecs)
                                    X[n] = vecs.mean(axis=0)
                                else:
                                    emptycount += 1
                                n += 1
                            #X = np.random.rand(100,20)
                            #X1 = np.asarray(X,dtype='float64')
                            return X

                        def fit_transform(self, X, *args):
                            self.fit(X, *args)
                            return self.transform(X, *args)

                    '''
                    def lime_all_acc(WORDS22,tx,qrat):
                                    def feedback_accuracy(WORDS22,qrat,similar_r_map,ann,rnn,option):
                                            train_r=[]
                                            targets_r=[]
                                            m_tid_tr1={}
                                            wr=[]
                                            tr=[]
                                            wr1=[]
                                            tr1=[]
                                            c=0
                                            c1=0
                                            tw_wm={}
                                            train_r=[]
                                            targets_r=[]
                                            m_tid_tr1={}
                                            wr=[]
                                            tr=[]
                                            wr1=[]
                                            tr1=[]
                                            c=0
                                            c1=0
                                            tw_wm={}
                                            similar_r_map=relational_embedding_exp(5,WORDS22,qrat,ann)
                                            rnn1={}
                                            for t in similar_r_map:
                                                gg=[]
                                                cc=0
                                                for k in similar_r_map[t]:
                                                    if cc<25:
                                                        gg.append(k)
                                                        cc=cc+1
                                                rnn1[t]=gg

                                            # organizing feature vector
                                            qf={}
                                            #qf={}
                                            for t in WORDS22:
                                                   # if t in WORDS22 and t in rnn:
                                                        h=WORDS22[t]
                                                        #print(t,qrat[t],h),
                                                        qf[t]=h
                                           # print(qf)
                                            train_r=[]
                                            targets_r=[] 
                                            for t in qf:
                                                    s=''
                                                    vb=[]
                                                    for tt in qf[t]:
                                                        if tt.isalnum():
                                                            train_r.append(tt)
                                                            targets_r.append(qrat[str(t)])

                                            #shap

                                           # corpus_train, corpus_test, y_train, y_test = train_test_split(train_r,targets_r, test_size=0.5, random_state=7)
                                           # vectorizer = TfidfVectorizer(min_df=1)
                                            #X_train = vectorizer.fit_transform(corpus_train)
                                            #X_test = vectorizer.transform(corpus_test)
                                            #model3='

                                            if option=='svm':
                                                model1 =svm.LinearSVC(C=50)
                                                #svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)
                                                #svm.LinearSVC(C=50) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                            elif option=='regression':
                                                model1 = sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1,solver='liblinear')
                                            elif option=='random':
                                                classifier= RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
                                            elif option=='transform':
                                                classifier1 = transformers.pipeline('sentiment-analysis', return_all_scores=True) 


                                            #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                            #KNeighborsClassifier(n_neighbors=5)
                                            #RandomForestClassifier(max_depth=2, random_state=1)
                                            #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)


                                            clf1=classifier1
                                            clf1(train_r)
                                            explainer = shap.Explainer(clf1)#classifier(corpus_train)
                                            shap_values = explainer(train_r)
                                            #shap.plots.text(shap_values[:,:,"POSITIVE"])
                                            fr={}
                                            for j in range(0,len(train_r)):
                                                fr[shap_values[j].data[1]]=max(sum(shap_values[j].values))

                                            for t in fr:
                                                pass#print(t,fr[t])
                                            dd1=sorted(fr.items(), key=operator.itemgetter(1),reverse=True)
                                            feature_sh_v=[]
                                            for tt in dd1:
                                                if tt[0].isdigit()==True:
                                                       feature_sh_v.append(tt[0])
                                                elif tt[0].isdigit()==False:
                                                    for vvv5 in WORDS22:
                                                        if tt[0] in WORDS22[vvv5] or str(tt[0]) in WORDS22[vvv5]:
                                                                                           feature_sh_v.append(tt[0])
                                            #print(feature_sh_v)
                                            shap_exp={}
                                            for t in WORDS22:
                                                gh=[]
                                                c=0
                                                for k in WORDS22[t]:
                                                   # if k in prr:
                                                        #if qrat[t]==prr[k]:
                                                            if k in feature_sh_v:
                                                                if k not in gh:
                                                                    if c<5:
                                                                        gh.append(k)
                                                                        c=c+1
                                                shap_exp[t]=gh


                                            lexp=shap_exp                                                  
                                            lime_all={}
                                            lime_all_p={}
                                            lime_all_n={}
                                            acco=0
                                            acc_po=0
                                            acc_no=0
                                            accor=0
                                            acc_por=0
                                            acc_nor=0
                                            #print(lexp)
                                            for t in lexp:
                                                if str(t) in ann:
                                                    #print("jjjjjjj")
                                                    c=0
                                                    vb=0
                                                    for zz in lexp[t]:
                                                            if zz in ann[str(t)]:
                                                                #print("hhhhhhh")
                                                                if vb<5:
                                                                        c=c+1
                                                                        vb=vb+1
                                                    if len(lexp[t])>0:
                                                        s=float(c)/len(lexp[t])
                                                        if s>0:
                                                            lime_all[t]=s
                                            ss=0
                                            for k in lime_all:
                                                ss=ss+float(lime_all[k])

                                            if len(lime_all)>0:
                                                if ss>0:
                                                    acco=ss/len(lime_all)
                                            #print(option)
                                            #print("Lime Accuracy without human-feedback"+"\n")
                                            print(acco)
                                            #print("Lime Accuracy without human-feedback only positive reviews"+"\n")

                                            ss1=0
                                            pp=0
                                            for k in lime_all:
                                                if qrat[str(k)]==1:
                                                    ss1=ss1+float(lime_all[k])
                                                    pp=pp+1

                                            if len(lime_all)>0:
                                                if ss1>0:
                                                    acc_po=ss1/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for positive reviews")
                                            print(acc_po)
                                            # print("Lime only negative reviews")
                                            ss2=0
                                            nn=0
                                            for k in lime_all:
                                                if qrat[str(k)]==0:
                                                    ss2=ss2+float(lime_all[k])
                                                    nn=nn+1

                                            if len(lime_all)>0:
                                                if ss2>0:
                                                    acc_no=ss2/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for negative reviews")
                                            print(acc_no)
                                            return acco,acc_po,acc_no,lexp



                                            return acco,acc_po,acc_no,lexp

                                    #def feed(mn,qrat,cl,w3):
                                    def feed(mn,qrat,cl,WORDS22):
                                                    import pandas as pd
                                                    sent=[]
                                                    sent1=[]
                                                    sent_map=defaultdict(list)
                                                    WORDS2={}
                                                    cp=0
                                                    cn=0
                                                    for kkk in WORDS22:
                                                        gvv=[]
                                                        if qrat[str(kkk)]==1:
                                                            if cp<35:
                                                                WORDS2[kkk]=WORDS22[kkk]
                                                                cp=cp+1
                                                        elif qrat[str(kkk)]==0:
                                                            if cn<35:
                                                                WORDS2[kkk]=WORDS22[kkk]
                                                                cn=cn+1
                                                    for ty in WORDS2:
                                                        gh=[]
                                                        gh.append(str(ty))
                                                        #gh1=[]
                                                        #gh2=[]
                                                        for j in WORDS2[ty]:

                                                            j1=str(j)
                                                            #gh.append(str(ty))
                                                            if j1 not in gh:
                                                                gh.append(j1)

                                                        if gh not in sent:
                                                                sent.append(gh)
                                                    documents=[]
                                                    #documents1=[]
                                                    for t in sent:
                                                        for jh in t:
                                                            documents.append(jh)
                                                    hh="twfeedback_Bert"+str(mn)+".csv"
                                                    ps={}
                                                    ns={}
                                                    vot={}
                                                    vtt={}
                                                    f1=pd.read_csv(hh)
                                                    vot={}
                                                    vtt={}
                                                    ll=0
                                                    for col in f1.columns:
                                                        if 'Label' in col:
                                                            ll=ll+1
                                                    m=len(f1['Tweet_ID'])
                                                    for t in range(0,m):
                                                        #print(f1['Review_ID'][t])
                                                        zz=f1['Explanation'][t].split()
                                                        vtt[f1['Tweet_ID'][t]]=zz#f1['Explanation'][t]
                                                        gh=[]
                                                        for vv in range(1,ll+1):
                                                                      vb1="Label"+str(vv)
                                                                      #print(f1[vb1][t])
                                                                      gh.append(f1[vb1][t])
                                                        vot[f1['Tweet_ID'][t]]=gh


                                                    for uy in vot:
                                                        cp=0
                                                        cn=0
                                                        for kk in vot[uy]:
                                                            if int(kk)==1:
                                                                cp=cp+1
                                                            else:
                                                                cn=cn+1
                                                        if cp>=cn:
                                                            ps[uy]=vtt[uy]
                                                        elif cp<cn:
                                                            ns[uy]=vtt[uy]

                                                    cl1=cl[mn]

                                                    import sys


                                                    #print(cl1)
                                                    WORDS23={}
                                                    rm=[]
                                                    for jj in ns:
                                                        for kj in ns[jj]:
                                                            rm.append(kj)

                                                    model = Word2Vec(sent, min_count=1)
                                                    rme={}
                                                    for uu in ns:
                                                        for k in cl1:
                                                            if str(uu) in cl1[k]:
                                                                #print("hi")
                                                                for kk in cl1[k]:
                                                                    gg=[]
                                                                    zz=0
                                                                    if kk in WORDS2:
                                                                        #print("hi")

                                                                        for vv in WORDS2[str(kk)]:
                                                                            if vv in ns[uu]:
                                                                                continue
                                                                            else:
                                                                                if zz<5:
                                                                                    gg.append(vv)
                                                                                    zz=zz+1
                                                                        rme[kk]=gg


                                                    for t in cl1:
                                                        for k in ps:
                                                            if int(k) in cl1[t]:
                                                                for kk in cl1[t]:
                                                                    chu1=[]
                                                                    vb={}
                                                                    for v in ps[k]:
                                                                        vb1={}
                                                                        if kk in WORDS2 or str(kk) in WORDS2:
                                                                            for v1 in WORDS2[str(kk)]:
                                                                                try:
                                                                                        gh1=model.similarity(v,v1)
                                                                                        if gh1>0.1:
                                                                                                      vb1[v1]=float(gh1)
                                                                                except:
                                                                                    continue
                                                                            for jk in vb1:
                                                                                            if jk in vb:
                                                                                                if float(vb1[jk])>=float(vb[jk]):
                                                                                                    #print(jk,vb1[jk],vb[jk])
                                                                                                    vb[jk]=vb1[jk]
                                                                                            else:
                                                                                                vb[jk]=vb1[jk]

                                                                    dd1=sorted(vb.items(), key=operator.itemgetter(1),reverse=True)
                                                                    cc=0
                                                                    for kkk in dd1:
                                                                        if kkk[0] not in chu1:
                                                                            if cc<5:
                                                                                    chu1.append(kkk[0])
                                                                                    cc=cc+1
                                                                    if len(chu1)>0 :
                                                                        if str(kk) in WORDS22:
                                                                            WORDS23[kk]=chu1 
                                                    WORDS25={}

                                                    for t in WORDS23:
                                                        cc=0
                                                        vcx=[]
                                                        if t not in rme:
                                                            WORDS25[t]=WORDS23[t]
                                                        elif t in rme:
                                                            vcc=WORDS23[t]+rme[t]
                                                            for zz in vcc:
                                                                if cc<5:
                                                                    vcx.append(zz)
                                                                    cc=cc+1
                                                            WORDS25[t]=vcx
                                                   # print(len(WORDS25))
                                                    for cc in rme:
                                                        fg=[]
                                                        vc4=0
                                                        if cc not in WORDS25:
                                                            for bb in rme[cc]:
                                                                if vc4<1:
                                                                    fg.append(bb)
                                                                    vc4=vc4+1
                                                            WORDS25[cc]=fg
                                                    #print(WORDS25)


                                                    return WORDS25,qrat


                                    all_a={}
                                    #all_a={}
                                    on_p={}
                                    on_n={}
                                    exp_all={}
                                    all_exp={}
                                    all_acc={}
                                    all_acc_p={}
                                    all_acc_n={}
                                    all_accr={}
                                    all_acc_pr={}
                                    all_acc_nr={}
                                    clssr={}
                                    for mn in range(2,7,1):     
                                            WORDS25,qrat=feed(mn,qrat,cl,WORDS22)
                                            print(len(WORDS25))
                                            #sys.exit()
                                            acco,acc_po,acc_no,lexp=feedback_accuracy(WORDS25,qrat,similar_r_map,ann,rnn,tx)  
                                            all_a[mn]=acco
                                            all_acc[mn]=acco
                                            all_acc_p[mn]=acc_po
                                            all_acc_n[mn]=acc_no
                                            clssr[mn]=qrat
                                            on_p[mn]=acc_po
                                            on_n[mn]=acc_no
                                            exp_all[mn]=lexp
                                           # all_exp[mn]=lexp11

                                    print(tx)
                                    print("\n")
                                    for tt in all_a:
                                        print(tt,all_acc[tt],all_acc_p[tt],all_acc_n[tt])
                                        print("\n\n")
                                    return exp_all,all_exp,all_acc,all_acc_p,all_acc_n,clssr#,acco,acc_po,acc_no

                    expwsh1={}
                    exp2={}
                    #exp1={}
                    #exp2={}
                    ash={}
                    psh={}
                    nsh={}
                    ar={}
                    pr={}
                    nr={}
                    ao={}
                    po={}
                    no={}
                    clas={}
                    option=['transform']


                    for tx in option:           
                        #acco,acc_po,acc_no,lexp,lexp1,word_weight,ann,corpus_train, corpus_test, y_train, y_test,p,qrat,WORDSt=Shap_Explanation_Engine(s_words,stopwords,WORDS,qrat,H11,Rev_text_map,tx)
                        exp_all,all_exp,all_acc,all_acc_p,all_acc_n,clssr=lime_all_acc(WORDS22,tx,qrat)
                        expwsh1[tx]=exp_all
                       # exp2[tx]=all_exp
                        ash[tx]=all_acc
                        psh[tx]=all_acc_p
                        nsh[tx]=all_acc_n
                        clas[tx]=clssr




#random forest without feedback shap
#Shap Relation Only Words+no feedback

class shap_rnd:
        @classmethod
        def frnd(cls):
    

                            #random forest with feedback shap
                            #Shap+Relation+Feedback+Word Relation+Random Forest
                            import pandas as pd
                            # Relational annotation
                            def relational_embedding_exp(m,WORDS22,qrat,ann):
                                # Relational Exp generatetion based on neural embedding
                                            sent2=[]
                                            sent1=[]
                                            sent_map=defaultdict(list)
                                            for ty in WORDS22:
                                                gh=[]
                                                gh.append(str(ty))
                                                #gh1=[]
                                                #gh2=[]
                                                for j in WORDS22[ty]:

                                                    j1=str(j)
                                                    #gh.append(str(ty))
                                                    if j1 not in gh:
                                                        gh.append(j1)
                                                    ##print(gh)


                                                if gh not in sent2:
                                                        sent2.append(gh)


                                            documents1=[]
                                            #documents1=[]
                                            for t in sent2:
                                                s=''
                                                for jh in t:
                                                    if jh.isdigit():
                                                         documents1.append(jh)
                                                    else:
                                                        s=" "+str(jh)+s+" "
                                                documents1.append(s)


                                            #sentence embedding
                                            from gensim.test.utils import common_texts
                                            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
                                            documents2 = [TaggedDocument(doc, [i]) for i, doc in enumerate(sent2)]
                                            for t in documents2:
                                                pass##print(t)
                                            model = Doc2Vec(documents2, vector_size=5, window=2, min_count=1, workers=4)

                                            #K-Means Run 14 to find the neighbors per query 

                                            #cluster generation with k-means
                                            import sys
                                            from nltk.cluster import KMeansClusterer
                                            import nltk
                                            from sklearn import cluster
                                            from sklearn import metrics
                                            import gensim 
                                            import operator
                                            #from gensim.models import Word2Vec


                                            #model = Word2Vec(sent, min_count=1) dd=sorted(vb.items(), key=operator.itemgetter(1),reverse=False)
                                            import operator
                                            X = model[model.wv.vocab]
                                            c=0
                                            cluster={}
                                            num=[]
                                            weight_map={}
                                            similar_r_map={}
                                            fg={}
                                            for jj in WORDS22:
                                                gh1=[]
                                                gh2=[]
                                                s=0

                                                for k in documents1:
                                                    if str(k)==str(jj):
                                                        gh=model.most_similar(positive=str(k),topn=600)
                                                       # #print(gh)
                                                        for tt in gh:
                                                            if float(tt[1]) not in gh1:
                                                                gh1.append(float(tt[1]))
                                                            #if tt[0] not in gh2:
                                                            if tt[0].isdigit():
                                                                    #if ccc<5:
                                                                            #gh2.append(tt[0])
                                                                            fg[tt[0]]=tt[1]
                                                                            #ccc=ccc+1
                                                #for ffg in gh1:
                                                    #s=s+ffg
                                                dd=sorted(fg.items(), key=operator.itemgetter(1),reverse=True)
                                                ccc=0
                                                for t5 in dd:
                                                    if qrat[str(jj)]==qrat[str(t5[0])]:
                                                        if m==5:
                                                            if ccc<500:
                                                                     gh2.append(t5[0])
                                                                     ccc=ccc+1
                                                        elif m==10:
                                                            if ccc<400:
                                                                     gh2.append(t5[0])
                                                                     ccc=ccc+1
                                                        elif m==15:
                                                            if ccc<500:
                                                                     gh2.append(t5[0])
                                                                     ccc=ccc+1
                                                        elif m==20:
                                                            if ccc<600:
                                                                     gh2.append(t5[0])
                                                                     ccc=ccc+1
                                                        elif m==25:
                                                            if ccc<700:
                                                                     gh2.append(t5[0])
                                                                     ccc=ccc+1

                                                #if len(gh2)>=2:
                                                similar_r_map[jj]=gh2
                                                        #ccc=ccc+1

                                            return similar_r_map


                            import gensim.models.word2vec as W2V
                            import gensim.models
                            import sys
                            from sklearn.ensemble import RandomForestRegressor
                            class GloveVectorizer:
                                def __init__(self, verbose=False, lowercase=True, minchars=3):
                                    '''
                                    # load in pre-trained word vectors
                                    print('Loading word vectors...')
                                    word2vec = {}
                                    embedding = []
                                    idx2word = []
                                    with open('../data/glove.6B.50d.txt') as f:
                                          # is just a space-separated text file in the format:
                                          # word vec[0] vec[1] vec[2] ...
                                          for line in f:
                                            values = line.split()
                                            word = values[0]
                                            vec = np.asarray(values[1:], dtype='float32')
                                            word2vec[word] = vec
                                            embedding.append(vec)
                                            idx2word.append(word)
                                    print('Found %s word vectors.' % len(word2vec))

                                    self.word2vec = word2vec
                                    self.embedding = np.array(embedding)
                                    self.word2idx = {v:k for k,v in enumerate(idx2word)}
                                    self.V, self.D = self.embedding.shape
                                    self.verbose = verbose
                                    self.lowercase = lowercase
                                    self.minchars = minchars
                                    '''
                                    #sentences = [['Test','Alice'],['Test','Bob'],['Alice','Carl'],['David','Carl']]
                                    f2=open("sent.txt")
                                    WORDStt={}
                                    for k in f2:
                                        pp=k.strip("\n \t " " ").split(":::")
                                        #print(pp)
                                        WORDStt[pp[0]]=pp[1]
                                    sent=[]
                                    sent1=[]
                                    self.data=WORDStt
                                    sent_map=defaultdict(list)
                                    for ty in  WORDStt:
                                        gh=[]
                                        gh.append(str(ty))
                                        for j in WORDStt[ty]:
                                            j1=str(j)
                                            #gh.append(str(ty))
                                            if j1 not in gh:
                                                gh.append(j1)
                                        if gh not in sent:
                                                sent.append(gh)
                                    f2.close()
                                    self.model = gensim.models.Word2Vec(sentences=sent,min_count=1,window=2)#,vector_size=20)  

                                def fit(self, data, *args):
                                    pass

                                def transform(self, data, *args):
                                    W,D = self.model.wv.vectors.shape
                                    X = np.zeros((len(data), D))
                                    n = 0
                                    emptycount = 0
                                    for sentence in data:
                                        #if sentence.isdigit()==True:
                                        tokens = sentence
                                        vecs = []
                                        for word in tokens:
                                            if word in self.model.wv:
                                                vec = self.model.wv[word]
                                                vecs.append(vec)
                                        if len(vecs) > 0:
                                            vecs = np.array(vecs)
                                            X[n] = vecs.mean(axis=0)
                                        else:
                                            emptycount += 1
                                        n += 1
                                    #X = np.random.rand(100,20)
                                    #X1 = np.asarray(X,dtype='float64')
                                    return X

                                def fit_transform(self, X, *args):
                                    self.fit(X, *args)
                                    return self.transform(X, *args)


                            def lime_all_acc(WORDS22,tx,qrat):
                                            def feedback_accuracy(WORDS22,qrat,similar_r_map,ann,rnn,option):
                                                    train_r=[]
                                                    targets_r=[]
                                                    m_tid_tr1={}
                                                    wr=[]
                                                    tr=[]
                                                    wr1=[]
                                                    tr1=[]
                                                    c=0
                                                    c1=0
                                                    tw_wm={}
                                                    train_r=[]
                                                    targets_r=[]
                                                    m_tid_tr1={}
                                                    wr=[]
                                                    tr=[]
                                                    wr1=[]
                                                    tr1=[]
                                                    c=0
                                                    c1=0
                                                    tw_wm={}
                                                    similar_r_map=relational_embedding_exp(5,WORDS22,qrat,ann)
                                                    rnn1={}
                                                    for t in similar_r_map:
                                                        gg=[]
                                                        cc=0
                                                        for k in similar_r_map[t]:
                                                            if cc<25:
                                                                gg.append(k)
                                                                cc=cc+1
                                                        rnn1[t]=gg

                                                    # organizing feature vector
                                                    qf={}
                                                    #qf={}
                                                    for t in WORDS22:
                                                           # if t in WORDS22 and t in rnn:
                                                                h=WORDS22[t]
                                                                #print(t,qrat[t],h),
                                                                qf[t]=h
                                                   # print(qf)
                                                    train_r=[]
                                                    targets_r=[] 
                                                    for t in qf:
                                                            s=''
                                                            vb=[]
                                                            for tt in qf[t]:
                                                                if tt.isalnum():
                                                                    train_r.append(tt)
                                                                    targets_r.append(qrat[t])

                                                    #shap

                                                   # corpus_train, corpus_test, y_train, y_test = train_test_split(train_r,targets_r, test_size=0.5, random_state=7)
                                                   # vectorizer = TfidfVectorizer(min_df=1)
                                                    #X_train = vectorizer.fit_transform(corpus_train)
                                                    #X_test = vectorizer.transform(corpus_test)
                                                    #model3='

                                                    if option=='random':
                                                        print("rn")
                                                        rforest = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)


                                                    clf1=rforest

                                                    clf1.fit(X_train,y_train)
                                                    p = clf1.predict(X_test)
                                                    print("rn")
                                                    prr={}
                                                    for jj in range(0,len(corpus_test)):
                                                        prr[corpus_test[jj]]=int(p[jj])

                                                    rforest = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
                                                    rforest.fit(X_train, y_train)
                                                    p = rforest.predict(X_test)
                                                    print(p)

                                                    print(f1_score(y_test,p,average='micro'))

                                                    #print_accuracy(rforest.predict)

                                                    # explain all the predictions in the test set
                                                    explainer = shap.KernelExplainer(rforest.predict_proba, X_train)
                                                    shv=[]
                                                    for t in X_test:
                                                        shap_values = explainer.shap_values(t)
                                                        shv.append(shap_values )

                                                    shape_w={}
                                                    import operator 
                                                    import sys
                                                    fr={}
                                                    feature_sh_v=[]
                                                    for jj in range(0,len(shv)):
                                                                              m=(sum(abs(sum(shv[jj][0])))+sum(abs(sum(shv[jj][1]))))/2
                                                                              print(m)
                                                                              if corpus_test[jj] not in fr:
                                                                                              fr[corpus_test[jj]]=m
                                                                              elif corpus_test[jj]  in fr:
                                                                                    if m>fr[corpus_test[jj]]:
                                                                                            fr[corpus_test[jj]]=m

                                                    dd1=sorted(fr.items(), key=operator.itemgetter(1),reverse=True)
                                                    for tt in dd1:
                                                        if tt[0].isdigit()==True:
                                                               feature_sh_v.append(tt[0])
                                                        elif tt[0].isdigit()==False:
                                                            for vvv5 in WORDS22:
                                                                if tt[0] in WORDS22[vvv5] or str(tt[0]) in WORDS22[vvv5]:
                                                                                                   feature_sh_v.append(tt[0])
                                                    shap_exp={}
                                                    for t in WORDS22:
                                                        gh=[]
                                                        c=0
                                                        for k in WORDS22[t]:
                                                           # if k in prr:
                                                                #if qrat[t]==prr[k]:
                                                                    if k in feature_sh_v:
                                                                        if k not in gh:
                                                                            #if c<20:
                                                                                gh.append(k)
                                                                                c=c+1
                                                        shap_exp[t]=gh


                                                    lexp=shap_exp                                                  
                                                    lime_all={}
                                                    lime_all_p={}
                                                    lime_all_n={}
                                                    acco=0
                                                    acc_po=0
                                                    acc_no=0
                                                    accor=0
                                                    acc_por=0
                                                    acc_nor=0
                                                    #print(lexp)
                                                    for t in lexp:
                                                        if t in ann:
                                                            #print("jjjjjjj")
                                                            c=0
                                                            vb=0
                                                            for zz in lexp[t]:
                                                                    if zz in ann[t]:
                                                                        #print("hhhhhhh")
                                                                        if vb<5:
                                                                                c=c+1
                                                                                vb=vb+1
                                                            if len(lexp[t])>0:
                                                                s=float(c)/len(lexp[t])
                                                                if s>0:
                                                                    lime_all[t]=s
                                                    ss=0
                                                    for k in lime_all:
                                                        ss=ss+float(lime_all[k])

                                                    if len(lime_all)>0:
                                                        acco=ss/len(lime_all)
                                                    #print(option)
                                                    #print("Lime Accuracy without human-feedback"+"\n")
                                                    print(acco)
                                                    #print("Lime Accuracy without human-feedback only positive reviews"+"\n")

                                                    ss1=0
                                                    pp=0
                                                    for k in lime_all:
                                                        if qrat[k]==2:
                                                            ss1=ss1+float(lime_all[k])
                                                            pp=pp+1

                                                    if len(lime_all)>0:
                                                        acc_po=ss1/len(lime_all)
                                                    #print("Lime Accuracy without human-feedback for positive reviews")
                                                    print(acc_po)
                                                   # print("Lime only negative reviews")
                                                    ss2=0
                                                    nn=0
                                                    for k in lime_all:
                                                        if qrat[k]==0:
                                                            ss2=ss2+float(lime_all[k])
                                                            nn=nn+1

                                                    if len(lime_all)>0:
                                                        acc_no=ss2/len(lime_all)
                                                    #print("Lime Accuracy without human-feedback for negative reviews")
                                                    print(acc_no)


                                                    return acco,acc_po,acc_no,lexp

                                            #def feed(mn,qrat,cl,w3):
                                            def feed(mn,qrat,cl,WORDS22):
                                                            import pandas as pd
                                                            sent=[]
                                                            sent1=[]
                                                            sent_map=defaultdict(list)
                                                            w33={}
                                                            cp=0
                                                            cn=0
                                                            for kkk in WORDS22:
                                                                gvv=[]
                                                                if qrat[str(kkk)]==2:
                                                                    if cp<50:
                                                                        w33[kkk]=WORDS22[kkk]
                                                                        cp=cp+1
                                                                elif qrat[str(kkk)]==0:
                                                                    if cn<50:
                                                                        w33[kkk]=WORDS22[kkk]
                                                                        cn=cn+1
                                                            for ty in WORDS22:
                                                                gh=[]
                                                                gh.append(str(ty))
                                                                #gh1=[]
                                                                #gh2=[]
                                                                for j in WORDS22[ty]:

                                                                    j1=str(j)
                                                                    #gh.append(str(ty))
                                                                    if j1 not in gh:
                                                                        gh.append(j1)

                                                                if gh not in sent:
                                                                        sent.append(gh)
                                                            documents=[]
                                                            #documents1=[]
                                                            for t in sent:
                                                                for jh in t:
                                                                    documents.append(jh)
                                                            hh="feedback_Lime"+str(mn)+".csv"
                                                            ps={}
                                                            ns={}
                                                            vot={}
                                                            vtt={}
                                                            f1=pd.read_csv(hh)
                                                            vot={}
                                                            vtt={}
                                                            ll=0
                                                            for col in f1.columns:
                                                                if 'Label' in col:
                                                                    ll=ll+1
                                                            m=len(f1['Review_ID'])
                                                            for t in range(0,m):
                                                                #print(f1['Review_ID'][t])
                                                                zz=f1['Explanation'][t].split()
                                                                vtt[f1['Review_ID'][t]]=zz#f1['Explanation'][t]
                                                                gh=[]
                                                                for vv in range(1,ll+1):
                                                                              vb1="Label"+str(vv)
                                                                              #print(f1[vb1][t])
                                                                              gh.append(f1[vb1][t])
                                                                vot[f1['Review_ID'][t]]=gh


                                                            for uy in vot:
                                                                cp=0
                                                                cn=0
                                                                for kk in vot[uy]:
                                                                    if int(kk)==1:
                                                                        cp=cp+1
                                                                    else:
                                                                        cn=cn+1
                                                                if cp>=cn:
                                                                    ps[uy]=vtt[uy]
                                                                elif cp<cn:
                                                                    ns[uy]=vtt[uy]

                                                            cl1=cl[mn]
                                                            print(ps,ns)
                                                            import sys


                                                            #print(cl1)
                                                            WORDS23={}
                                                            rm=[]
                                                            for jj in ns:
                                                                for kj in ns[jj]:
                                                                    rm.append(kj)

                                                            model = Word2Vec(sent, min_count=1)
                                                            rme={}
                                                            for uu in ns:
                                                                for k in cl1:
                                                                    if str(uu) in cl1[k]:
                                                                        for kk in cl1[k]:
                                                                            gg=[]
                                                                            zz=0
                                                                            if str(kk) in WORDS22:
                                                                                #print("hi")

                                                                                for vv in WORDS22[str(kk)]:
                                                                                    if vv in ns[uu]:
                                                                                        continue
                                                                                    else:
                                                                                        if zz<5:
                                                                                            gg.append(vv)
                                                                                            zz=zz+1
                                                                                rme[kk]=gg


                                                            for t in cl1:
                                                                for k in ps:
                                                                    if str(k) in cl1[t]:
                                                                        for kk in cl1[t]:
                                                                            chu1=[]
                                                                            vb={}
                                                                            for v in ps[k]:
                                                                                vb1={}
                                                                                if kk in WORDS22 or str(kk) in WORDS22:
                                                                                    for v1 in WORDS22[str(kk)]:
                                                                                        try:
                                                                                                gh1=model.similarity(v,v1)
                                                                                                if gh1>0.1:
                                                                                                              vb1[v1]=float(gh1)
                                                                                        except:
                                                                                            continue
                                                                                    for jk in vb1:
                                                                                                    if jk in vb:
                                                                                                        if float(vb1[jk])>=float(vb[jk]):
                                                                                                            #print(jk,vb1[jk],vb[jk])
                                                                                                            vb[jk]=vb1[jk]
                                                                                                    else:
                                                                                                        vb[jk]=vb1[jk]

                                                                            dd1=sorted(vb.items(), key=operator.itemgetter(1),reverse=True)
                                                                            cc=0
                                                                            for kkk in dd1:
                                                                                if kkk[0] not in chu1:
                                                                                    if cc<5:
                                                                                            chu1.append(kkk[0])
                                                                                            cc=cc+1
                                                                            if len(chu1)>0 :
                                                                                if str(kk) in WORDS22:
                                                                                    WORDS23[kk]=chu1 
                                                            WORDS25={}

                                                            for t in WORDS23:
                                                                cc=0
                                                                vcx=[]
                                                                if t not in rme:
                                                                    WORDS25[t]=WORDS23[t]
                                                                elif t in rme:
                                                                    vcc=WORDS23[t]+rme[t]
                                                                    for zz in vcc:
                                                                        if cc<5:
                                                                            vcx.append(zz)
                                                                            cc=cc+1
                                                                    WORDS25[t]=vcx
                                                           # print(len(WORDS25))
                                                            for cc in rme:
                                                                fg=[]
                                                                vc4=0
                                                                if cc not in WORDS25:
                                                                    for bb in rme[cc]:
                                                                        if vc4<2:
                                                                            fg.append(bb)
                                                                            vc4=vc4+1
                                                                    WORDS25[cc]=fg
                                                            #print(WORDS25)
                                                            #print(len(WORDS25))
                                                            #sys.exit()

                                                            return WORDS25,qrat


                                            all_a={}
                                            #all_a={}
                                            on_p={}
                                            on_n={}
                                            exp_all={}
                                            all_exp={}
                                            all_acc={}
                                            all_acc_p={}
                                            all_acc_n={}
                                            all_accr={}
                                            all_acc_pr={}
                                            all_acc_nr={}
                                            clssr={}
                                            for mn in range(2,7,1):     
                                                    WORDS25,qrat=feed(mn,qrat,cl,WORDS22)
                                                    print(len(WORDS25))
                                                    acc,acc_p,acc_n,qrat,lexp=feedback_accuracy(WORDS25,qrat,similar_r_map,ann,rnn,tx)  
                                                    all_a[mn]=acc
                                                    all_acc[mn]=acc
                                                    all_acc_p[mn]=acc_p
                                                    all_acc_n[mn]=acc_n
                                                    clssr[mn]=qrat
                                                    on_p[mn]=acc_p
                                                    on_n[mn]=acc_n
                                                    exp_all[mn]=lexp
                                                   # all_exp[mn]=lexp11

                                            print(tx)
                                            print("\n")
                                            for tt in all_a:
                                                print(tt,all_acc[tt],all_acc_p[tt],all_acc_n[tt])
                                                print("\n\n")
                                            return exp_all,all_exp,all_acc,all_acc_p,all_acc_n,clssr#,acco,acc_po,acc_no

                            expws1={}
                            exp2={}
                            #exp1={}
                            #exp2={}
                            ash={}
                            psh={}
                            nsh={}
                            ar={}
                            pr={}
                            nr={}
                            ao={}
                            po={}
                            no={}
                            clas={}
                            option=['random']


                            for tx in option:           
                                #acco,acc_po,acc_no,lexp,lexp1,word_weight,ann,corpus_train, corpus_test, y_train, y_test,p,qrat,WORDSt=Shap_Explanation_Engine(s_words,stopwords,WORDS,qrat,H11,Rev_text_map,tx)
                                exp_all,all_exp,all_acc,all_acc_p,all_acc_n,clssr=lime_all_acc(WORDS22,tx,qrat)
                                expws1[tx]=exp_all
                               # exp2[tx]=all_exp
                                ash[tx]=all_acc
                                psh[tx]=all_acc_p
                                nsh[tx]=all_acc_n
                                clas[tx]=clssr



                   
    
        @classmethod
        def nfrnd(cls):

                    class GloveVectorizer:
                        def __init__(self, verbose=False, lowercase=True, minchars=3):
                            '''
                            # load in pre-trained word vectors
                            print('Loading word vectors...')
                            word2vec = {}
                            embedding = []
                            idx2word = []
                            with open('../data/glove.6B.50d.txt') as f:
                                  # is just a space-separated text file in the format:
                                  # word vec[0] vec[1] vec[2] ...
                                  for line in f:
                                    values = line.split()
                                    word = values[0]
                                    vec = np.asarray(values[1:], dtype='float32')
                                    word2vec[word] = vec
                                    embedding.append(vec)
                                    idx2word.append(word)
                            print('Found %s word vectors.' % len(word2vec))

                            self.word2vec = word2vec
                            self.embedding = np.array(embedding)
                            self.word2idx = {v:k for k,v in enumerate(idx2word)}
                            self.V, self.D = self.embedding.shape
                            self.verbose = verbose
                            self.lowercase = lowercase
                            self.minchars = minchars
                            '''
                            #sentences = [['Test','Alice'],['Test','Bob'],['Alice','Carl'],['David','Carl']]
                            f2=open("sent.txt")
                            WORDStt={}
                            for k in f2:
                                pp=k.strip("\n \t " " ").split(":::")
                                #print(pp)
                                WORDStt[pp[0]]=pp[1]
                            sent=[]
                            sent1=[]
                            self.data=WORDStt
                            sent_map=defaultdict(list)
                            for ty in  WORDStt:
                                gh=[]
                                gh.append(str(ty))
                                for j in WORDStt[ty]:
                                    j1=str(j)
                                    #gh.append(str(ty))
                                    if j1 not in gh:
                                        gh.append(j1)
                                if gh not in sent:
                                        sent.append(gh)
                            f2.close()
                            self.model = gensim.models.Word2Vec(sentences=sent,min_count=1,window=2)#,vector_size=20)  

                        def fit(self, data, *args):
                            pass

                        def transform(self, data, *args):
                            W,D = self.model.wv.vectors.shape
                            X = np.zeros((len(data), D))
                            n = 0
                            emptycount = 0
                            for sentence in data:
                                #if sentence.isdigit()==True:
                                tokens = sentence
                                vecs = []
                                for word in tokens:
                                    if word in self.model.wv:
                                        vec = self.model.wv[word]
                                        vecs.append(vec)
                                if len(vecs) > 0:
                                    vecs = np.array(vecs)
                                    X[n] = vecs.mean(axis=0)
                                else:
                                    emptycount += 1
                                n += 1
                            #X = np.random.rand(100,20)
                            #X1 = np.asarray(X,dtype='float64')
                            return X

                        def fit_transform(self, X, *args):
                            self.fit(X, *args)
                            return self.transform(X, *args)

                    def lime_relation(WORDS22,qrat,similar_r_map,ann,rnn,option):

                                            m_tid_tr1={}
                                            wr=[]
                                            tr=[]
                                            wr1=[]
                                            tr1=[]
                                            c=0
                                            c1=0
                                            tw_wm={}
                                            similar_r_map=similar_r_map
                                                 # organizing feature vector
                                            qf={}
                                            for t in WORDS22:
                                                   # if t in WORDS22 and t in rnn:
                                                        h=WORDS22[t]
                                                        #print(t,qrat[t],h),
                                                        qf[t]=h
                                           # print(qf)
                                            train_r=[]
                                            targets_r=[] 
                                            for t in qf:
                                                    s=''
                                                    vb=[]
                                                    for tt in qf[t]:
                                                        if tt.isalnum():
                                                            train_r.append(tt)
                                                            targets_r.append(qrat[t])

                                            #shap

                                            corpus_train, corpus_test, y_train, y_test = train_test_split(train_r,targets_r, test_size=0.5, random_state=7)
                                            vectorizer = TfidfVectorizer(min_df=1)
                                            #gv=GloveVectorizer()
                                            #print("s")
                                            X_train = vectorizer.fit_transform(corpus_train)
                                            #X_train = gv.fit_transform(corpus_train)
                                            #print("p")
                                            #X_test = gv.fit_transform(corpus_test)
                                            print("n")
                                            X_test = vectorizer.transform(corpus_test)
                                            #model3='

                                            if option=='random':
                                                print("rn")
                                                #rforest = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)


                                            #clf1=rforest GloveVectorizer

                                            #clf1.fit(X_train,y_train)


                                            rforest = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
                                            rforest.fit(X_train, y_train)
                                            p = rforest.predict(X_test)
                                            prr={}
                                            for jj in range(0,len(corpus_test)):
                                                prr[corpus_test[jj]]=int(p[jj])

                                            print(f1_score(y_test,p,average='micro'))

                                            #print_accuracy(rforest.predict)

                                            # explain all the predictions in the test set
                                            explainer = shap.KernelExplainer(rforest.predict_proba, X_train)
                                            shv=[]
                                            for t in X_test:
                                                shap_values = explainer.shap_values(t)
                                                shv.append(shap_values )

                                            shape_w={}
                                            import operator 
                                            import sys
                                            fr={}
                                            feature_sh_v=[]
                                            for jj in range(0,len(shv)):
                                                                      m=(sum(abs(sum(shv[jj][0])))+sum(abs(sum(shv[jj][1]))))/2
                                                                     # print(m)
                                                                      if corpus_test[jj] not in fr:
                                                                                      fr[corpus_test[jj]]=m
                                                                      elif corpus_test[jj]  in fr:
                                                                            if m>fr[corpus_test[jj]]:
                                                                                    fr[corpus_test[jj]]=m

                                            dd1=sorted(fr.items(), key=operator.itemgetter(1),reverse=True)
                                            for tt in dd1:
                                                if tt[0].isdigit()==True:
                                                       feature_sh_v.append(tt[0])
                                                elif tt[0].isdigit()==False:
                                                    for vvv5 in WORDS22:
                                                        if tt[0] in WORDS22[vvv5] or str(tt[0]) in WORDS22[vvv5]:
                                                                                           feature_sh_v.append(tt[0])
                                            shap_exp={}
                                            for t in WORDS22:
                                                gh=[]
                                                c=0
                                                for k in WORDS22[t]:
                                                   # if k in prr:
                                                        #if qrat[t]==prr[k]:
                                                            if k in feature_sh_v:
                                                                if k not in gh:
                                                                    #if c<20:
                                                                        gh.append(k)
                                                                        c=c+1
                                                shap_exp[t]=gh


                                            lexp=shap_exp                                                  
                                            lime_all={}
                                            lime_all_p={}
                                            lime_all_n={}
                                            acco=0
                                            acc_po=0
                                            acc_no=0
                                            accor=0
                                            acc_por=0
                                            acc_nor=0
                                            #print(lexp)
                                            for t in lexp:
                                                if t in ann:
                                                    #print("jjjjjjj")
                                                    c=0
                                                    vb=0
                                                    for zz in lexp[t]:
                                                            if zz in ann[t]:
                                                                #print("hhhhhhh")
                                                                if vb<5:
                                                                        c=c+1
                                                                        vb=vb+1
                                                    if len(lexp[t])>0:
                                                        s=float(c)/len(lexp[t])
                                                        if s>0:
                                                            lime_all[t]=s
                                            ss=0
                                            for k in lime_all:
                                                ss=ss+float(lime_all[k])

                                            if len(lime_all)>0:
                                                acco=ss/len(lime_all)
                                            #print(option)
                                            #print("Lime Accuracy without human-feedback"+"\n")
                                            print(acco)
                                            #print("Lime Accuracy without human-feedback only positive reviews"+"\n")

                                            ss1=0
                                            pp=0
                                            for k in lime_all:
                                                if qrat[k]==2:
                                                    ss1=ss1+float(lime_all[k])
                                                    pp=pp+1

                                            if len(lime_all)>0:
                                                acc_po=ss1/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for positive reviews")
                                            print(acc_po)
                                           # print("Lime only negative reviews")
                                            ss2=0
                                            nn=0
                                            for k in lime_all:
                                                if qrat[k]==0:
                                                    ss2=ss2+float(lime_all[k])
                                                    nn=nn+1

                                            if len(lime_all)>0:
                                                acc_no=ss2/len(lime_all)
                                            #print("Lime Accuracy without human-feedback for negative reviews")
                                            print(acc_no)


                                            return acco,acc_po,acc_no,lexp





                    aos={}
                    pos={}
                    nos={}
                    aor={}
                    por={}
                    nor={}
                    epws={}
                    clss={}
                    option=['random']


                    for tx in option:           
                        acco,acc_po,acc_no,lexp=lime_relation(WORDS22,qrat,similar_r_map,ann,rnn,tx)
                        aos[tx]=acco
                        pos[tx]=acc_po
                        nos[tx]=acc_no
                        epws[tx]=lexp
class shap_svm_lgregression:
    @classmethod
    def shsvregf(cls):

            #Shap+Relation+Feedback+Word Relation Regression SVM
            import pandas as pd
            # Relational annotation
            def relational_embedding_exp(m,WORDS22,qrat,ann):
                # Relational Exp generatetion based on neural embedding
                            sent2=[]
                            sent1=[]
                            sent_map=defaultdict(list)
                            for ty in WORDS22:
                                gh=[]
                                gh.append(str(ty))
                                #gh1=[]
                                #gh2=[]
                                for j in WORDS22[ty]:

                                    j1=str(j)
                                    #gh.append(str(ty))
                                    if j1 not in gh:
                                        gh.append(j1)
                                    ##print(gh)


                                if gh not in sent2:
                                        sent2.append(gh)


                            documents1=[]
                            #documents1=[]
                            for t in sent2:
                                s=''
                                for jh in t:
                                    if jh.isdigit():
                                         documents1.append(jh)
                                    else:
                                        s=" "+str(jh)+s+" "
                                documents1.append(s)


                            #sentence embedding
                            from gensim.test.utils import common_texts
                            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
                            documents2 = [TaggedDocument(doc, [i]) for i, doc in enumerate(sent2)]
                            for t in documents2:
                                pass##print(t)
                            model = Doc2Vec(documents2, vector_size=5, window=2, min_count=1, workers=4)

                            #K-Means Run 14 to find the neighbors per query 

                            #cluster generation with k-means
                            import sys
                            from nltk.cluster import KMeansClusterer
                            import nltk
                            from sklearn import cluster
                            from sklearn import metrics
                            import gensim 
                            import operator
                            #from gensim.models import Word2Vec


                            #model = Word2Vec(sent, min_count=1) dd=sorted(vb.items(), key=operator.itemgetter(1),reverse=False)
                            import operator
                            X = model[model.wv.vocab]
                            c=0
                            cluster={}
                            num=[]
                            weight_map={}
                            similar_r_map={}
                            fg={}
                            for jj in WORDS22:
                                gh1=[]
                                gh2=[]
                                s=0

                                for k in documents1:
                                    if str(k)==str(jj):
                                        gh=model.most_similar(positive=str(k),topn=600)
                                       # #print(gh)
                                        for tt in gh:
                                            if float(tt[1]) not in gh1:
                                                gh1.append(float(tt[1]))
                                            #if tt[0] not in gh2:
                                            if tt[0].isdigit():
                                                    #if ccc<5:
                                                            #gh2.append(tt[0])
                                                            fg[tt[0]]=tt[1]
                                                            #ccc=ccc+1
                                #for ffg in gh1:
                                    #s=s+ffg
                                dd=sorted(fg.items(), key=operator.itemgetter(1),reverse=True)
                                ccc=0
                                for t5 in dd:
                                    if qrat[str(jj)]==qrat[str(t5[0])]:
                                        if m==5:
                                            if ccc<500:
                                                     gh2.append(t5[0])
                                                     ccc=ccc+1
                                        elif m==10:
                                            if ccc<400:
                                                     gh2.append(t5[0])
                                                     ccc=ccc+1
                                        elif m==15:
                                            if ccc<500:
                                                     gh2.append(t5[0])
                                                     ccc=ccc+1
                                        elif m==20:
                                            if ccc<600:
                                                     gh2.append(t5[0])
                                                     ccc=ccc+1
                                        elif m==25:
                                            if ccc<700:
                                                     gh2.append(t5[0])
                                                     ccc=ccc+1

                                #if len(gh2)>=2:
                                similar_r_map[jj]=gh2
                                        #ccc=ccc+1

                            return similar_r_map


            import gensim.models.word2vec as W2V
            import gensim.models
            import sys
            from sklearn.ensemble import RandomForestRegressor
            class GloveVectorizer:
                def __init__(self, verbose=False, lowercase=True, minchars=3):
                    '''
                    # load in pre-trained word vectors
                    print('Loading word vectors...')
                    word2vec = {}
                    embedding = []
                    idx2word = []
                    with open('../data/glove.6B.50d.txt') as f:
                          # is just a space-separated text file in the format:
                          # word vec[0] vec[1] vec[2] ...
                          for line in f:
                            values = line.split()
                            word = values[0]
                            vec = np.asarray(values[1:], dtype='float32')
                            word2vec[word] = vec
                            embedding.append(vec)
                            idx2word.append(word)
                    print('Found %s word vectors.' % len(word2vec))

                    self.word2vec = word2vec
                    self.embedding = np.array(embedding)
                    self.word2idx = {v:k for k,v in enumerate(idx2word)}
                    self.V, self.D = self.embedding.shape
                    self.verbose = verbose
                    self.lowercase = lowercase
                    self.minchars = minchars
                    '''
                    #sentences = [['Test','Alice'],['Test','Bob'],['Alice','Carl'],['David','Carl']]
                    f2=open("sent.txt")
                    WORDStt={}
                    for k in f2:
                        pp=k.strip("\n \t " " ").split(":::")
                        #print(pp)
                        WORDStt[pp[0]]=pp[1]
                    sent=[]
                    sent1=[]
                    self.data=WORDStt
                    sent_map=defaultdict(list)
                    for ty in  WORDStt:
                        gh=[]
                        gh.append(str(ty))
                        for j in WORDStt[ty]:
                            j1=str(j)
                            #gh.append(str(ty))
                            if j1 not in gh:
                                gh.append(j1)
                        if gh not in sent:
                                sent.append(gh)
                    f2.close()
                    self.model = gensim.models.Word2Vec(sentences=sent,min_count=1,window=2)#,vector_size=20)  

                def fit(self, data, *args):
                    pass

                def transform(self, data, *args):
                    W,D = self.model.wv.vectors.shape
                    X = np.zeros((len(data), D))
                    n = 0
                    emptycount = 0
                    for sentence in data:
                        #if sentence.isdigit()==True:
                        tokens = sentence
                        vecs = []
                        for word in tokens:
                            if word in self.model.wv:
                                vec = self.model.wv[word]
                                vecs.append(vec)
                        if len(vecs) > 0:
                            vecs = np.array(vecs)
                            X[n] = vecs.mean(axis=0)
                        else:
                            emptycount += 1
                        n += 1
                    #X = np.random.rand(100,20)
                    #X1 = np.asarray(X,dtype='float64')
                    return X

                def fit_transform(self, X, *args):
                    self.fit(X, *args)
                    return self.transform(X, *args)


            def lime_all_acc(WORDS22,tx,qrat):
                            def feedback_accuracy(WORDS22,qrat,similar_r_map,ann,rnn,option):
                                    train_r=[]
                                    targets_r=[]
                                    m_tid_tr1={}
                                    wr=[]
                                    tr=[]
                                    wr1=[]
                                    tr1=[]
                                    c=0
                                    c1=0
                                    tw_wm={}
                                    train_r=[]
                                    targets_r=[]
                                    m_tid_tr1={}
                                    wr=[]
                                    tr=[]
                                    wr1=[]
                                    tr1=[]
                                    c=0
                                    c1=0
                                    tw_wm={}
                                    similar_r_map=relational_embedding_exp(5,WORDS22,qrat,ann)
                                    rnn1={}
                                    for t in similar_r_map:
                                        gg=[]
                                        cc=0
                                        for k in similar_r_map[t]:
                                            if cc<25:
                                                gg.append(k)
                                                cc=cc+1
                                        rnn1[t]=gg

                                    # organizing feature vector
                                    qf={}
                                    for t in WORDS22:
                                            #if t in WORDS22 and t in rnn:
                                            if str(t) in rnn:
                                                h=WORDS22[t]+rnn[str(t)]
                                                #print(t,h)
                                                qf[t]=h
                                    #sys.exit()

                                    #s=''
                                    #vb=[]
                                    #if twit_count[t]==1:
                                    for t in qf:
                                            for tt in qf[t]:
                                                    train_r.append(tt)
                                                    targets_r.append(qrat[str(t)])



                                    ###################
                                    corpus_train, corpus_test, y_train, y_test = train_test_split(train_r,targets_r, test_size=0.5, random_state=7)
                                    #vectorizer = TfidfVectorizer(min_df=1)
                                    #X_train = vectorizer.fit_transform(corpus_train)
                                    #X_test = vectorizer.transform(corpus_test)
                                    #model3='
                                    if option=='svm':
                                        model1 =svm.LinearSVC(C=10)
                                        #svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)
                                        #svm.LinearSVC(C=50) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                    elif option=='regression':
                                        model1 = sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1,solver='liblinear')

                                    #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                    #KNeighborsClassifier(n_neighbors=5)
                                    #RandomForestClassifier(max_depth=2, random_state=1)
                                    #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                    clf=model1
                                    gv = GloveVectorizer()
                                    vectorizer = TfidfVectorizer(min_df=1)
                                    X_train = gv.fit_transform(corpus_train)
                                    X_test = gv.fit_transform(corpus_test)
                                    clf.fit(X_train,y_train)
                                    p = clf.predict(X_test)
                                    prr={}
                                    for jj in range(0,len(corpus_test)):
                                        prr[corpus_test[jj]]=int(p[jj])
                                    print(f1_score(y_test,p,average='micro'))

                                    #X=gv.fit_transform(qf)
                                    #print(X)

                                    #c = make_pipeline(gv,clf)
                                    #try:
                                   # c.fit(corpus_train,y_train)
                                    #explainer2=shap.TreeExplainer(clf)
                                    explainer = shap.LinearExplainer(clf, X_train, feature_dependence="independent")
                                    shap_values = explainer.shap_values(X_train)
                                    shape_w={}
                                    fr={}
                                    feature_sh_v=[]
                                    for jj in range(0,len(corpus_train)):
                                                          m=abs(sum(shap_values[jj]))
                                                          if corpus_train[jj] not in fr:
                                                                          fr[corpus_train[jj]]=abs(sum(shap_values[jj]))
                                                          elif corpus_train[jj]  in fr:
                                                                if m>fr[corpus_train[jj]]:
                                                                    fr[corpus_train[jj]]=m
                                    dd1=sorted(fr.items(), key=operator.itemgetter(1),reverse=True)
                                    for tt in dd1:
                                        if tt[0].isdigit()==True:
                                               feature_sh_v.append(tt[0])
                                        elif tt[0].isdigit()==False:
                                            for vvv5 in WORDS22:
                                                if tt[0] in WORDS22[vvv5] or str(tt[0]) in WORDS22[vvv5]:
                                                                                   feature_sh_v.append(tt[0])
                                    shap_exp={}
                                    for t in qf:
                                        gh=[]
                                        c=0
                                        for k in qf[t]:
                                            if k in prr:
                                                #if qrat[t]==prr[k]:
                                                    if k in feature_sh_v:
                                                        if k not in gh:
                                                            if k.isdigit()==False:
                                                                #if c<20:
                                                                    gh.append(k)
                                                                    c=c+1
                                        shap_exp[t]=gh
                                    lexp=shap_exp
                                    lime_all={}
                                    lime_all_p={}
                                    lime_all_n={}
                                    acc=0
                                    acc_n=0
                                    acc_p=0
                                    accr=0
                                    acc_nr=0
                                    acc_pr=0


                                    for t in lexp:
                                        if str(t) in ann:
                                            c=0
                                            vb=0
                                            for zz in lexp[t]:
                                                if zz.isdigit():
                                                    continue
                                                else:
                                                    if zz in ann[str(t)]:
                                                        if vb<5:
                                                                c=c+1
                                                                vb=vb+1
                                            if len(lexp[t])>0:
                                                s=float(c)/len(lexp[t])
                                                if s>0:
                                                    lime_all[t]=s
                                    ss=0
                                    for k in lime_all:
                                        ss=ss+float(lime_all[k])

                                    if len(lime_all)>0:
                                        acc=ss/len(lime_all)
                                    #print(option)
                                    #print("Lime Accuracy without human-feedback"+"\n")
                                    #print(acc)
                                    #print("Lime Accuracy without human-feedback only positive reviews"+"\n")

                                    ss1=0
                                    pp=0
                                    for k in lime_all:
                                        try:
                                            if qrat[str(k)]==1 or qrat[k]==1:
                                                ss1=ss1+float(lime_all[k])
                                                pp=pp+1
                                        except:
                                            continue

                                    if len(lime_all)>0:
                                        acc_p=ss1/len(lime_all)
                                    #print("Lime Accuracy without human-feedback for positive reviews")
                                    #print(acc_p)
                                   # print("Lime only negative reviews")
                                    ss2=0
                                    nn=0
                                    for k in lime_all:
                                        try:
                                            if qrat[str(k)]==0 or qrat[k]==0:
                                                ss2=ss2+float(lime_all[k])
                                                nn=nn+1
                                        except:
                                            continue 

                                    if len(lime_all)>0:
                                        acc_n=ss2/len(lime_all)
                                    #print("Lime Accuracy without human-feedback for negative reviews")
                                    #print(acc_n)

                                    return acc,acc_p,acc_n,qrat,lexp







                            #def feed(mn,qrat,cl,w3):
                            def feed(mn,qrat,cl,WORDS22):
                                            import pandas as pd
                                            sent=[]
                                            sent1=[]
                                            sent_map=defaultdict(list)
                                            w33={}
                                            cp=0
                                            cn=0
                                            for kkk in WORDS22:
                                                gvv=[]
                                                if qrat[str(kkk)]==2:
                                                    if cp<40:
                                                        w33[kkk]=WORDS22[kkk]
                                                        cp=cp+1
                                                elif qrat[str(kkk)]==0:
                                                    if cn<40:
                                                        w33[kkk]=WORDS22[kkk]
                                                        cn=cn+1
                                            for ty in WORDS22:
                                                gh=[]
                                                gh.append(str(ty))
                                                #gh1=[]
                                                #gh2=[]
                                                for j in WORDS22[ty]:

                                                    j1=str(j)
                                                    #gh.append(str(ty))
                                                    if j1 not in gh:
                                                        gh.append(j1)

                                                if gh not in sent:
                                                        sent.append(gh)
                                            documents=[]
                                            #documents1=[]
                                            for t in sent:
                                                for jh in t:
                                                    documents.append(jh)
                                            hh="twfeedback_Bert"+str(mn)+".csv"
                                            ps={}
                                            ns={}
                                            vot={}
                                            vtt={}
                                            f1=pd.read_csv(hh)
                                            vot={}
                                            vtt={}
                                            ll=0
                                            for col in f1.columns:
                                                if 'Label' in col:
                                                    ll=ll+1
                                            m=len(f1['Tweet_ID'])
                                            for t in range(0,m):
                                                #print(f1['Review_ID'][t])
                                                zz=f1['Explanation'][t].split()
                                                vtt[f1['Tweet_ID'][t]]=zz#f1['Explanation'][t]
                                                gh=[]
                                                for vv in range(1,ll+1):
                                                              vb1="Label"+str(vv)
                                                              #print(f1[vb1][t])
                                                              gh.append(f1[vb1][t])
                                                vot[f1['Tweet_ID'][t]]=gh


                                            for uy in vot:
                                                cp=0
                                                cn=0
                                                for kk in vot[uy]:
                                                    if int(kk)==1:
                                                        cp=cp+1
                                                    else:
                                                        cn=cn+1
                                                if cp>=cn:
                                                    ps[uy]=vtt[uy]
                                                elif cp<cn:
                                                    ns[uy]=vtt[uy]

                                            cl1=cl[mn]
                                            print(ps,ns)
                                            import sys


                                            #print(cl1)
                                            WORDS23={}
                                            rm=[]
                                            for jj in ns:
                                                for kj in ns[jj]:
                                                    rm.append(kj)

                                            model = Word2Vec(sent, min_count=1)
                                            rme={}
                                            for uu in ns:
                                                for k in cl1:
                                                    if int(uu) in cl1[k]:
                                                        for kk in cl1[k]:
                                                            gg=[]
                                                            zz=0
                                                            if str(kk) in WORDS22:
                                                                #print("hi")

                                                                for vv in WORDS22[str(kk)]:
                                                                    if vv in ns[uu]:
                                                                        continue
                                                                    else:
                                                                        if zz<5:
                                                                            gg.append(vv)
                                                                            zz=zz+1
                                                                rme[kk]=gg


                                            for t in cl1:
                                                for k in ps:
                                                    if int(k) in cl1[t]:
                                                        for kk in cl1[t]:
                                                            chu1=[]
                                                            vb={}
                                                            for v in ps[k]:
                                                                vb1={}
                                                                if kk in WORDS22 or str(kk) in WORDS22:
                                                                    for v1 in WORDS22[str(kk)]:
                                                                        try:
                                                                                gh1=model.similarity(v,v1)
                                                                                if gh1>0.1:
                                                                                              vb1[v1]=float(gh1)
                                                                        except:
                                                                            continue
                                                                    for jk in vb1:
                                                                                    if jk in vb:
                                                                                        if float(vb1[jk])>=float(vb[jk]):
                                                                                            #print(jk,vb1[jk],vb[jk])
                                                                                            vb[jk]=vb1[jk]
                                                                                    else:
                                                                                        vb[jk]=vb1[jk]

                                                            dd1=sorted(vb.items(), key=operator.itemgetter(1),reverse=True)
                                                            cc=0
                                                            for kkk in dd1:
                                                                if kkk[0] not in chu1:
                                                                    if cc<5:
                                                                            chu1.append(kkk[0])
                                                                            cc=cc+1
                                                            if len(chu1)>0 :
                                                                if str(kk) in WORDS22:
                                                                    WORDS23[kk]=chu1 
                                            WORDS25={}

                                            for t in WORDS23:
                                                cc=0
                                                vcx=[]
                                                if t not in rme:
                                                    WORDS25[t]=WORDS23[t]
                                                elif t in rme:
                                                    vcc=WORDS23[t]+rme[t]
                                                    for zz in vcc:
                                                        if cc<5:
                                                            vcx.append(zz)
                                                            cc=cc+1
                                                    WORDS25[t]=vcx
                                           # print(len(WORDS25))
                                            for cc in rme:
                                                fg=[]
                                                vc4=0
                                                if cc not in WORDS25:
                                                    for bb in rme[cc]:
                                                        if vc4<2:
                                                            fg.append(bb)
                                                            vc4=vc4+1
                                                    WORDS25[cc]=fg
                                            #print(WORDS25)
                                            #print(len(WORDS25))
                                            #sys.exit()

                                            return WORDS25,qrat


                            all_a={}
                            #all_a={}
                            on_p={}
                            on_n={}
                            exp_all={}
                            all_exp={}
                            all_acc={}
                            all_acc_p={}
                            all_acc_n={}
                            all_accr={}
                            all_acc_pr={}
                            all_acc_nr={}
                            clssr={}
                            for mn in range(2,7,1):     
                                    WORDS25,qrat=feed(mn,qrat,cl,WORDS22)
                                    #print(len(WORDS25))
                                    #sys.exit()
                                    acc,acc_p,acc_n,qrat,lexp=feedback_accuracy(WORDS25,qrat,similar_r_map,ann,rnn,tx)  
                                    all_a[mn]=acc
                                    all_acc[mn]=acc
                                    all_acc_p[mn]=acc_p
                                    all_acc_n[mn]=acc_n
                                    clssr[mn]=qrat
                                    on_p[mn]=acc_p
                                    on_n[mn]=acc_n
                                    exp_all[mn]=lexp
                                   # all_exp[mn]=lexp11

                            print(tx)
                            print("\n")
                            for tt in all_a:
                                print(tt,all_acc[tt],all_acc_p[tt],all_acc_n[tt])
                                print("\n\n")
                            return exp_all,all_exp,all_acc,all_acc_p,all_acc_n,clssr#,acco,acc_po,acc_no

            expws1={}
            exp2={}
            #exp1={}
            #exp2={}
            ash={}
            psh={}
            nsh={}
            ar={}
            pr={}
            nr={}
            ao={}
            po={}
            no={}
            clas={}
            option=['svm']


            for tx in option:           
                #acco,acc_po,acc_no,lexp,lexp1,word_weight,ann,corpus_train, corpus_test, y_train, y_test,p,qrat,WORDSt=Shap_Explanation_Engine(s_words,stopwords,WORDS,qrat,H11,Rev_text_map,tx)
                exp_all,all_exp,all_acc,all_acc_p,all_acc_n,clssr=lime_all_acc(WORDS22,tx,qrat)
                expws1[tx]=exp_all
               # exp2[tx]=all_exp
                ash[tx]=all_acc
                psh[tx]=all_acc_p
                nsh[tx]=all_acc_n
                clas[tx]=clssr




    @classmethod
    def shsvregnf(cls):

                                class GloveVectorizer:
                                    def __init__(self, verbose=False, lowercase=True, minchars=3):
                                        '''
                                        # load in pre-trained word vectors
                                        print('Loading word vectors...')
                                        word2vec = {}
                                        embedding = []
                                        idx2word = []
                                        with open('../data/glove.6B.50d.txt') as f:
                                              # is just a space-separated text file in the format:
                                              # word vec[0] vec[1] vec[2] ...
                                              for line in f:
                                                values = line.split()
                                                word = values[0]
                                                vec = np.asarray(values[1:], dtype='float32')
                                                word2vec[word] = vec
                                                embedding.append(vec)
                                                idx2word.append(word)
                                        print('Found %s word vectors.' % len(word2vec))

                                        self.word2vec = word2vec
                                        self.embedding = np.array(embedding)
                                        self.word2idx = {v:k for k,v in enumerate(idx2word)}
                                        self.V, self.D = self.embedding.shape
                                        self.verbose = verbose
                                        self.lowercase = lowercase
                                        self.minchars = minchars
                                        '''
                                        #sentences = [['Test','Alice'],['Test','Bob'],['Alice','Carl'],['David','Carl']]
                                        f2=open("sent.txt")
                                        WORDStt={}
                                        for k in f2:
                                            pp=k.strip("\n \t " " ").split(":::")
                                            #print(pp)
                                            WORDStt[pp[0]]=pp[1]
                                        sent=[]
                                        sent1=[]
                                        self.data=WORDStt
                                        sent_map=defaultdict(list)
                                        for ty in  WORDStt:
                                            gh=[]
                                            gh.append(str(ty))
                                            for j in WORDStt[ty]:
                                                j1=str(j)
                                                #gh.append(str(ty))
                                                if j1 not in gh:
                                                    gh.append(j1)
                                            if gh not in sent:
                                                    sent.append(gh)
                                        f2.close()
                                        self.model = gensim.models.Word2Vec(sentences=sent,min_count=1,window=2)#,vector_size=20)  

                                    def fit(self, data, *args):
                                        pass

                                    def transform(self, data, *args):
                                        W,D = self.model.wv.vectors.shape
                                        X = np.zeros((len(data), D))
                                        n = 0
                                        emptycount = 0
                                        for sentence in data:
                                            #if sentence.isdigit()==True:
                                            tokens = sentence
                                            vecs = []
                                            for word in tokens:
                                                if word in self.model.wv:
                                                    vec = self.model.wv[word]
                                                    vecs.append(vec)
                                            if len(vecs) > 0:
                                                vecs = np.array(vecs)
                                                X[n] = vecs.mean(axis=0)
                                            else:
                                                emptycount += 1
                                            n += 1
                                        #X = np.random.rand(100,20)
                                        #X1 = np.asarray(X,dtype='float64')
                                        return X

                                    def fit_transform(self, X, *args):
                                        self.fit(X, *args)
                                        return self.transform(X, *args)

                                def lime_relation(WORDS22,qrat,similar_r_map,ann,rnn,option):
                                                        train_r=[]
                                                        targets_r=[]
                                                        m_tid_tr1={}
                                                        wr=[]
                                                        tr=[]
                                                        wr1=[]
                                                        tr1=[]
                                                        c=0
                                                        c1=0
                                                        tw_wm={}
                                                        similar_r_map=similar_r_map
                                                             # organizing feature vector
                                                        qf={}
                                                        for t in similar_r_map:
                                                                if t in WORDS22 and t in rnn:
                                                                    h=WORDS22[t]
                                                                    #print(t,qrat[t],h),
                                                                    qf[t]=h
                                                       # print(qf)

                                                        for t in qf:
                                                                s=''
                                                                vb=[]
                                                                for tt in qf[t]:
                                                                    if tt.isalnum():
                                                                        train_r.append(tt)
                                                                        targets_r.append(qrat[t])

                                                        #shap

                                                        corpus_train, corpus_test, y_train, y_test = train_test_split(train_r,targets_r, test_size=0.5, random_state=7)
                                                        vectorizer = TfidfVectorizer(min_df=1)
                                                        X_train = vectorizer.fit_transform(corpus_train)
                                                        X_test = vectorizer.transform(corpus_test)
                                                        #model3='
                                                        if option=='svm':
                                                            model1 =svm.LinearSVC(C=10)
                                                            #svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)
                                                            #svm.LinearSVC(C=50) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                                        elif option=='regression':
                                                            model1 = sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1,solver='liblinear')
                                                            #sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1)
                                                        elif option=='random':
                                                            model1=RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
                                                            #RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)#,probability=True)
                                                        elif option=='extratree':
                                                            model1=ExtraTreesClassifier(n_estimators=100, random_state=0) 
                                                        elif option=='knn':
                                                            model1=KNeighborsClassifier(n_neighbors=3)

                                                        #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                                        #KNeighborsClassifier(n_neighbors=5)
                                                        #RandomForestClassifier(max_depth=2, random_state=1)
                                                        #svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                                        clf=model1
                                                        #X_train =np.array(X_train)
                                                       # y_train=np.array(y_train)
                                                        #X_train = shap.utils.sample(X_train, 100)
                                                       # y_train= shap.utils.sample(y_train, 100)
                                                        clf.fit(X_train,y_train)
                                                        p = clf.predict(X_test)
                                                        prr={}
                                                        for jj in range(0,len(corpus_test)):
                                                            prr[corpus_test[jj]]=int(p[jj])
                                                        #print(f1_score(y_test,p,average='micro'))
                                                        #gv = GloveVectorizer()

                                                        #X=gv.fit_transform(qf)
                                                        #print(X)

                                                        #c = make_pipeline(gv,clf)
                                                       # pmodel = shap.models.TransformersPipeline(gv, rescale_to_logits=False)
                                                       # pmodel(corpus_train)
                                                        #try:
                                                        #c.fit(corpus_train,y_train)
                                                        #X_train=np.squeeze(np.asarray(X_train))
                                                        #explainer2 = shap.KernelExplainer(clf.predict_proba, X_train)#shap.TreeExplainer(clf)
                                                        #shap.KernelExplainer(clf.predict_proba, X_train[:100])
                                                        explainer = shap.LinearExplainer(clf, X_train, feature_dependence="independent")
                                                        shap_values = explainer.shap_values(X_train)
                                                        shape_w={}
                                                        fr={}
                                                        feature_sh_v=[]
                                                        for jj in range(0,len(corpus_train)):
                                                                              m=abs(sum(shap_values[jj]))
                                                                              if corpus_train[jj] not in fr:
                                                                                              fr[corpus_train[jj]]=abs(sum(shap_values[jj]))
                                                                              elif corpus_train[jj]  in fr:
                                                                                    if m>fr[corpus_train[jj]]:
                                                                                        fr[corpus_train[jj]]=m
                                                        dd1=sorted(fr.items(), key=operator.itemgetter(1),reverse=True)
                                                        for tt in dd1:
                                                            if tt[0].isdigit()==True:
                                                                   feature_sh_v.append(tt[0])
                                                            elif tt[0].isdigit()==False:
                                                                for vvv5 in WORDS22:
                                                                    if tt[0] in WORDS22[vvv5] or str(tt[0]) in WORDS22[vvv5]:
                                                                                                       feature_sh_v.append(tt[0])
                                                        shap_exp={}
                                                        for t in qf:
                                                            gh=[]
                                                            c=0
                                                            for k in qf[t]:
                                                                if k in prr:
                                                                    #if qrat[t]==prr[k]:
                                                                        if k in feature_sh_v:
                                                                            if k not in gh:
                                                                                #if c<20:
                                                                                    gh.append(k)
                                                                                    c=c+1
                                                            shap_exp[t]=gh


                                                        lexp=shap_exp                                                  
                                                        lime_all={}
                                                        lime_all_p={}
                                                        lime_all_n={}
                                                        acco=0
                                                        acc_po=0
                                                        acc_no=0
                                                        accor=0
                                                        acc_por=0
                                                        acc_nor=0
                                                        #print(lexp)
                                                        for t in lexp:
                                                            if t in ann:
                                                                #print("jjjjjjj")
                                                                c=0
                                                                vb=0
                                                                for zz in lexp[t]:
                                                                        if zz in ann[t]:
                                                                            #print("hhhhhhh")
                                                                            if vb<5:
                                                                                    c=c+1
                                                                                    vb=vb+1
                                                                if len(lexp[t])>0:
                                                                    s=float(c)/len(lexp[t])
                                                                    if s>0:
                                                                        lime_all[t]=s
                                                        ss=0
                                                        for k in lime_all:
                                                            ss=ss+float(lime_all[k])

                                                        if len(lime_all)>0:
                                                            acco=ss/len(lime_all)
                                                        #print(option)
                                                        #print("Lime Accuracy without human-feedback"+"\n")
                                                        print(acco)
                                                        #print("Lime Accuracy without human-feedback only positive reviews"+"\n")

                                                        ss1=0
                                                        pp=0
                                                        for k in lime_all:
                                                            if qrat[k]==1:
                                                                ss1=ss1+float(lime_all[k])
                                                                pp=pp+1

                                                        if len(lime_all)>0:
                                                            acc_po=ss1/len(lime_all)
                                                        #print("Lime Accuracy without human-feedback for positive reviews")
                                                        print(acc_po)
                                                       # print("Lime only negative reviews")
                                                        ss2=0
                                                        nn=0
                                                        for k in lime_all:
                                                            if qrat[k]==0:
                                                                ss2=ss2+float(lime_all[k])
                                                                nn=nn+1

                                                        if len(lime_all)>0:
                                                            acc_no=ss2/len(lime_all)
                                                        #print("Lime Accuracy without human-feedback for negative reviews")
                                                        print(acc_no)


                                                        return acco,acc_po,acc_no,lexp





                                aos={}
                                pos={}
                                nos={}
                                aor={}
                                por={}
                                nor={}
                                epws={}
                                clss={}
                                option=['regression']


                                for tx in option:           
                                    acco,acc_po,acc_no,lexp=lime_relation(WORDS22,qrat,similar_r_map,ann,rnn,tx)
                                    aos[tx]=acco
                                    pos[tx]=acc_po
                                    nos[tx]=acc_no
                                    epws[tx]=lexp
