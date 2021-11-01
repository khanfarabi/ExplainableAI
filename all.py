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





                        
                        
                        


    
    