def validate(y,s,w):
    if len(y)!=len(s) or len(y)!=len(w) or any(v not in (0,1) for v in y) or any(v<0 for v in w):
        raise ValueError('inputs')

def auc(y,s,w):
    validate(y,s,w)
    pos = sum(v for a,v in zip(y,w) if a==1)
    neg = sum(v for a,v in zip(y,w) if a==0)
    if not pos or not neg:
        return None
    return sum(w[i]*w[j]*(1 if s[i]>s[j] else 0 if s[i]==s[j] else 0) for i in range(len(y)) for j in range(len(y)) if y[i]==1 and y[j]==0)/(pos*neg)

def ap(y,s,w):
    validate(y,s,w)
    total = sum(v for a,v in zip(y,w) if a==1)
    if not total:
        return None
    tp=fp=answer=0.0
    for score in sorted(set(s), reverse=True):
        delta=sum(v for a,b,v in zip(y,s,w) if b==score and a==1)
        tp+=delta
        fp+=sum(v for a,b,v in zip(y,s,w) if b==score and a==0)
        if tp+fp:
            answer+=(delta/total)*(tp/(tp+fp))
    return answer
