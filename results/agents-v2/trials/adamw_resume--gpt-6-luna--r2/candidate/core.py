import math

def update(w,m,v,g,t,lr,b1,b2,eps,decay):
    nm=[b1*a+(1-b1)*b for a,b in zip(m,g)]
    nv=[b2*a+(1-b2)*b*b for a,b in zip(v,g)]
    nw=[a*(1-lr*decay)-lr*(b/(1-b1**t))/(math.sqrt(c/(1-b2**t))+eps) for a,b,c in zip(w,nm,nv)]
    return nw,nm,nv
