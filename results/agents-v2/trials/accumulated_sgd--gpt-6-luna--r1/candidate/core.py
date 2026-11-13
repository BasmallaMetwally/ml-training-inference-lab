import math

def gradient(weights,x,y):
    if not x:
        return [0.0]*len(weights)
    return [sum((sum(a*b for a,b in zip(row,weights))-target)*row[j] for row,target in zip(x,y))/len(x) for j in range(len(weights))]

def step(weights,velocity,g,lr,momentum,clip):
    if clip<0:
        raise ValueError('clip')
    norm=math.sqrt(sum(v*v for v in g))
    scale=min(1.0,clip/norm) if norm else 1.0
    velocity=[momentum*v+d for v,d in zip(velocity,g)]
    return [w-lr*v for w,v in zip(weights,velocity)],velocity
