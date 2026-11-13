def count(actual,predicted,labels):
    if len(actual)!=len(predicted) or len(labels)!=len(set(labels)):
        raise ValueError('inputs')
    names=sorted(set(labels)|set(actual)|set(predicted))
    matrix=[[0]*len(names) for _ in names]
    for a,b in zip(actual,predicted):
        matrix[names.index(a)][names.index(b)]+=1
    return names,matrix

def merge(left,right):
    names=sorted(set(left[0])|set(right[0]))
    matrix=[[0]*len(names) for _ in names]
    for labels,values in (left,right):
        for i,a in enumerate(labels):
            for j,b in enumerate(labels):
                matrix[names.index(b)][names.index(a)]+=values[i][j]
    return names,matrix
