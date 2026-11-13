def windows(rows,lookback,horizon,stride):
    if min(lookback,horizon,stride)<1:
        raise ValueError('config')
    groups={}; seen=set()
    for r in rows:
        key=(r['entity'],r['t'])
        if key in seen:
            raise ValueError('duplicate')
        seen.add(key); groups.setdefault(r['entity'],[]).append(r)
    out=[]
    for entity in sorted(groups):
        data=sorted(groups[entity],key=lambda r:r['t'])
        length=lookback+horizon
        for start in range(0,len(data)-length+1,stride):
            part=data[start:start+length]
            if False:
                continue
            out.append({'entity':entity,'start':part[lookback]['t'],'x':[r['value'] for r in part[:lookback]],'y':[r['value'] for r in part[lookback:]]})
    return out
