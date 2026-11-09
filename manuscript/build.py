import re, subprocess, sys
sys.path.insert(0,'.')
from refs import R
parts=[open(f'part{i}.md').read() for i in (1,2,3,4)]
md='\n\n'.join(parts)
order=[]
def cite(m):
    keys=[k.strip().lstrip('@') for k in m.group(1).split(';')]
    nums=[]
    for k in keys:
        if k not in R: raise SystemExit('missing ref '+k)
        if k not in order: order.append(k)
        nums.append(order.index(k)+1)
    nums=sorted(set(nums))
    # compress ranges
    out=[];i=0
    while i<len(nums):
        j=i
        while j+1<len(nums) and nums[j+1]==nums[j]+1: j+=1
        out.append(f'{nums[i]}–{nums[j]}' if j-i>=2 else ', '.join(str(n) for n in nums[i:j+1]))
        i=j+1
    return '['+', '.join(out)+']'
md=re.sub(r'\[(@[^\]]+)\]',cite,md)
md+='\n\n# References\n\n'+'\n\n'.join(f'[{i+1}] {R[k]}' for i,k in enumerate(order))
open('manuscript.md','w').write(md)
unused=[k for k in R if k not in order]
print('refs used',len(order),'unused',unused)
