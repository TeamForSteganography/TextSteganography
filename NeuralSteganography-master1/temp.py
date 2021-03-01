def temp(a):
    print('gellp')

from base64 import *
import re
def dfs(arr, pos, res):
    res.append(''.join(arr))
    i=pos
    for i in range(i,len(arr)):
        arr[i]=arr[i].lower()
        dfs(arr, i+1, res)
        arr[i]=arr[i].upper()
    return res
    
arr=list('ABC')
res = []
res = dfs(arr, 0, res)
print(res)

temp(1)