import numpy as np
import random

a=np.random.random_integers(100,size=(6,2))
print("打乱前：")
print(a)

print("打乱后：")
np.random.shuffle(a)
print(a)

ind=np.arange(6,dtype=np.int32)
np.random.shuffle(ind)
ind=ind[:3]

print(a[ind])


print(a[:,1].shape)#(6,)
