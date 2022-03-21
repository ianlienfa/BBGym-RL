import numpy as np
import random
import string
import sys

job_num_max = 50
size = np.random.randint(0, job_num_max, size=(1,))[0]
p = np.random.randint(0, 100, size=(size,))
r = np.random.randint(0, 100, size=(size,))
w = np.random.randint(0, 100, size=(size,))
filename = ''.join(random.choices(string.ascii_lowercase, k=10)) + '.in'

sys.stdout = open(filename, 'w')
print(size)
v = [p, r, w]
for j in v:
    for i in j:
        print(i, end=' ')
    print()
print(-1)  
sys.stdout.close()