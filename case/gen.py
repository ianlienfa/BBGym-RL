import numpy as np
import random
import string
import sys

job_num_min = 20
job_num_max = 30
size = np.random.randint(job_num_min, job_num_max, size=(1,))[0]
p = np.random.randint(1, 50, size=(size,))
r = np.random.randint(1, 50, size=(size,))
w = np.random.randint(1, 50, size=(size,))
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