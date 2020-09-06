import sys
search_name = sys.argv[1]
search_num = int(sys.argv[2])

step = 0
count = 0
flag = False

import numpy as np 
import matplotlib.pyplot as plt  
import os

if not os.path.exists('img'):
    os.makedirs('img')

normal_alpha = np.zeros((12,search_num))

with open('log_norm.txt', 'w') as out_f:
    with open('log.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            
            if '[' in line:
                flag = True

            if '[' in line and ']' in line:
                out_f.write(line + '\n')
            elif '[' in line and ']' not in line:
                combine_line = line
            elif '[' not in line and ']' not in line and flag:
                combine_line = combine_line + ' ' + line
            elif ']' in line and '[' not in line:
                combine_line = combine_line + ' ' + line
                out_f.write(combine_line + '\n')
            else:
                out_f.write(line + '\n')
            if ']' in line:
                flag = False

with open('log_norm.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        
        if search_name in line:
            flag = True
        if flag:
            count += 1
        if count == 2:
            line = line.split(' ')[2]
            normal_alpha[0,step] = float(line)
        if count == 4:
            line = line.split(' ')[2]
            normal_alpha[1,step] = float(line)
        if count == 6:
            line = line.split(' ')[2]
            normal_alpha[2,step] = float(line)
        if count == 8:
            line = line.split(' ')[2]
            normal_alpha[3,step] = float(line)
        if count == 10:
            line = line.split(' ')[2]
            normal_alpha[4,step] = float(line)
        if count == 12:
            line = line.split(' ')[2]
            normal_alpha[5,step] = float(line)
        if count == 14:
            line = line.split(' ')[2]
            normal_alpha[6,step] = float(line)
        if count == 16:
            line = line.split(' ')[2]
            normal_alpha[7,step] = float(line)
        if count == 18:
            line = line.split(' ')[2]
            normal_alpha[8,step] = float(line)
        if count == 20:
            line = line.split(' ')[2]
            normal_alpha[9,step] = float(line)
        if count == 22:
            line = line.split(' ')[2]
            normal_alpha[10,step] = float(line)
        if count == 24:
            line = line.split(' ')[2]
            normal_alpha[11,step] = float(line)
            count = 0
            flag = False
            step += 1


print(normal_alpha)

f, ax = plt.subplots(figsize=(6, 2.5))
x=np.arange(normal_alpha.shape[1])
ax1 = plt.subplot(121)

lns1=ax1.plot(x, normal_alpha[0], color='#9f79ee', label='correct_loss')
lns2=ax1.plot(x, normal_alpha[1], color='#F76809', label='correct_entropy')
ax2 = ax1.twinx()
lns3 = ax2.plot(x, normal_alpha[2], color='#F1C40F', label='correct_cost')
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
legend=ax1.legend(lns, labs, loc='lower left')
frame = legend.get_frame() 
frame.set_alpha(1) 
frame.set_facecolor('none')

ax1 = plt.subplot(122)

lns1=ax1.plot(x, normal_alpha[4], color='#9f79ee', label='wrong_loss')
lns2=ax1.plot(x, normal_alpha[5], color='#F76809', label='wrong_entropy')
ax2 = ax1.twinx()
lns3 = ax2.plot(x, normal_alpha[6], color='#F1C40F', label='wrong_cost')
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
legend=ax1.legend(lns, labs, loc='lower left')
frame = legend.get_frame() 
frame.set_alpha(1) 
frame.set_facecolor('none')

plt.tight_layout()

plt.savefig('img/loss_entropy_cost_correct_wrong.png') 
plt.show(block=False)
plt.clf()

f, ax = plt.subplots(figsize=(4, 3))
x=np.arange(normal_alpha.shape[1])
ax1 = plt.subplot(111)

lns1=ax1.plot(x, normal_alpha[8], color='#9f79ee', label='total_loss')
lns2=ax1.plot(x, normal_alpha[9], color='#F76809', label='total_entropy')
ax2 = ax1.twinx()
lns3 = ax2.plot(x, normal_alpha[10], color='#F1C40F', label='total_cost')
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
legend=ax1.legend(lns, labs)
frame = legend.get_frame() 
frame.set_alpha(1) 
frame.set_facecolor('none')

plt.tight_layout()

plt.savefig('img/loss_entropy_cost_total.png') 
plt.show(block=False)
plt.clf()




