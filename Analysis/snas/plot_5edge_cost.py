import sys
file_name = sys.argv[1]
search_name = sys.argv[2]
search_num = int(sys.argv[3])

step = 0
count = 0
flag = False

import numpy as np 
import matplotlib.pyplot as plt  
import os

if not os.path.exists('img'):
    os.makedirs('img')

normal_alpha = np.zeros((5,search_num,8))

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
            line = line.split('[[[')[1].split(']')[0].split(',')
            for i in range(normal_alpha.shape[-1]):
                normal_alpha[0,step, i] = float(line[i])
        if count == 3:
            line = line.split('[')[1].split(']')[0].split(',')
            for i in range(normal_alpha.shape[-1]):
                normal_alpha[1,step, i] = float(line[i])
        if count == 4:
            line = line.split('[')[1].split(']')[0].split(',')
            for i in range(normal_alpha.shape[-1]):
                normal_alpha[2,step, i] = float(line[i])
        if count == 5:
            line = line.split('[')[1].split(']')[0].split(',')
            for i in range(normal_alpha.shape[-1]):
                normal_alpha[3,step, i] = float(line[i])
        if count == 6:
            line = line.split('[')[1].split(']')[0].split(',')
            for i in range(normal_alpha.shape[-1]):
                normal_alpha[4, step, i] = float(line[i])
            count = 0
            flag = False
            step += 1

#with open('log_norm.txt', 'r') as f:
#    for line in f.readlines():
#        line = line.strip()
#        
#        if search_name in line:
#            flag = True
#        if flag:
#            count += 1
#        
#        if count == 2:
#            line = line.split('[[[')[1].split(',')[:-1]
#            for i in range(normal_alpha.shape[-1]-1):
#                normal_alpha[0,step, i] = float(line[i])
#        if count == 3:
#            line = line.split(']')[0]
#            normal_alpha[0,step,-1] = float(line)
#        if count == 4:
#            line = line.split('[')[1].split(',')[:-1]
#            for i in range(normal_alpha.shape[-1]-1):
#                normal_alpha[1,step, i] = float(line[i])
#        if count == 5:
#            line = line.split(']')[0]
#            normal_alpha[1,step,-1] = float(line)
#        if count == 6:
#            line = line.split('[')[1].split(',')[:-1]
#            for i in range(normal_alpha.shape[-1]-1):
#                normal_alpha[2,step, i] = float(line[i])
#        if count == 7:
#            line = line.split(']')[0]
#            normal_alpha[2,step,-1] = float(line)
#        if count == 8:
#            line = line.split('[')[1].split(',')[:-1]
#            for i in range(normal_alpha.shape[-1]-1):
#                normal_alpha[3,step, i] = float(line[i])
#        if count == 9:
#            line = line.split(']')[0]
#            normal_alpha[3,step,-1] = float(line)
#        if count == 10:
#            line = line.split('[')[1].split(',')[:-1]
#            for i in range(normal_alpha.shape[-1]-1):
#                normal_alpha[4,step, i] = float(line[i])
#        if count == 11:
#            line = line.split(']]]')[0]
#            normal_alpha[4,step,-1] = float(line)
#            count = 0
#            flag = False
#            step += 1
#
#        if step == search_num:
#            break

#normal_alpha[:,-1,:]=final_reward

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

#for i in range(normal_alpha.shape[0]):
#    for j in range(normal_alpha.shape[1]):
#     normal_alpha[i,j,:] = softmax(normal_alpha[i,j,:])
print(normal_alpha)

#for i in range(normal_alpha.shape[-1]):
x=np.arange(normal_alpha.shape[1])
# Plot the points using matplotlib 
#name = 'op_' + str(i)
#plt.title(name) 
#plt.plot(x, normal_alpha[0,:,i], '*')
#plt.plot(x, normal_alpha[0,:].sum(-1), label='edge 0')
#plt.plot(x, normal_alpha[1,:,i], '*')
plt.plot(x, normal_alpha[1,:].sum(-1), label='edge 1')
#plt.plot(x, normal_alpha[2,:,i], '*')
#plt.plot(x, normal_alpha[2,:].sum(-1), label='edge 2')
#plt.plot(x, normal_alpha[3,:,i], '*')
plt.plot(x, normal_alpha[3,:].sum(-1)/8, label='edge 3')
#plt.plot(x, normal_alpha[4,:,i], '*',color='red')
plt.plot(x, normal_alpha[4,:].sum(-1)/8,'k' , label='edge 4')
plt.legend()
#plt.xticks(x)
save_name = file_name
plt.savefig('img/'+save_name) 
plt.show(block=False)
plt.clf()

