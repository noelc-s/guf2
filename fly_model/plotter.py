import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def npread(file, sep=',', header=None):
    df = pd.read_csv(file, sep=sep, header=header)
    arr = np.asarray(df)
    return arr

def npwrite(arr, file):
    df = pd.DataFrame(arr)
    df.to_csv(file, header=False, index=False)

def mix(color1, color2, val):
    res = []
    for c1, c2 in zip(color1,color2):
        res.append(c1*(1-val)+c2*(val))
    return res

gray = (0.1,0.1,0,1)
blue = (0,0.5,0.9)
red = (0.9,0.2,0)

sa = []

f0 = npread('ForceCharacterization/fz_0.csv')
wb = np.arange(f0.shape[0])/100

fp = []



mag = np.arange(-5,5)
# stroke_avg = []
# for f in fp[::-1]:
#     stroke_avg.append(np.mean(f))
# stroke_avg.append(np.mean(f0))
#
# plt.plot(mag,stroke_avg)
directions = ['fx','fy','fz','mx','my','mz']

for file in range(0,6):
    for direction in range(0,6):
        stroke_avg = []
        for j in range(-5,5):
            f_j = npread('ForceCharacterization/'+directions[file]+'_{}.csv'.format(j))[:,direction]
            stroke_avg.append(np.mean(f_j[100:]))
            # fp.append(f_j)
        ax=plt.subplot(2,3,direction+1)
        ax.set_title(directions[direction])
        # for i in range(6):
        #     plt.plot(mag[i:i+2],stroke_avg[i:i+2], '.-', lw=3, ms=10, color=mix(red,gray,file/5))
        # for i in np.arange(11):
            # plt.plot(mag[i:i+2],stroke_avg[i:i+2], '.-', lw=3, ms=10)
        plt.plot(mag,stroke_avg,'.-',lw=2, ms=5)
plt.legend(directions)

# plt.xlabel("Control Magnitude")
# plt.ylabel("Stroke-Averaged $F_z$")
# plt.gca().spines["right"].set_visible(False)
# plt.gca().spines["top"].set_visible(False)
# plt.plot(wb,np.transpose(fp))
plt.show()