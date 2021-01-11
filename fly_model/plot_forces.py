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

f0 = npread('data/fz_ctrl00.csv')
wb = np.arange(f0.shape[0])/100

fp = []
for i in range(5):
	fp.append(npread('data/fz_ctrlp{}.csv'.format(i+1)))

fn = []
for i in range(5):
	fn.append(npread('data/fz_ctrln{}.csv'.format(i+1)))

# plt.plot(wb,f0, color=gray)
# for i, data in enumerate(fp):
# 	plt.plot(wb,data, color=mix(gray,red,0.25*(i+2)))
# for i, data in enumerate(fn):
# 	plt.plot(wb,data, color=mix(gray,blue,0.25*(i+2)))

# plt.gca().spines["right"].set_visible(False)
# plt.gca().spines["top"].set_visible(False)
# plt.legend(["Hovering","_nolegend_","_nolegend_","$-F_z$ mode","_nolegend_","_nolegend_","$+F_z$ mode"], loc=3)
# plt.xlabel("Time (wingbeats)")
# plt.ylabel("$F_z$ (au)")
# plt.ylim([-15,30])
# plt.show()

mag = np.arange(-5,6)
stroke_avg = []
for f in fp[::-1]:
	stroke_avg.append(np.mean(f))
stroke_avg.append(np.mean(f0))
for f in fn:
	stroke_avg.append(np.mean(f))

# plt.plot(mag,stroke_avg)
for i in range(6):
	plt.plot(mag[i:i+2],stroke_avg[i:i+2], '.-', lw=3, ms=10, color=mix(red,gray,i/5))
for i in np.arange(6,11):
	plt.plot(mag[i:i+2],stroke_avg[i:i+2], '.-', lw=3, ms=10, color=mix(gray,blue,(i-5)/5))

plt.xlabel("Control Magnitude")
plt.ylabel("Stroke-Averaged $F_z$")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.show()