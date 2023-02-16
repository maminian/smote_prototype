from matplotlib import pyplot
import numpy as np

import smote

m = 20
p = 1273
np.random.seed(57721)   # Euler-Mascheroni, if you were wondering

# controls plot's legend's properties.
props = dict(boxstyle='square', 
             facecolor=[0.8,0.8,1.0], 
             edgecolor='k', 
             alpha=1)

th = 2*np.pi*np.random.rand(m)
x = np.cos(th)
y = np.sin(th)

data = np.vstack((x,y)).T
labels = np.zeros(m)

fig,ax = pyplot.subplots(2,2, 
                         sharex=True, 
                         sharey=True, 
                         figsize=(8,8),
                         constrained_layout=True
                         )

# data_synth_all = [[smote_oneclass(data,p,k) for k in [1,2,3,4]]]
data_synth_all = []
for k in [1,2,3,4]:
    generator = smote.SmoteGenerator()
    generator.params['k'] = k
    generator.fit(data,labels)

    data_synth_all.append( generator.generate(np.zeros(p)) )
#

for j,k in enumerate([1,2,3,4]):
    ii, jj = int(j//2), j%2 # map to 2-by-2 grid
    
    ax[ii,jj].scatter(data_synth_all[j][:,0], data_synth_all[j][:,1], c='r', s=1)
    ax[ii,jj].scatter(data[:,0],data[:,1], c='k', marker=r'$\odot$', s=400)

    ax[ii,jj].set_title('k=%i'%k)

    for col,lab,marker,si in [['k','Original data',r'$\odot$',100],['r','Synthetic data','.',20]]:
        ax[ii,jj].scatter([],[],c=col,label=lab, marker=marker, s=si)
    #
    #ax[j].axis('equal')
#
ax[0,1].legend(loc='upper right')

fig.suptitle('SMOTE, varying number of nearest neighbors')


fig.show()

