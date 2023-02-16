import numpy as np
from matplotlib import pyplot
import smote

##############################
#
# Example 2, with multiple classes.
#

m = 20      # Data points per circle
p = 1273    # Total additional data points.
np.random.seed(57721)   # Euler-Mascheroni

props = dict(boxstyle='square', facecolor=[0.95,0.95,0.95], edgecolor='k', alpha=1)

th = 2*np.pi*np.random.rand(m)
x0 = np.cos(th)
y0 = np.sin(th)
th = 2*np.pi*np.random.rand(m)
x1 = np.cos(th) + 0.5
y1 = np.sin(th)

x = np.hstack((x0,x1))
y = np.hstack((y0,y1))

data = np.vstack((x,y)).T
labels = np.hstack( (np.zeros(m), np.ones(m)) )

fig,ax = pyplot.subplots(2,2, 
                         sharex=True, 
                         sharey=True, 
                         figsize=(6,6),
                         constrained_layout=True
                         )


for j,k in enumerate([1,2,3,4]):
    ii, jj = int(j//2), j%2 # map to 2-by-2 grid
    
    synth_labels = np.random.choice([0,1], p)
    colors = ['b' if s else 'r' for s in synth_labels]
    generator = smote.SmoteGenerator()

    generator.params['k'] = k
    generator.fit(data,labels)

    data_synth = generator.generate(synth_labels)

    ax[ii,jj].scatter(data_synth[:,0], data_synth[:,1], c=colors, s=1)
    ax[ii,jj].scatter(data[:,0],data[:,1], c='k', s=40)
    
    ax[ii,jj].text(0.25,0.9, r'$k=%i$'%k, fontsize=14, transform=ax[ii,jj].transAxes, ha='center', va='center', bbox=props)
#

fig.suptitle('SMOTE, varying number of nearest neighbors', fontsize=18)

if False:
    fig.savefig('smote_multiclass_demo.png')

fig.show()
