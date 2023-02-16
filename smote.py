
class SmoteGenerator:
    
    def __init__(self):
        self.data = None
        self.labels = None
        self.d_mat = None
        self.params = {}
        self.params['k'] = 1    # Number of nearest neighbors to choose from

        self.results = {}
    #

    def fit(self,data,labels=None):
        '''
        Args:
            - data: training data arranged in rows, 
                shape m,n (m data points of dimension n)
            - labels: training labels. If not specified, assumed to be all of the same class.
                NOT IMPLEMENTED: per-class sampling.
        '''
        import numpy as np
        from scipy.spatial import distance_matrix

        m,n = np.shape(data)

        self.data = np.array(data)
        if len(labels)==0:
            self.labels = np.zeros(m, dtype=int)
        else:
            self.labels = np.array(labels)
        #

        d_mat = distance_matrix(data,data)
        labels_u = np.unique(self.labels)

        self.results['eq_tr'] = {lu: np.where(labels==lu)[0] for lu in labels_u}
        self.results['nearest_neighbors'] = {}
        # for lu in labels_u:
        #     self.results['nearest_neighbors']

        # Get the k nearest neighbors *belonging to the same class* to the point.
        self.results['nearest_neighbors'] = {}
        for i in range(m):
            all_neighbors = d_mat[i]
            sameclass = all_neighbors[ self.results['eq_tr'][self.labels[i]] ]
            closest_rel = np.argsort(sameclass)[1:self.params['k']+1]
            closest = self.results['eq_tr'][self.labels[i]][closest_rel]

            self.results['nearest_neighbors'][i] = closest
        #

        return
    #

    def generate(self, synthlabels=None):
        '''
        Generate synthetic labels based on the input data.

        Inputs:
            synthlabels : list-like of labels for which corresponding synthetic
                data is requested. If None (default), it's assumed the data 
                all come from the same class, with label 0. 
                This is *not* safe with respect to a user training with multiple 
                labels and forgetting to provide labels here.
        Outputs:
            synthdata : numpy array of shape len(synthlabels)-by-n, where n is
                the dimensionality of the input data. Synthetic data 
                is generated on a per-class basis.
        '''
        import numpy as np
        m,n = np.shape(self.data)
        ms = len(synthlabels)

        synthlabels_u = np.unique(synthlabels)
        synthdata = np.zeros( (ms, n) )

        eq_sl = {slu: np.where(synthlabels==slu)[0] for slu in synthlabels_u}

        nn = self.results['nearest_neighbors']
        eq_tr = self.results['eq_tr']


        for i,slu in enumerate(synthlabels_u):
            p_t = len(eq_tr[slu])   # Number of data points of the class in the training data
            p_s = len(eq_sl[slu])   # Number of requested synth data points in the class

            if p_s%p_t==0:  # If we happened to request a perfect multiple of the original data...
                nreps = p_s//p_t
            else: # Else, round up to the next multiple, and later shuffle the remainder data to avoid possible biases.
                nreps = p_s//p_t + 1
            #

            # Loop over the data points in the training data.
            newData = np.zeros( (p_t*nreps, n) )
            # import pdb
            # pdb.set_trace()
            idx = 0
            for k,ptr in enumerate(eq_tr[slu]):

                selections = np.random.choice(nn[ptr], nreps, replace=True)

                for j in selections:

                    # diff = _data[neighbor_index,:] - _data[index,:]
                    t = np.random.rand()
                    newData[idx] = (1-t)*self.data[ptr] + t*self.data[j]

                    idx += 1
                #
            #

            # Collect the appropriate data within the class.
            part0 = newData[:(nreps-1)*p_t]
            shuffle = np.random.permutation(np.arange((nreps-1)*p_t,p_s))
            part1 = newData[shuffle]
            synthdata_slu = np.vstack( (part0,part1) )

            # Merge in to master array.
            synthdata[eq_sl[slu]] = synthdata_slu
        #

        return synthdata
    #
