#LSH based on random projection "Locality-Sensitive Hashing Scheme Based on p-Stable Distributions”
import numpy as np 
from collections import defaultdict
import matplotlib.pyplot as plt
import time





#deterministic functions 
class E2LSH:
    def __init__(self,dim, w=4.0, k=20, P=2**32-5, tableSize=100000):
        self.dim = dim
        self.w = w
        self.k = k
        self.P = P
        self.a = np.random.randn(k, dim)
        self.b = np.random.uniform(0, w, size=k)
        self.bins = defaultdict(list)
        self.binsweights = np.random.randn(k)
        self.fingerprintsweights = np.random.randn(k)
        self.fingerprints = defaultdict(list)
        self.tableSize=tableSize

    def _hash(self, x):
        """Compute hash key for vector x across k hashes"""
        return tuple(np.floor((self.a @ x + self.b) / self.w).astype(int))
    
    def insert(self, x, label):
        key=self._hash(x) #hi(x)
        nkey = (self.binsweights.dot(key) % self.P) % self.tableSize
        fingerprint = self.fingerprintsweights.dot(key) % self.P
        self.bins[nkey].append((label,x)) #put pointer to x
        self.fingerprints[label] = fingerprint 

    def rNN(self, r, x):
        pairs = set()
        for bucket in self.bins.values():
            idxs = [label for label, _ in bucket]
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    pairs.add(tuple(sorted((idxs[i], idxs[j])))) #to impose tuple order for the set, avoids adding same tuple

        final_pairs = []
        for i, j in pairs:
            #this are only pairs from the same bucket (checking fingerprints)
            if  self.fingerprints[i]==self.fingerprints[j] and np.linalg.norm(x[i] - x[j]) <= r:
                final_pairs.append((i, j))
        return final_pairs 


class EuclideanLSH:
    def __init__(self, dim, w=4.0, k=20, L=20, a=None, b=None):
        """
        LSH for Euclidean space.
        dim: dimension of input vectors
        w: bucket width
        k: number of hash functions per hash table
        L: number of hash tables
        """
        self.dim = dim
        self.w = w
        self.k = k
        self.L = L
        
        self.a=a
        if a==None:
            self.a = [np.random.randn(k, dim) for _ in range(L)]
        
        self.b=b
        if b==None:
            self.b = [np.random.uniform(0, w, size=k) for _ in range(L)]
        self.tables = [defaultdict(list) for _ in range(L)]

    def _hash(self, x, i):
        """Compute hash key for vector x in table i across k hashes"""
        a = self.a[i]
        b = self.b[i]
        return tuple(np.floor((a @ x + b) / self.w).astype(int))
    
    def insert(self, vec, label):
        """Insert a vector with given label into all hash tables."""
        for l in range(self.L):
            key = self._hash(vec, l)
            self.tables[l][key].append((label, vec))#modify to just save pointer to original vector (or idx)
    
    def query(self, vec, m=1):
        """Query m nearest candidates for given vector"""
        candidates = set()
        for l in range(self.L):
            key = self._hash(vec, l)
            for label, cand_vec in self.tables[l].get(key, []):
                candidates.add((label, tuple(cand_vec)))

        candidates = [ (label, np.linalg.norm(vec - np.array(cand_vec))) for label, cand_vec in candidates ]
        candidates.sort(key=lambda x: x[1])
        return candidates[:m]
    
    def rNN(self, r, x):
        pairs = set()
        for l in range(self.L):
            for bucket in self.tables[l].values():
                idxs = [label for label, _ in bucket]
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        pairs.add(tuple(sorted((idxs[i], idxs[j])))) #to impose tuple order for the set, avoids adding same tuple

        final_pairs = []
        for i, j in pairs:
            #this are only pairs from the same bucket
            if np.linalg.norm(x[i] - x[j]) <= r:
                final_pairs.append((i, j))
        return final_pairs 
    


def eemoc(B,z):
    #efficient exact moc
    #we need to do line search in B up to bin z
    mx = 0
    searchSet = [i  for i in  B.keys() if i<= z]
    for i in searchSet:
        
        mx = max(mx, B[i])
    return mx

    

def moc(X,Y, dx, dy, t):
    #assuming X, Y are metric spaces (discrete sets), 
    #where Y[i] corresponds to function evaluated at X[i].
    maxf=0
    for i in range(len(X)):
        for j in range(len(X)):
            if dx(X[i],X[j])<=t:
                maxf = max(maxf, dy(Y[i],Y[j]))
    return maxf

def annmoc(X,Y,dy):
    #assuming X contains pairs of (indices of ) points at distance <= t
    maxf=0
    for i,j in X:
        maxf = max(maxf, dy(Y[i],Y[j]))
    return maxf


def euclidean(x,y):
    return np.linalg.norm(x - y)



np.random.seed(0)
dim = [3600, 10000, 512**2] 
num_points = [10000,100000]
dx=euclidean
dy=euclidean
max_dist = 300
L=100
K =10
h=1




def f1(x):
    return np.dot(2*np.ones_like(x), x)


def f2(x):
    return np.sum(x**2)


def f3(x):
    return np.sum(x**3)


functions=[f1, f2, f3]

b = [np.random.uniform(0, max_dist, size=K) for _ in range(L)]
for d in dim:
    a=[np.random.randn(K, d) for _ in range(L)]
    for n in num_points:
        x = [np.random.rand(d)*np.random.randint(-max_dist/2,max_dist/2) for _ in range(n)]
        

        f=[[],[],[]]
        f[0] = [f1(x1) for x1 in x]
        f[1]=[f2(x1) for x1 in x]
        f[2]=[f3(x1) for x1 in x]

        B=[[],[],[]]

        for experiment in range(1):
            t= np.arange(0, max_dist+1, h)
            fmocs=[[],[],[]]
            annfmocs=[[],[],[]]
            aannfmocs=[[],[],[]]
            timesteps = []
            timesteps.append(time.time())

            for idxf in range(3):
                for i in range(len(x)):
                    for j in range(len(x)):
                        sij=dx(x[i],x[j])
                        dij=dy(f[idxf][i],f[idxf][j])
                        binindex = int(np.floor(sij/h))
                        if binindex >= len(B[idxf]):
                            B[idxf].extend([0.0] * (binindex + 1 - len(B[idxf])))
                        B[idxf][binindex]=max(B[idxf][binindex], dij)

            Bprefix= [np.maximum.accumulate(B[0]), np.maximum.accumulate(B[1]), np.maximum.accumulate(B[2]) ]
            exactCost=time.time()
            timesteps.append(exactCost)

            for delta in t:
                lsh = EuclideanLSH(dim=d, w=delta, k=K,L=L, a=a, b=b)
                
                for i, vec in enumerate(x):
                    lsh.insert(vec,i)
                    
                pairs = lsh.rNN(delta, x)

                for j in range(3): #the number of functions
                    annfmocs[j].append(annmoc(pairs,f[j],euclidean))
                    timesteps.append(time.time())
                    print(f"LSH rNN time for delta {delta}: {timesteps[-1] - timesteps[-2]} seconds")
                    fmocs[j].append(Bprefix[j][int(np.floor(delta/h))])
                    timesteps.append(time.time())
                    print(f"Exact rNN time for delta {delta}: {timesteps[-1] - timesteps[-2]+ (exactCost)/len(t)} seconds")

                alsh = E2LSH(dim=d, w=delta, k=K, tableSize=n)
                for i,vec in enumerate(x):
                    alsh.insert(vec, i)
                apairs = alsh.rNN(delta, x)

                for j in range(3):
                    aannfmocs[j].append(annmoc(apairs, f[j],euclidean))
                    timesteps.append(time.time())
                    print(f"aLSH rNN time for delta {delta}: {timesteps[-1] - timesteps[-2]} seconds")
            


            fig, ax = plt.subplots(3, 3, figsize=(8, 10))

            for j in range(3):
                ax[j, 0].plot(t, fmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
                ax[j, 0].set_title(f"MOC for f{j}")
                ax[j, 1].plot(t, annfmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
                ax[j, 1].set_title(f"ANN MOC for f{j}")
                ax[j, 2].plot(t, aannfmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
                ax[j, 2].set_title(f"aANN MOC for f{j}")

            xlims = ax[0, 0].get_xlim()
            ylims = ax[0, 0].get_ylim()

            for i in range(3):
                for j in range(2):
                    ax[i, j].set_xlim(xlims)
                    #ax[i, j].set_ylim(ylims)

            fig.tight_layout()
            plt.savefig(f"B{K}-{L}-{n}-{d}-{dx.__name__}-{dy.__name__}moc_comparison{experiment+1}.png", dpi=300)  # high-resolution PNG
            plt.clf() 
            plt.close('all')
