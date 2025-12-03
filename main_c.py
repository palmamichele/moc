#LSH based on random projection "Locality-Sensitive Hashing Scheme Based on p-Stable Distributions”
import numpy as np 
import ctypes
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import time #perf_counter() is safer 
from ctypes import c_double, c_size_t, c_float, POINTER, byref


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
def call_compute_bprefix(x_np, f_np, h: float, idxf: int = 0):
    """
    x_np: array-like shape (n, dx)
    f_np: array-like shape (n, f_dim) OR shape (n,) for scalar features
    h: bin width
    idxf: ignored by C wrapper but kept for API parity
    """
    # convert to numpy arrays if necessary
    x_arr = np.asarray(x_np, dtype=np.float64)
    f_arr = np.asarray(f_np, dtype=np.float64)

    # If user passed 1-D f (list of scalars), turn into (n,1)
    if f_arr.ndim == 1:
        f_arr = f_arr.reshape(-1, 1)

    # Validate shapes
    if x_arr.ndim != 2:
        raise ValueError(f"x must be 2D (n,dx). Got shape {x_arr.shape}")
    if f_arr.ndim != 2:
        raise ValueError(f"f must be 2D (n,f_dim). Got shape {f_arr.shape}")

    n = x_arr.shape[0]
    dx_dim = x_arr.shape[1]
    if f_arr.shape[0] != n:
        raise ValueError(f"x and f must have same number of rows. x.rows={n}, f.rows={f_arr.shape[0]}")

    # ensure contiguous double arrays
    x_flat = np.ascontiguousarray(x_arr.reshape(-1), dtype=np.float64)
    f_flat = np.ascontiguousarray(f_arr.reshape(-1), dtype=np.float64)

    x_ptr = x_flat.ctypes.data_as(POINTER(c_double))
    f_ptr = f_flat.ctypes.data_as(POINTER(c_double))

    out_len = c_size_t(0)
    res_ptr = lib.compute_bprefix_c(
        x_ptr,
        c_size_t(n),
        c_size_t(dx_dim),
        f_ptr,
        c_size_t(f_arr.shape[1]),
        c_size_t(idxf),     # harmless, ignored by wrapper
        c_double(h),
        byref(out_len)
    )

    if not bool(res_ptr):
        # Either Bprefix was empty or C wrapper failed.
        return []

    length = out_len.value
    # create numpy array from returned pointer and copy to Python-owned memory
    buf_type = ctypes.c_float * length
    buf = ctypes.cast(res_ptr, POINTER(ctypes.c_float))
    np_result = np.frombuffer(buf_type.from_address(ctypes.addressof(buf.contents)), dtype=np.float32).copy()

    # free C buffer
    lib.free_buffer(res_ptr)

    return np_result.tolist()


np.random.seed(0)
dim = [3600, 10000, 512**2] 
num_points = [1000, 5000, 10000, 60000, 80000]
dx=euclidean
dy=euclidean
max_dist =50 #300 
L=100
K =10
h=1
num_functions = 3


os.environ['OMP_NUM_THREADS'] = '14'   # pick desired number


lib = ctypes.CDLL("./libbprefix.so")
lib.compute_bprefix_c.argtypes = [
    POINTER(c_double),  # x_data
    c_size_t,           # n
    c_size_t,           # dx_dim
    POINTER(c_double),  # f_data (for idxf)
    c_size_t,           # f_dim
    c_size_t,           # idxf (unused here)
    c_double,           # h
    POINTER(c_size_t)   # out_len (pointer)
]
lib.compute_bprefix_c.restype = POINTER(c_float)
lib.free_buffer.argtypes = [POINTER(c_float)]
lib.free_buffer.restype = None


lib.omp_debug_print()

def f1(x):
    return np.dot(2*np.ones_like(x), x)


def f2(x):
    return np.sum(x**2)


def f3(x):
    return np.sum(x**3)



b = [np.random.uniform(0, max_dist, size=K) for _ in range(L)]
for d in dim:
    a=[np.random.randn(K, d) for _ in range(L)]
    for n in num_points:
        x = [np.random.rand(d)*np.random.randint(-max_dist/2,max_dist/2) for _ in range(n)]
        

        f=[[],[],[]]
        f[0] = [f1(x1) for x1 in x]
        f[1]=[f2(x1) for x1 in x]
        f[2]=[f3(x1) for x1 in x]


        print(f'Ready {n}-{d}')

        timesteps = []
        timesteps.append(time.time())
        B = [None, None, None]
        for k in range(num_functions):
            B[k] = call_compute_bprefix(x, f[k], h, idxf=k) #remove idxf arg
        exactCost=time.time()-timesteps[-1]
        timesteps.append(time.time())
        print(f"exact rNN time {exactCost} seconds")

        for experiment in range(1):
            t= np.arange(0, max_dist+1, h)
            fmocs=[[],[],[]]
            annfmocs=[[],[],[]]
            #aannfmocs=[[],[],[]]
            
            timesteps.append(time.time())

        
            for delta in t:

                lsh = EuclideanLSH(dim=d, w=delta, k=K,L=L, a=a, b=b)
                
                for i, vec in enumerate(x):
                    lsh.insert(vec,i)
                    
                pairs = lsh.rNN(delta, x)

                for j in range(num_functions): #the number of functions
                    annfmocs[j].append(annmoc(pairs,f[j],euclidean))
                    timesteps.append(time.time())
                    print(f"LSH rNN time for delta {delta}: {timesteps[-1] - timesteps[-2]} seconds")
                    fmocs[j].append(B[j][int(np.floor(delta/h))])
                    timesteps.append(time.time())
         


            fig, ax = plt.subplots(3, 3, figsize=(8, 10))


            for j in range(num_functions):
                ax[j, 0].plot(t, fmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
                ax[j, 0].set_title(f"MOC for f{j}")
                ax[j, 1].plot(t, annfmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
                ax[j, 1].set_title(f"ANN MOC for f{j}")
                #ax[j, 2].plot(t, aannfmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
                #ax[j, 2].set_title(f"aANN MOC for f{j}")

        

            fig.tight_layout()
            plt.savefig(f"B{K}-{L}-{n}-{d}-{dx.__name__}-{dy.__name__}moc_comparison{experiment+1}.png", dpi=300)  # high-resolution PNG
            plt.clf() 
            plt.close('all')
