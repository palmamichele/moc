#LSH based on random projection "Locality-Sensitive Hashing Scheme Based on p-Stable Distributions”
import numpy as np 
import ctypes
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import time #perf_counter() is safer 
from ctypes import c_double, c_size_t, c_float, POINTER, byref

np.random.seed(0)
os.environ['OMP_NUM_THREADS'] = '14' 
lib = ctypes.CDLL("./libcombined.dylib")
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

lib.compute_lsh_moc_c.argtypes = [
    POINTER(c_double),  # x_data
    c_size_t,           # n
    c_size_t,           # dx_dim
    POINTER(c_double),  # f_data
    c_size_t,           # f_dim
    POINTER(c_double),  # a_data (L * K * dx_dim)
    POINTER(c_double),  # b_data (L * K)
    c_size_t,           # K
    c_size_t,           # L
    c_double,           # w (bucket width)
    c_double,           # h (bin width)
    POINTER(c_size_t)   # out_len
]
lib.compute_lsh_moc_c.restype = POINTER(c_float)
lib.free_buffer_lsh.argtypes = [POINTER(c_float)]
lib.free_buffer_lsh.restype = None

lib.compute_max_distance_c.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),  # x_data
    c_size_t,  # n
    c_size_t   # dx_dim
]
lib.compute_max_distance_c.restype = c_double


lib.compute_lipschitz_c.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),  # x_data
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),  # f_data
    c_size_t,  # n
    c_size_t,  # dx_dim
    c_size_t   # f_dim
]
lib.compute_lipschitz_c.restype = c_double


def compute_max_distance(x):
    """
    Docstring for compute_max_distance
    
    :param x: np.ascontiguousarray(x, dtype=np.float64)
    """
    n, d = x.shape
    return lib.compute_max_distance_c(x, n, d)


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



def compute_moc(X, F, delta_step):
    """
    Given a set of points X, and related function values, for different functions, compute exact moc for them across meaningful delta values 
    if functions are vector valued this computes the joint moc (e.g. not moc coordinate wise)

    coordinate moc on each F[i] could be computed as F_k = F[:, k].reshape(n, 1), Bprefix_k = compute_bprefix(X, F_k, h)...


    :param X: list of n datapoints, each d-dimensional np array (they are the common input to all functions)
    :param F: list of p functions, each F[i] contains list (n datapoints, k=1 dimensional) of corresponding function values

    """
    num_functions = len(F)

    #do the conversions for the C bindings 
    x_np = np.ascontiguousarray(X, dtype=np.float64)
    x_np = np.ascontiguousarray(np.vstack(x_np), dtype=np.float64)  # shape (n, d)

    #determine the max euclidean distance between points, delta will range from 0 to max_dist
    max_dist = compute_max_distance(x_np)
    print(max_dist)
    deltas= np.arange(0, max_dist+delta_step, delta_step) #from 0 to max_dist
    print(deltas)

    #apply exact moc computation
    #x_ptr = x_np.reshape(-1).ctypes.data_as(POINTER(c_double))
    f_arrays = [np.ascontiguousarray(fj, dtype=np.float64).reshape(-1, 1) 
                 if np.ndim(fj)==1 else np.ascontiguousarray(fj, dtype=np.float64) 
                 for fj in F]

    B = [None]*num_functions
    for k in range(num_functions):
        B[k] = call_compute_bprefix(x_np, f_arrays[k], delta_step, idxf=k) #remove idxf arg in cleanup
    #         exactCost=time.time()-timesteps[-1]
    #         timesteps.append(time.time())
    #         print(f"exact rNN time {exactCost} seconds")

    
    fmocs=[[] for _ in range(num_functions)]
    #             annfmocs = [[] for _ in range(num_functions)]
    out_len = c_size_t(0)
    #             #aannfmocs=[[],[],[]]
                
    #             timesteps.append(time.time())
            

    for k in range(num_functions):
        #f_ptr = f_arrays[k].reshape(-1).ctypes.data_as(POINTER(c_double))
        
        # B_lsh_ptr = lib.compute_lsh_moc_c(
        #     x_ptr,
        #     c_size_t(len(x)),
        #     c_size_t(x_np.shape[1]),
        #     f_ptr,
        #     c_size_t(f_arrays[j].shape[1]),
        #     a_ptr,
        #     b_ptr,
        #     c_size_t(K),
        #     c_size_t(L),
        #     c_double(max_dist),  # maximum delta range
        #     c_double(h),
        #     byref(out_len)
        # )
        
        # if not bool(B_lsh_ptr) or out_len.value == 0:
        #     continue
        
        # length = out_len.value
        # buf_type = ctypes.c_float * length
        # buf = ctypes.cast(B_lsh_ptr, POINTER(ctypes.c_float))
        # B_lsh_j = np.frombuffer(buf_type.from_address(ctypes.addressof(buf.contents)), dtype=np.float32).copy()
        
        #lib.free_buffer_lsh(B_lsh_ptr)
        
        for t in deltas:
            bin_index = int(np.floor(t / delta_step))
            if bin_index >= len(B[k]):
                fmocs[k].append(B[k][-1])
            else:
                fmocs[k].append(B[k][bin_index])
            #val = 0.0
            # if bin_index < len(B_lsh_j):
            #     val = B_lsh_j[bin_index]
            # annfmocs[j].append(val)







    return fmocs,[], deltas


# def euclidean(x,y):
#     return np.linalg.norm(x - y)

def plot_mocs(fmocs, deltas, savename):
    """
    mocs contains moc values (from 0 to max_dist) for k different functions, it is an array of size k x max_dist+1 x 
    """

    num_functions = len(fmocs)
    fig, ax = plt.subplots(num_functions, 1)
    ax = np.ravel(ax)

    for j in range(num_functions):
        ax[j].plot(deltas, fmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
        ax[j].set_title(f"exact MOC for f{j}")

        #later add column for lshmoc
        #ax[j, 1].plot(t, annfmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
        #ax[j, 1].set_title(f"ANN MOC for f{j}")
        #ax[j, 2].plot(t, aannfmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
        #ax[j, 2].set_title(f"aANN MOC for f{j}")

    fig.tight_layout()
    plt.savefig(f"{savename}", dpi=300)  # high-resolution PNG
    plt.clf() 
    plt.close('all')




if __name__=="__main__":

    savename="a.png" #{max_dist}B{K}-{L}-{n}-{d}-{dx.__name__}-{dy.__name__}moc_comparison{experiment+1}

    #simple test
    X = [[1],[2], [3], [4], [5], [6]]
    F = [[[1],[3],[2],[5],[4],[6]], [[3],[2],[3],[3],[4],[3]]] #each f value must also be seen as q-dim array

    fmocs, deltas = compute_moc(X,F,1)
    plot_mocs(fmocs, deltas, savename)

    #generate some random datapoints
    dim = [100, 512**2, 200000] 
    num_points = [1000, 5000, 10000, 60000, 80000, 300000]
    

    def f1(x):
        return np.dot(2*np.ones_like(x), x)


    def f2(x):
        return np.sum(x**2)


    def f3(x):
        return np.sum(x**3)

    #lsh moc pars s
    L=100
    K =10

    for d in dim:
        for n in num_points:
            X = [np.random.rand(d)*np.random.randint(-50/2,50/2) for _ in range(n)] 
            F=[[],[], []]
            F[0] = [f1(x1) for x1 in X]
            F[1]=[f2(x1) for x1 in X]
            F[2]=[f3(x1) for x1 in X]
            
            fmocs, lshmocs, deltas = compute_moc(X,F,1)
            plot_mocs(fmocs, deltas, f"{d}-{n}.png")






    # for d in dim:

    #     a=[np.random.randn(K, d) for _ in range(L)]

    #     for n in num_points:
    #         x = [np.random.rand(d)*np.random.randint(-300/2,300/2) for _ in range(n)]

            

    #         b = [np.random.uniform(0, max_dist, size=K) for _ in range(L)]

    #         f=[[],[],[]]
    #         f[0] = [f1(x1) for x1 in x]
    #         f[1]=[f2(x1) for x1 in x]
    #         f[2]=[f3(x1) for x1 in x]
    #         a_flat = np.ascontiguousarray(np.array([ai.reshape(-1) for ai in a]), dtype=np.float64).reshape(-1)
    #         b_flat = np.ascontiguousarray(np.array([bi.reshape(-1) for bi in b]), dtype=np.float64).reshape(-1)
            

         

    #         
            
    #         a_ptr = a_flat.ctypes.data_as(POINTER(c_double))
    #         b_ptr = b_flat.ctypes.data_as(POINTER(c_double))
            


    #         print(f'Ready {n}-{d}')

    #         timesteps = []
    #         timesteps.append(time.time())
    #        

    #         for experiment in range(1):
    #             t= np.arange(0, max_dist+1, h)
    #             fmocs=[[] for _ in range(num_functions)]
    #             annfmocs = [[] for _ in range(num_functions)]
    #             out_len = c_size_t(0)
    #             #aannfmocs=[[],[],[]]
                
    #             timesteps.append(time.time())
            

    #             for j in range(num_functions):
    #                 f_ptr = f_arrays[j].reshape(-1).ctypes.data_as(POINTER(c_double))
                    
                    
    #                 B_lsh_ptr = lib.compute_lsh_moc_c(
    #                     x_ptr,
    #                     c_size_t(len(x)),
    #                     c_size_t(x_np.shape[1]),
    #                     f_ptr,
    #                     c_size_t(f_arrays[j].shape[1]),
    #                     a_ptr,
    #                     b_ptr,
    #                     c_size_t(K),
    #                     c_size_t(L),
    #                     c_double(max_dist),  # maximum delta range
    #                     c_double(h),
    #                     byref(out_len)
    #                 )
                    
    #                 if not bool(B_lsh_ptr) or out_len.value == 0:
    #                     continue
                    
    #                 length = out_len.value
    #                 buf_type = ctypes.c_float * length
    #                 buf = ctypes.cast(B_lsh_ptr, POINTER(ctypes.c_float))
    #                 B_lsh_j = np.frombuffer(buf_type.from_address(ctypes.addressof(buf.contents)), dtype=np.float32).copy()
                    
    #                 lib.free_buffer_lsh(B_lsh_ptr)
                    
    #                 for delta in t:
    #                     bin_index = int(np.floor(delta / h))
    #                     fmocs[j].append(B[j][bin_index])
    #                     val = 0.0
    #                     if bin_index < len(B_lsh_j):
    #                         val = B_lsh_j[bin_index]
    #                     annfmocs[j].append(val)

    #             timesteps.append(time.time())
    #             print(f"LSH rNN time: {timesteps[-1] - timesteps[-2]} seconds")

            



    #             fig, ax = plt.subplots(3, 3, figsize=(8, 10))

    #             for j in range(num_functions):
    #                 ax[j, 0].plot(t, fmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
    #                 ax[j, 0].set_title(f"MOC for f{j}")
    #                 ax[j, 1].plot(t, annfmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
    #                 ax[j, 1].set_title(f"ANN MOC for f{j}")
    #                 #ax[j, 2].plot(t, aannfmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
    #                 #ax[j, 2].set_title(f"aANN MOC for f{j}")

            

    #             fig.tight_layout()
    #             plt.savefig(f"{max_dist}B{K}-{L}-{n}-{d}-{dx.__name__}-{dy.__name__}moc_comparison{experiment+1}.png", dpi=300)  # high-resolution PNG
    #             plt.clf() 
    #             plt.close('all')

