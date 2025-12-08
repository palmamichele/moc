#LSH based on random projection "Locality-Sensitive Hashing Scheme Based on p-Stable Distributions”
import numpy as np 
import ctypes
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import time #perf_counter() is safer 
from ctypes import c_double, c_size_t, c_float, POINTER, byref



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
dim = [100, 3600, 10000, 512**2, 200000] 
num_points = [1000, 5000, 10000, 60000, 80000, 1000000]
dx=euclidean
dy=euclidean
max_dist =50 #300 
L=100
K =10
h=1
num_functions = 3


os.environ['OMP_NUM_THREADS'] = '14'   # pick desired number

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
        a_flat = np.ascontiguousarray(np.array([ai.reshape(-1) for ai in a]), dtype=np.float64).reshape(-1)
        b_flat = np.ascontiguousarray(np.array([bi.reshape(-1) for bi in b]), dtype=np.float64).reshape(-1)
        x_np = np.ascontiguousarray(np.vstack(x), dtype=np.float64)  # shape (n, d)

        f_arrays = [np.ascontiguousarray(fj, dtype=np.float64).reshape(-1, 1) 
            if np.ndim(fj)==1 else np.ascontiguousarray(fj, dtype=np.float64) 
            for fj in f]
        
        a_ptr = a_flat.ctypes.data_as(POINTER(c_double))
        b_ptr = b_flat.ctypes.data_as(POINTER(c_double))
        x_ptr = x_np.reshape(-1).ctypes.data_as(POINTER(c_double))


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
            fmocs=[[] for _ in range(num_functions)]
            annfmocs = [[] for _ in range(num_functions)]
            out_len = c_size_t(0)
            #aannfmocs=[[],[],[]]
            
            timesteps.append(time.time())


            for j in range(num_functions):
                f_ptr = f_arrays[j].reshape(-1).ctypes.data_as(POINTER(c_double))
                
                
                B_lsh_ptr = lib.compute_lsh_moc_c(
                    x_ptr,
                    c_size_t(len(x)),
                    c_size_t(x_np.shape[1]),
                    f_ptr,
                    c_size_t(f_arrays[j].shape[1]),
                    a_ptr,
                    b_ptr,
                    c_size_t(K),
                    c_size_t(L),
                    c_double(max_dist),  # maximum delta range
                    c_double(h),
                    byref(out_len)
                )
                
                if not bool(B_lsh_ptr) or out_len.value == 0:
                    continue
                
                length = out_len.value
                buf_type = ctypes.c_float * length
                buf = ctypes.cast(B_lsh_ptr, POINTER(ctypes.c_float))
                B_lsh_j = np.frombuffer(buf_type.from_address(ctypes.addressof(buf.contents)), dtype=np.float32).copy()
                
                lib.free_buffer_lsh(B_lsh_ptr)
                
                for delta in t:
                    bin_index = int(np.floor(delta / h))
                    fmocs[j].append(B[j][bin_index])
                    val = 0.0
                    if bin_index < len(B_lsh_j):
                        val = B_lsh_j[bin_index]
                    annfmocs[j].append(val)

            timesteps.append(time.time())
            print(f"LSH rNN time: {timesteps[-1] - timesteps[-2]} seconds")

         



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
