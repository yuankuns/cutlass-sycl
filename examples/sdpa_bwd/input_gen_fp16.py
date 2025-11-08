import numpy as np
import torch
import argparse

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", default=8192, type=int)
    parser.add_argument("--N", default=8192, type=int)
    parser.add_argument("--K", default=4096, type=int)
    parser.add_argument("--L", default=2, type=int)
    parser.add_argument("--At", default='N', choices=['N', 'n', 'T', 't'], type=str)
    parser.add_argument("--Bt", default='N', choices=['N', 'n', 'T', 't'], type=str)
    return parser.parse_args()

def get_configs(options):
    M = options.M
    N = options.N
    K = options.K
    L = options.L
    At = options.At.upper() != 'N'
    Bt = options.Bt.upper() != 'N'
    return M, N, K, L, At, Bt

def init_tensors(M: int, N: int, K: int, L: int, At: bool, Bt: bool, dtype: torch.dtype):
    if At:
        A = torch.zeros(L, K, M, dtype=torch.float32)
    else:
        A = torch.zeros(L, M, K, dtype=torch.float32)
    if Bt:
        B = torch.zeros(L, N, K, dtype=torch.float32)
    else:
        B = torch.zeros(L, K, N, dtype=torch.float32)
    A.uniform_(-1, 1)
    B.uniform_(-1, 1)
    A = A.to(dtype)
    B = B.to(dtype)
    return A, B

def run_pt_gemm(A, At, B, Bt):
    if At:
        A = A.transpose(-1, -2)
    if Bt:
        B = B.transpose(-1, -2)
    C = torch.matmul(A, B)
    return C

def run_np_gemm(A, At, B, Bt):
    C = []
    for i in range(A.shape[0]):
        if At:
            Ai = np.transpose(A[i])
        else:
            Ai = A[i]
        if Bt:
            Bi = np.transpose(B[i])
        else:
            Bi = B[i]
        C.append(np.matmul(Ai, Bi))
    return np.array(C)

def verify(Cpt, Cnp):
    return np.allclose(Cpt.float().detach().numpy(), Cnp, atol=5e-3, rtol=5e-3)

def dump(A, B, C, dtype):
    AA = A.view(torch.uint16).detach().numpy()
    BB = B.view(torch.uint16).detach().numpy()
    CC = C.to(dtype).view(torch.uint16).detach().numpy()
    AA.tofile('A.bin')
    BB.tofile('B.bin')
    CC.tofile('C.bin')

def main():
    torch.manual_seed(123)
    dtype = torch.float16

    options = get_options()
    M, N, K, L, At, Bt = get_configs(options)
    A, B = init_tensors(M, N, K, L, At, Bt, dtype)
    Cpt = run_pt_gemm(A, At, B, Bt)
    AA = A.float().detach().numpy()
    BB = B.float().detach().numpy()
    Cnp = run_np_gemm(AA, At, BB, Bt)
    print('Verify: ', verify(Cpt, Cnp))
    dump(A, B, Cpt, dtype)

if __name__ == '__main__':
    main()
