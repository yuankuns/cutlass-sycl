import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy as np
def is_close(refe: torch.Tensor,
             test: torch.Tensor):
    test = test.to(torch.float32)
    refe = refe.to(torch.float32)
    cosfactor = F.cosine_similarity(test.reshape(-1), refe.reshape(-1), dim=0) > 0.99
    allclose = torch.allclose(test, refe, atol=3e-3, rtol=3e-3)
    return cosfactor and allclose

def set_dict(dump_dict, name, value):
    if value.dtype == torch.bfloat16 or value.dtype == torch.float16:
        dump_dict[name] = value.detach().clone().view(torch.uint16).numpy()
    elif value.dtype == torch.bool:
        dump_dict[name] = value.detach().clone().to(torch.uint16).numpy()
    else:
        dump_dict[name] = value.detach().clone().numpy()

def test_reg_reuse(dtype,
                   seed: int,
                   batch: int,
                   num_heads_q: int,
                   num_heads_kv: int,
                   seq_len_qo: int,
                   seq_len_kv: int,
                   head_size_qk: int,
                   head_size_vo: int,
                   is_causal:bool=False,
                   is_bhsd:bool=True):
    q = torch.randn(batch, num_heads_q, seq_len_qo, head_size_qk).to(dtype)
    k = torch.randn(batch, num_heads_kv, seq_len_kv, head_size_qk).to(dtype)
    v = torch.randn(batch, num_heads_kv, seq_len_kv, head_size_vo).to(dtype)
    do = torch.randn(batch, num_heads_q, seq_len_qo, head_size_vo).to(dtype)
    p = torch.matmul(q, k.transpose(-1, -2))
    dp = torch.matmul(do, v.transpose(-1, -2))
    dk = torch.matmul(dp.transpose(-1, -2), q)
    dv = torch.matmul(torch.transpose(do, -1, -2), p)
    dq = torch.matmul(dp, k)
    dump_dict = {}
    print(f"seed {seed} bsz {batch} nh_q {num_heads_q} nh_kv {num_heads_kv} sl_qo {seq_len_qo} sl_kv {seq_len_kv} hs_qk {head_size_qk} hs_vo {head_size_vo} is_causal {is_causal} is_bhsd {is_bhsd}")
    set_dict(dump_dict, 'q', q)
    set_dict(dump_dict, 'k', k)
    set_dict(dump_dict, 'v', v)
    set_dict(dump_dict, 'do', do)
    set_dict(dump_dict, 'dp', dp)
    set_dict(dump_dict, 'p', p)
    set_dict(dump_dict, 'dv', dv)
    set_dict(dump_dict, 'dk', dk)
    set_dict(dump_dict, 'dq', dq)
    shape = np.array([batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo, is_causal, is_bhsd], dtype=np.int32)
    dump_dict['shape'] = shape
    np.savez(f'reg_reuse_{batch}_{num_heads_q}_{num_heads_kv}_{seq_len_qo}_{seq_len_kv}_{head_size_qk}_{head_size_vo}_{int(is_causal)}_{int(is_bhsd)}.npz', **dump_dict)

def loop_run():
    for h in [4]:
        # for seq_q in list(range(512, 512+32)):
        #     for seq_k in list(range(512, 512+32)):
        for seq_q in [512, 513, 523, 527, 535, 543]:
            for seq_k in [512, 513, 523, 527, 535, 543]:
                for dim in [64, 96, 128, 192]:
                    # print('test_run', 4, 4, h, seq_q, seq_k, dim, dim)
                    # bhsd
                    test_sdpa(torch.float16, 123, 4, 4, h, seq_q, seq_k, dim, dim, is_causal=True, is_bhsd = True)
                    GRAD_DICT = {}
                    # bshd
                    test_sdpa(torch.float16, 123, 4, 4, h, seq_q, seq_k, dim, dim, is_causal=True, is_bhsd = False)
                    GRAD_DICT = {}

if __name__ == '__main__':
    test_reg_reuse(torch.float16, 123, 4, 4, 4, 512, 512, 128, 128, is_causal=False)
    # test_sdpa(torch.float16, 123, 4, 4, 4, 513, 784, 128, 128, is_causal=True)
    # GRAD_DICT = {}
    # test_sdpa(torch.float16, 123, 4, 4, 4, 513, 784, 128, 128, is_causal=False)
    # GRAD_DICT = {}
    # test_sdpa(torch.float16, 123, 4, 4, 2, 513, 784, 128, 128, is_causal=False)
    # GRAD_DICT = {}
    # test_sdpa(torch.float16, 123, 4, 4, 2, 513, 784, 128, 128, is_causal=True)
    # GRAD_DICT = {}
    # test_sdpa(torch.float16, 123, 4, 4, 1, 513, 784, 128, 128, is_causal=False)
    # GRAD_DICT = {}
    # test_sdpa(torch.float16, 123, 4, 4, 1, 513, 784, 128, 128, is_causal=True)
    # GRAD_DICT = {}
    # test_sdpa(torch.bfloat16, 123, 4, 4, 2, 513, 513, 128, 64, is_causal=True)
    # GRAD_DICT = {}
    # test_sdpa(torch.bfloat16, 123, 4, 4, 2, 513, 513, 128, 64, 0.3, is_causal=False)
    # GRAD_DICT = {}
    # test_sdpa(torch.bfloat16, 123, 4, 4, 2, 513, 513, 128, 64, is_causal=False)
    # GRAD_DICT = {}
    # GRAD_DICT = {}
    # test_sdpa(torch.bfloat16, 123, 4, 4, 4, 513, 513, 128, 64, is_causal=False)
    # test_sdpa(torch.bfloat16, 123, 4, 4, 1, 513, 513, 128, 64, False)
    # test_sdpa(torch.bfloat16, 123, 4, 4, 4, 1024, 513, 128, 128)
    # test_sdpa(123, 2, 16, 1, 513, 513, 128, 128)
