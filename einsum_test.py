import torch


def test0():
    q = torch.randn(1, 3, 4, 5)
    k = torch.randn(1, 6, 4, 5)

    aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
    print(aw.shape)  # 1, 4, 3, 6

    q_transposed = q.permute(0, 2, 3, 1)
    k_transposed = k.permute(0, 2, 3, 1)
    result = torch.matmul(q_transposed.transpose(-2, -1), k_transposed)

    print(result.shape)

    print(torch.sum(torch.abs(aw - result)))


def test1():
    aw = torch.randn(1, 4, 3, 6)
    v = torch.randn(1, 6, 4, 5)
    x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
    print(x.shape)  # 1, 3, 4, 5

    v_permuted = v.permute(0, 2, 1, 3)
    x_matmul = torch.matmul(aw, v_permuted).permute(0, 2, 1, 3)
    print(x_matmul.shape)

    print(torch.sum(torch.abs(x - x_matmul)))


def test2():
    embed = torch.randn(1, 4, 5, 7, 8)
    guide = torch.randn(1, 6, 4, 5)
    aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
    print(aw.shape)
    
    guide_permuted = guide.permute(0, 2, 3, 1)
    embed_permuted = embed.permute(0, 1, 3, 4, 2)
    shape = embed_permuted.shape
    embed_permuted = embed_permuted.reshape(shape[0], shape[1], -1, shape[-1])
    x = torch.matmul(embed_permuted, guide_permuted).reshape(*shape[:4], -1)
    print(x.shape)
    
    print(torch.sum(torch.abs(aw - x)))
    
def test3():
    x = torch.randn(1, 3, 7, 8)
    w = torch.randn(1, 9, 3)
    res0 = torch.einsum("bchw,bkc->bkhw",x , w)
    print(res0.shape)
    
    shape = x.shape
    x_permuted = x.permute(0, 2, 3, 1).reshape(shape[0], -1, shape[1])
    w_permuted = w.permute(0, 2, 1)
    res = torch.matmul(x_permuted, w_permuted).reshape(shape[0], *shape[-2:], -1).permute(0, 3, 1, 2)
    
    print(res.shape)
    
    print(torch.sum(torch.abs(res0 - res)))

# test0()
# test1()
# test2()
test3()