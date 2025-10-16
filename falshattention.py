
import time
import torch

torch.manual_seed(0)
NEG_INF = -1e10  # -infinity
EPSILON = 1e-10

Q_LEN = 6
K_LEN = 6
Q_BLOCK_SIZE = 3
KV_BLOCK_SIZE = 3
P_DROP = 0.2

Tr = Q_LEN // Q_BLOCK_SIZE # Tr 块数
Tc = K_LEN // KV_BLOCK_SIZE # Tc 块数

Q = torch.randn(1, 1, Q_LEN, 4, requires_grad=True).to(device='cpu')
K = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')
V = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')

# step 4
Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)

print("----------------FlashAttentionV1------------------------")
O = torch.zeros_like(Q, requires_grad=True)
l = torch.zeros(Q.shape[:-1])[..., None]
m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF
# print(O.shape, l.shape, m.shape)

# step 5
O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))
# print(O_BLOCKS[0].shape, l_BLOCKS[0].shape, m_BLOCKS[0].shape)

# step 6
start_time1 = time.time()
for j in range(Tc):
    # step 7
    Kj = K_BLOCKS[j]
    Vj = V_BLOCKS[j]
    # step 8
    for i in range(Tr):
        # step 9
        Qi = Q_BLOCKS[i]
        Oi = O_BLOCKS[i]
        li = l_BLOCKS[i]
        mi = m_BLOCKS[i]

        # step 10
        # S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi, Kj) # Qi*Kj.T 
        S_ij = torch.einsum("... i d, ... j d -> ... i j", Qi, Kj)
        
        # step 11
        # mask = S_ij.ge(0.5)
        # S_ij = torch.masked_fill(S_ij, mask, value=0)
        
        # step 12
        m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
        P_ij = torch.exp(S_ij - m_block_ij)
        l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON

        P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

        # step 13
        mi_new = torch.maximum(m_block_ij, mi)

        li_new = torch.exp(mi - mi_new) * li + \
                 torch.exp(m_block_ij - mi_new) * l_block_ij

        # step 14
        # m = torch.nn.Dropout(p=P_DROP)
        # P_ij_Vj = m(P_ij_Vj)

        # Step 15
        O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi \
                      + (torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
        # print(f'-----------Attention : Q{i}xK{j}---------')
        # print(O_BLOCKS[i].shape)
        # print(O_BLOCKS[0])
        # print(O_BLOCKS[1])
        # print('\n')

        # step 16
        l_BLOCKS[i] = li_new
        m_BLOCKS[i] = mi_new

O = torch.cat(O_BLOCKS, dim=2)
l = torch.cat(l_BLOCKS, dim=2)
m = torch.cat(m_BLOCKS, dim=2)
print(O.shape, time.time()-start_time1)
print(O)

print("----------------FlashAttentionV2------------------------")
O2 = torch.zeros_like(Q, requires_grad=True)
O2_BLOCKS = list(torch.split(O2, Q_BLOCK_SIZE, dim=2))

start_time2 = time.time()
for i in range(Tr):
    Qi = Q_BLOCKS[i]
    Oi = O2_BLOCKS[i]
    li = torch.zeros((*Q.shape[:-2], Q_BLOCK_SIZE, 1))
    mi = torch.ones((*Q.shape[:-2], Q_BLOCK_SIZE, 1)) * NEG_INF
    for j in range(Tc):
        Kj = K_BLOCKS[j]
        Vj = V_BLOCKS[j]

        S_ij = torch.einsum("... i d, ... j d -> ... i j", Qi, Kj)

        mi_new = torch.maximum(torch.max(S_ij, dim=-1, keepdims=True)[0], mi)
        P_ij = torch.exp(S_ij - mi_new)
        
        li = torch.exp(mi-mi_new)*li+torch.sum(P_ij, dim=-1, keepdims=True)+EPSILON
        P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

        Oi = torch.exp(mi-mi_new) * Oi + P_ij_Vj
        mi = mi_new
    O2_BLOCKS[i] = Oi/li

O2 = torch.cat(O2_BLOCKS, dim=2)
print(O2.shape, time.time()-start_time2)
print(O2)

print("----------------Standard Self-Attention------------------------")
start_time3 = time.time()
scores = torch.matmul(Q, K.transpose(-2, -1))
attention_weights = torch.softmax(scores, dim=-1)
output = torch.matmul(attention_weights, V)
print(output.shape, time.time()-start_time3)
print(output)