import numpy as np
inputs = np.array([1, 1, 3, 3, 3], dtype=np.float16)

#softmax
eInputs = np.exp(inputs)
result = eInputs/np.sum(eInputs)
print(result)

# save-softmax
max_val = max(inputs)
emInputs = np.exp(inputs-max_val)
result1 = emInputs/np.sum(emInputs)
print(result1)