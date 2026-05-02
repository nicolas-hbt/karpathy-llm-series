# 1. Why do we seem to run x into a layer that generates a joint QKV tensor, if in the end we choose to split it?

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
...
qkv = self.c_attn(x)
# slice the concatenated tensor back into three separate tensors so we can perform the dot products
q, k, v = qkv.split(self.n_embd, dim=2)
...
```

Gemini: Mathematically, both approaches yield the exact same result, but the single-layer approach runs significantly faster on hardware accelerators like GPUs.

## 1. Saturating the GPU (Bigger is Better)
GPUs are massively parallel processors that thrive on executing large, contiguous matrix multiplications (GEMMs - General Matrix Multiplications).
If you use three separate nn.Linear(config.n_embd, config.n_embd) layers, you are asking the GPU to perform three smaller, independent matrix multiplications. 
By stacking the weights into a single nn.Linear(config.n_embd, 3 * config.n_embd) layer, you are feeding the GPU one massive matrix to multiply. 
GPUs are so wide that they can compute this $1 \times 3$ sized matrix operation in roughly the same amount of time it would take to compute just one of the smaller matrices. You are effectively getting the other two computed for "free" by keeping the GPU cores fully fed and saturated.

## 2. Reducing Kernel Launch Overhead
Every time you call an operation in PyTorch (like passing data through a linear layer), the CPU has to tell the GPU to start working. This command is called a "kernel launch."

* Separate approach: 3 linear layers = 3 kernel launches.
* Combined approach: 1 linear layer = 1 kernel launch.

While a kernel launch only takes microseconds, doing it three times instead of once per attention head, per layer, per training step adds up to a massive bottleneck over a multi-day training run. Slicing a tensor in memory via .split() is practically instantaneous and requires no heavy computation.

## 3. Memory Bandwidth Optimization
To perform a linear projection, the GPU needs to read your input tensor $x$ from its VRAM into its computing registers.If you have three separate layers, the GPU has to read $x$ from memory three times to multiply it against $W_q$, $W_k$, and $W_v$.
If you have a single concatenated layer, the GPU reads $x$ from memory once, multiplies it against the combined weight matrix, and writes the output back.In modern deep learning, fetching data from memory is often a bigger bottleneck than the math itself.



