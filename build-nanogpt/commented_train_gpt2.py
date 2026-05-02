# interact with the operating system, mainly for file paths, ddp env variables, and directory creation
import os
# used strictly for calculating the cosine learning rate decay schedule
import math
# required to measure the time taken for each training step (dt) to compute tokens per second throughput
import time
# used to dynamically check if the 'fused' parameter exists in our PyTorch version's AdamW signature
import inspect
# provides a clean way to define the configuration object with type hints and default values
from dataclasses import dataclass
import torch
import torch.nn as nn
from hellaswag import render_example, iterate_examples

# -----------------------------------------------------------------------------

# implements the multi-head masked self-attention mechanism
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        # properly initializes the parent nn.Module class to register parameters
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # better than doing three separate linear layers, we can do one for all of them and then split the result
        # advantage is that we only have to do one matrix multiply instead of three to saturate the gpu compute
        # "c" below stands for "combined"/"concatenated" projection for query, key and value
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # without the line above, we would have 3 linear layers of shape (n_embd, n_embd)
        # but we concatenate them along the output dimension to dispatch a single large kernel
        # output projection. Necessary to recombine the multiple heads back into the residual stream, 
        # but we could also let the heads remain split and just flatten all of the head outputs into a single vector of size n_embd, which is what this projection layer does.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # a special flag to scale down the initialization of residual layers to prevent early training instability
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    # the forward pass computing the attention mechanism
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        # slice the concatenated tensor back into three separate tensors so we can perform the dot products
        q, k, v = qkv.split(self.n_embd, dim=2)
        # reshape and transpose to isolate the heads in the batch dimension, allowing independent parallel attention computation per head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # reshape the query identically so its dimensions align with the keys for the matmul
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # reshape the values identically so they can be multiplied by the attention weights later
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # delegate the attention computation to highly optimized cuda kernels (flash attention) to skip materializing the massive T x T matrix
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        # transpose the heads back and flatten them so they can be linearly projected back to the residual stream dimension
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection to mix the independent head representations back together
        y = self.c_proj(y)
        # return the final attention features
        return y

# implements the feedforward network that processes the tokens individually after they've communicated via attention
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # expands the dimensionality by 4x to allow the network space to memorize facts and learn rich representations
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        # uses the approximated GELU activation to exactly match the historical implementation of OpenAI's GPT-2 model
        self.gelu    = nn.GELU(approximate='tanh')
        # projects the 4x expanded dimension back down to the residual stream dimension
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        # flag for scaling the initialization variance, helping to keep deep residual networks stable at initialization
        self.c_proj.NANOGPT_SCALE_INIT = 1

    # process the features through the layers
    def forward(self, x):
        # project into the higher dimensional space
        x = self.c_fc(x)
        # apply the non-linearity
        x = self.gelu(x)
        # project back down to combine the learned features into the residual stream
        x = self.c_proj(x)
        # return the transformed features
        return x

# a single transformer block consisting of attention and mlp, representing one "layer" of depth
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # applies layer normalization before the attention mechanism (pre-norm formulation) to stabilize gradients
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # instantiate the self-attention block configured for this specific model size
        self.attn = CausalSelfAttention(config)
        # applies a second layer normalization before the MLP to further stabilize the forward pass
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # instantiate the feedforward network
        self.mlp = MLP(config)

    # the forward pass defining how the data flows through this block
    def forward(self, x):
        # add the attention output to the residual stream (x) so information is compounded, not replaced
        x = x + self.attn(self.ln_1(x))
        # add the feedforward output to the residual stream for the same reason
        x = x + self.mlp(self.ln_2(x))
        # return the enriched residual stream
        return x

# a clean dataclass to hold the architectural hyperparameters of the model
@dataclass
class GPTConfig:
    # defines the maximum context window the model can process at once
    block_size: int = 1024 # max sequence length
    # defines the number of discrete tokens the model understands, plus special tokens
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    # defines the depth of the network
    n_layer: int = 12 # number of layers
    # defines the number of independent attention mechanisms per layer
    n_head: int = 12 # number of heads
    # defines the size of the residual stream (the main highway of information)
    n_embd: int = 768 # embedding dimension

# the top-level GPT model wrapping the token embeddings, positional embeddings, and transformer blocks
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # groups the transformer components into a dictionary so PyTorch natively tracks their parameters
        self.transformer = nn.ModuleDict(dict(
            # the embedding layer that translates discrete token IDs into dense continuous vectors
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # the embedding layer that gives the model a sense of absolute position in the sequence
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # a list of the stacked transformer blocks (the "meat" of the network)
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # the final layer normalization applied before the output projection to stabilize logits
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # the final linear layer that projects the residual stream back to the vocabulary space to predict the next token
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        # ties the input embedding and output projection weights, dramatically saving parameters and ensuring syntactic alignment
        # explanation in more details in my Google Docs
        self.transformer.wte.weight = self.lm_head.weight

        # iterates over all modules and applies our custom weight initialization logic (defined below)
        self.apply(self._init_weights)

    # custom weight initialization function called recursively on all sub-modules
    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            # standard deviation for the normal distribution, matching typical transformer heuristics
            std = 0.02
            # check if this is a residual projection layer that we specifically flagged earlier
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # scale down the variance based on the depth of the network so early gradients don't explode
                # as we stack more and more layers, the variance of the activations grows, so we need to scale down the initialization even more aggressively to prevent the early layers from blowing up
                # this means that the value for std gets smaller as we increase the number of layers
                std *= (2 * self.config.n_layer) ** -0.5 # 20**-0.5 same thing as 1/sqrt(20) so we see that as self.config.n_layer increases, the std decreases
            # apply the calculated normal distribution to the weights
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # strictly initialize biases to zero to prevent them from skewing the early network dynamics
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # initialize embedding weights with the standard normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # the forward pass that turns token indices into logits and computes the loss
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        # extract the batch size and sequence length to generate the positional sequence
        B, T = idx.size()
        # enforce the context length constraint so we don't index out of bounds on the positional embeddings
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        # create a sequence of integers from 0 to T-1 to represent the positions of the tokens
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        # look up the learned positional vectors for these positions
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        # look up the learned token vectors for the input indices
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        # broadcast and add the positional embeddings to the token embeddings to form the initial residual stream
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        # pass the data sequentially through every layer of the network
        for block in self.transformer.h:
            # update the residual stream with the output of the block
            x = block(x)
        # forward the final layernorm and the classifier
        # normalize the final state so the language modeling head behaves predictably
        x = self.transformer.ln_f(x)
        # project the final normalized state into the vocabulary dimension to get raw, unnormalized probability scores
        logits = self.lm_head(x) # (B, T, vocab_size)
        # default the loss to None in case we are just running inference and don't have targets
        loss = None
        # check if we are in training/evaluation mode and actually passed targets
        if targets is not None:
            # calculate the cross entropy loss, flattening the batch and time dimensions as PyTorch expects
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        # return both the raw predictions and the computed loss
        return logits, loss

    # a utility classmethod to load OpenAI's official GPT-2 weights into our implementation
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        # sanity check to ensure the user requested a valid GPT-2 size
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # import the huggingface library dynamically so it's not a hard dependency for people who only want to train
        from transformers import GPT2LMHeadModel
        # notify the user that the weights are being fetched
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        # mapping the model string to the exact architectural hyperparameters used by OpenAI
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # hardcode the vocab size to match the standard GPT-2 tokenizer
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        # hardcode the block size since all original GPT-2 models were trained on 1024 tokens
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        # instantiate the dataclass with the resolved hyperparameters
        config = GPTConfig(**config_args)
        # build the blank model architecture
        model = GPT(config)
        # extract the underlying state dictionary (the raw parameter tensors) of our empty model
        sd = model.state_dict()
        # get a list of all parameter names in our model
        sd_keys = sd.keys()
        # strip out the causal mask buffers, because they are structurally determined and not learned parameters
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        # download and initialize the official weights from the hub
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        # extract the huggingface model's state dictionary
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        # get a list of all parameter names in the huggingface model
        sd_keys_hf = sd_hf.keys()
        # strip out the huggingface causal mask buffer equivalents
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        # strip out the huggingface self-attention bias mask
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        # define the specific layers where OpenAI used 1D Convolutions instead of standard Linear layers
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them to fix the math
        # ensure we have mapped 1-to-1 the number of expected parameters
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        # iterate over every parameter in the huggingface model
        for k in sd_keys_hf:
            # check if the parameter belongs to one of the Conv1D layers that needs transposing
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                # verify that the shape matches perfectly once transposed
                assert sd_hf[k].shape[::-1] == sd[k].shape
                # temporarily disable gradient tracking since we are just moving memory around
                with torch.no_grad():
                    # copy the transposed huggingface tensor into our model's parameter
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                # verify the shapes match perfectly
                assert sd_hf[k].shape == sd[k].shape
                # temporarily disable gradient tracking
                with torch.no_grad():
                    # explicitly copy the raw tensor data over
                    sd[k].copy_(sd_hf[k])

        # return the fully populated model
        return model

    # configures the optimizer, explicitly separating parameters that should be weight-decayed from those that shouldn't
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        # grab all named parameters from the model module
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out any parameters that are frozen or don't require gradients
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't to prevent underfitting
        # filter for multi-dimensional tensors (weights/embeddings) to apply the L2 penalty
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # filter for 1D tensors (biases/layernorms) to exempt them from the L2 penalty
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # organize the parameters into two discrete dictionaries for the optimizer constructor
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # calculate the total number of parameters that are being decayed for logging purposes
        num_decay_params = sum(p.numel() for p in decay_params)
        # calculate the total number of parameters that are exempted from decay for logging purposes
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # only print the optimizer stats on the master process to avoid spamming the console in distributed setups
        if master_process:
            # log the decay group stats
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            # log the no-decay group stats
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # use python reflection to see if 'fused' is an accepted argument in this PyTorch version
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # only use the fused implementation if we are actually running on a CUDA device
        use_fused = fused_available and device_type == "cuda"
        # log whether the highly optimized fused kernel is active
        if master_process:
            # print the boolean flag
            print(f"using fused AdamW: {use_fused}")
        # instantiate the optimizer with the standard GPT-2 beta hyperparameters and the organized parameter groups
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        # return the configured optimizer ready for the training loop
        return optimizer

# -----------------------------------------------------------------------------
# import the official OpenAI tokenizer library for extremely fast BPE tokenization
import tiktoken
# import numpy for efficient multidimensional array loading from disk
import numpy as np

# utility function to load raw numpy token arrays and cast them to pytorch tensors
def load_tokens(filename):
    # read the raw binary data from disk into a numpy array
    npt = np.load(filename)
    # forcefully cast the numpy array to int32 because pytorch doesn't fully support uint16
    npt = npt.astype(np.int32) # added after video
    # convert the numpy array directly into a native pytorch tensor formatted as long integers
    ptt = torch.tensor(npt, dtype=torch.long)
    # return the final tensor ready for the model
    return ptt

# a lightweight data loader designed to handle chunked/sharded datasets and distribute them across GPUs
class DataLoaderLite:
    # initialize the loader with batch size, context length, and distributed cluster topology
    def __init__(self, B, T, process_rank, num_processes, split):
        # store micro batch size locally
        self.B = B
        # store sequence length locally
        self.T = T
        # store the current GPU's ID (rank) so it knows which chunks of data to process
        self.process_rank = process_rank
        # store the total number of GPUs to calculate the global stride
        self.num_processes = num_processes
        # ensure we only initialize with recognized dataset splits
        assert split in {'train', 'val'}

        # get the shard filenames
        # point to the local directory containing the tokenized dataset
        data_root = "edu_fineweb10B"
        # list all the physical files inside the dataset directory
        shards = os.listdir(data_root)
        # filter the files to only include those belonging to the requested split (e.g., 'train')
        shards = [s for s in shards if split in s]
        # sort the shards alphabetically so every GPU processes the dataset in the exact same order
        shards = sorted(shards)
        # prepend the full directory path to the filenames so we can load them
        shards = [os.path.join(data_root, s) for s in shards]
        # store the finalized list of file paths
        self.shards = shards
        # panic if no files were found, meaning the dataset is missing or misnamed
        assert len(shards) > 0, f"no shards found for split {split}"
        # strictly log the successful shard discovery from the master process
        if master_process:
            # print the number of discovered shards
            print(f"found {len(shards)} shards for split {split}")
        # call reset to initialize the pointer to the very first token
        self.reset()

    # sets the dataset pointers back to the beginning of the first shard
    def reset(self):
        # state, init at shard zero
        # set the active file index to the first file in the list
        self.current_shard = 0
        # load the entire token array of the first shard into memory
        self.tokens = load_tokens(self.shards[self.current_shard])
        # calculate the initial starting index for this specific GPU based on its rank
        self.current_position = self.B * self.T * self.process_rank

    # fetches the next block of inputs and targets, then advances the pointers
    def next_batch(self):
        # localized variables for cleaner slicing syntax
        B, T = self.B, self.T
        # slice out the contiguous chunk of tokens needed for both inputs and targets (+1 token for the target offset)
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        # the inputs are everything from the slice except the very last token
        x = (buf[:-1]).view(B, T) # inputs
        # the targets are everything from the slice except the very first token, effectively shifted by one
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        # jump the pointer forward by the total amount of tokens consumed by all GPUs combined (global batch size)
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        # check if we are dangerously close to the end of the current tensor
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            # increment the shard index, wrapping back to 0 if we hit the end of the dataset (looping)
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            # wipe the old tokens and load the new shard into memory
            self.tokens = load_tokens(self.shards[self.current_shard])
            # reset the localized pointer index for the new shard based on the GPU rank
            self.current_position = B * T * self.process_rank
        # return the input and target tensors
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

# processes the model's logits to figure out which multiple-choice option it thinks is most probable
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    # remove the logits for the very last token because we don't have a ground truth target for it
    shift_logits = (logits[..., :-1, :]).contiguous()
    # shift the token sequences over by one to align with the predictions
    shift_tokens = (tokens[..., 1:]).contiguous()
    # flatten the logits tensor so cross_entropy can process it easily
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    # flatten the target tokens tensor as well
    flat_shift_tokens = shift_tokens.view(-1)
    # calculate the individual loss for every single token prediction, keeping the tensor unreduced
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    # reshape the 1D loss array back into the original batch x time matrix format
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    # shift the mask tensor similarly to align with the shifted losses
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    # multiply the losses by the mask to strictly zero out the losses coming from the prompt context
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    # aggregate all valid losses across the sequence length
    sum_loss = masked_shift_losses.sum(dim=1)
    # divide the summed loss by the length of the actual completion to get a normalized average
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    # grab the row index corresponding to the minimum average loss
    pred_norm = avg_loss.argmin().item()
    # return the index of the best answer
    return pred_norm

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
# import the necessary distributed backend initialization functions
from torch.distributed import init_process_group, destroy_process_group
# import the PyTorch wrapper that automatically handles synchronizing gradients across multiple GPUs
from torch.nn.parallel import DistributedDataParallel as DDP
# import the main distributed module to handle collective operations like all_reduce
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
# detect if the user launched the script via torchrun by checking if the RANK variable exists
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
# branch the setup logic based on whether we are running distributed or solitary
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    # DDP is primarily built for and accelerated by CUDA, so we forcefully verify it's available
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    # bootstrap the communication backend for multi-gpu networking (NVIDIA's NCCL is standard)
    init_process_group(backend='nccl')
    # get the global GPU ID across the entire cluster
    ddp_rank = int(os.environ['RANK'])
    # get the localized GPU ID specifically on this physical machine
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    # get the total number of GPUs running the job across all nodes
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    # construct the device string dynamically based on the local rank
    device = f'cuda:{ddp_local_rank}'
    # force PyTorch to strictly bind this specific process to its designated GPU
    torch.cuda.set_device(device)
    # designate rank 0 as the master process to handle single-writer tasks like saving logs and checkpoints
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    # mock the distributed variables so the rest of the code works transparently without DDP
    ddp_rank = 0
    # set local rank to 0
    ddp_local_rank = 0
    # set the total world size to 1 since it's just this machine
    ddp_world_size = 1
    # automatically designate this solitary process as the master process
    master_process = True
    # attempt to autodetect device
    # default to the CPU as the lowest common denominator
    device = "cpu"
    # upgrade the device to CUDA if NVIDIA hardware is detected
    if torch.cuda.is_available():
        # set device
        device = "cuda"
    # alternatively, upgrade to Apple Silicon's MPS backend if running on a newer Mac
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # set device
        device = "mps"
    # log the hardware target
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
# strip down the specific device string (e.g., 'cuda:0') to its generic family (e.g., 'cuda') for autocast APIs
device_type = "cuda" if device.startswith("cuda") else "cpu"

# seed the random number generator deterministically so runs are reproducible
torch.manual_seed(1337)
# explicitly seed the CUDA random number generator as well to ensure total determinism
if torch.cuda.is_available():
    # apply the seed
    torch.cuda.manual_seed(1337)

# load the official GPT-2 Byte Pair Encoding tokenizer using tiktoken
enc = tiktoken.get_encoding("gpt2")

# define the desired absolute number of tokens processed per optimization step
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
# define the small, memory-constrained batch size that a single GPU actually processes in a single forward pass
B = 64 # micro batch size
# define the sequence context length we are training the model on
T = 1024 # sequence length
# mathematically enforce that the math works out perfectly without fractional batches
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
# calculate exactly how many micro-steps every GPU must take before we have accumulated enough gradients to match total_batch_size
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
# log the calculated accumulation strategy on the main process
if master_process:
    # print the target global batch size
    print(f"total desired batch size: {total_batch_size}")
    # print the number of micro steps needed
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# initialize the streaming dataloader specifically pointing to the 'train' split files
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
# initialize a secondary dataloader pointing to the 'val' split for evaluation runs
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# unlock the powerful TensorFloat32 (TF32) hardware cores on newer Ampere+ GPUs for massive matrix multiplication speedups
torch.set_float32_matmul_precision('high')

# create model
# initialize our custom model, expanding the vocabulary slightly to 50304 to make it divisible by 64 (better memory layout and compute efficiency)
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
# transfer all of the model's parameters to the active hardware accelerator
model.to(device)
# torch.compile currently breaks some advanced eval logic, so we keep it off by default
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
# if manually enabled, use PyTorch 2.0's compiler to fuse operations and speed up execution
if use_compile:
    # wrap the model in the compiled graph
    model = torch.compile(model)
# if we are running in a cluster, wrap the model in the DDP container to handle distributed gradients
if ddp:
    # initialize DDP locally
    model = DDP(model, device_ids=[ddp_local_rank])
# keep a reference to the un-wrapped inner model because DDP obscures direct access to attributes and methods
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# set the peak learning rate according to the GPT-3 paper scaling laws for a model of this size
max_lr = 6e-4
# define the floor learning rate that the schedule will eventually decay down to
min_lr = max_lr * 0.1
# set the number of steps to linearly ramp up the learning rate to prevent early optimization explosions
warmup_steps = 715
# define the total budget of steps for an entire epoch (10B tokens divided by 0.5M tokens per step)
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
# dynamic function to calculate the current learning rate based on the step number
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    # check if we are still in the initial warmup phase
    if it < warmup_steps:
        # scale the learning rate up linearly
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    # check if we have completely finished the schedule
    if it > max_steps:
        # cap it at the minimum forever
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    # calculate what percentage of the decay phase we have completed
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    # strictly verify the ratio is valid just in case
    assert 0 <= decay_ratio <= 1
    # map the linear progress percentage to a cosine curve for a smoother decay
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    # apply the cosine coefficient to the learning rate range
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
# instantiate the configured AdamW optimizer utilizing the inner un-wrapped model
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# create the log directory we will write checkpoints to and log to
# set a standard folder name
log_dir = "log"
# safely create the directory if it doesn't already exist
os.makedirs(log_dir, exist_ok=True)
# define the complete path to the main text log file
log_file = os.path.join(log_dir, f"log.txt")
# open the file with write ('w') permissions to instantly wipe out any old runs
with open(log_file, "w") as f: # open for writing to clear the file
    # execute an empty operation purely to close the file back up
    pass

# begin the primary training loop
for step in range(max_steps):
    # capture the exact start time of this step to measure iterations per second later
    t0 = time.time()
    # flag if this is the absolute final iteration so we can forcefully run validations
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    # trigger validation periodically or on the absolute final step
    if step % 250 == 0 or last_step:
        # switch the model to evaluation mode, which disabled dropout and freezes batchnorm stats
        model.eval()
        # rewind the validation dataloader back to the start to ensure consistent evaluation sets
        val_loader.reset()
        # locally disable the massive gradient tape tracking to save immense amounts of memory
        with torch.no_grad():
            # initialize a counter for the accumulated loss
            val_loss_accum = 0.0
            # set how many batches to sample for the validation calculation
            val_loss_steps = 20
            # loop through the validation batches
            for _ in range(val_loss_steps):
                # grab the next chunk of test data
                x, y = val_loader.next_batch()
                # push the validation inputs and targets to the accelerator
                x, y = x.to(device), y.to(device)
                # automatically downcast precision-safe operations to bfloat16 for immense speedups
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    # compute the forward pass
                    logits, loss = model(x, y)
                # normalize the loss by the number of steps to get an accurate mean
                loss = loss / val_loss_steps
                # accumulate the scalar loss value, disconnecting it from the computation graph with detach()
                val_loss_accum += loss.detach()
        # synchronize the independent loss calculations across all the nodes in the cluster
        if ddp:
            # mathematically average the loss tensor across all ranks
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        # only let the master node write to the disk and console to prevent overlapping outputs
        if master_process:
            # print out the computed validation loss
            print(f"validation loss: {val_loss_accum.item():.4f}")
            # open the central log file in append mode
            with open(log_file, "a") as f:
                # write the step number and validation score
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            # occasionally export the entire model state to disk so we can resume or deploy it
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                # construct the filename dynamically to include the step number padding
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                # create a comprehensive dictionary of all necessary state
                checkpoint = {
                    # grab the raw parameter tensors
                    'model': raw_model.state_dict(),
                    # grab the architecture definition
                    'config': raw_model.config,
                    # record the exact step we are on
                    'step': step,
                    # record the validation loss for comparison
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                # serialize the dictionary to disk using pytorch's binary format
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    # trigger the common-sense reasoning benchmark periodically, avoiding it if using the compiler due to bugs
    if (step % 250 == 0 or last_step) and (not use_compile):
        # track how many questions the model answers correctly
        num_correct_norm = 0
        # track the total number of questions answered
        num_total = 0
        # iterate through the entire hellaswag validation dataset
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            # efficiently shard the workload by distributing the questions dynamically across all GPUs
            if i % ddp_world_size != ddp_rank:
                # skip this question if it belongs to another GPU
                continue
            # render the example into tokens and labels
            # parse the complex dictionary into raw context, completions, masks, and the ground truth index
            _, tokens, mask, label = render_example(example)
            # push the generated tokens to the accelerator
            tokens = tokens.to(device)
            # push the mask to the accelerator
            mask = mask.to(device)
            # get the logits
            # disable the gradient tape
            with torch.no_grad():
                # enable mixed precision inference
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    # compute the forward pass, extracting the raw predictions
                    logits, loss = model(tokens)
                # use our helper function to isolate the loss over the completions and find the lowest one
                pred_norm = get_most_likely_row(tokens, mask, logits)
            # increment the local GPU's total counter
            num_total += 1
            # increment the local GPU's correct counter if the lowest-loss prediction matched the ground truth label
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        # aggregate the local scores into a global score across the cluster
        if ddp:
            # package the integer variable into a tensor so NCCL can transmit it
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            # package the correct answer count as well
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            # mathematically sum all of the total question counts across the network
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            # mathematically sum all of the correct answer counts across the network
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            # extract the aggregated raw integer
            num_total = num_total.item()
            # extract the aggregated correct count
            num_correct_norm = num_correct_norm.item()
        # calculate the global accuracy percentage
        acc_norm = num_correct_norm / num_total
        # strictly log the results from the master node
        if master_process:
            # print the fraction and the formatted percentage
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            # append the score to the central log file
            with open(log_file, "a") as f:
                # write the formatted string
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    # trigger qualitative evaluation occasionally to visually see what the model is learning
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        # set to eval mode to disable any training-specific behavior
        model.eval()
        # ask the model to generate four distinct continuations simultaneously in a batch
        num_return_sequences = 4
        # limit the generation to prevent it from rambling endlessly
        max_length = 32
        # process the string prefix into integer tokens
        tokens = enc.encode("Hello, I'm a language model,")
        # format the python list into a native tensor
        tokens = torch.tensor(tokens, dtype=torch.long)
        # duplicate the 1D sequence tensor into a 2D batch tensor of size (4, length)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        # push the prompt tensor to the accelerator
        xgen = tokens.to(device)
        # isolate the random generator state specifically for this sampling run
        sample_rng = torch.Generator(device=device)
        # seed the generator uniquely per GPU so they don't generate the exact same text
        sample_rng.manual_seed(42 + ddp_rank)
        # enter an autoregressive loop to generate tokens one by one
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            # strictly disable gradient tracking
            with torch.no_grad():
                # enable mixed precision inference
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    # run the massive forward pass on the continuously growing sequence
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                # we only care about the very last probability distribution to predict the next single token
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                # transform the raw, unbounded logits into mathematically valid probabilities (summing to 1)
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # aggressively truncate the distribution to only include the top 50 most likely candidates, destroying the long tail of noise
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                # roll the dice to randomly select a token index based on the remaining weights
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                # map the local top-50 index back to the global vocabulary ID
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                # concatenate the newly predicted token onto the running context
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        # loop through the batch dimension to print each sample separately
        for i in range(num_return_sequences):
            # extract the raw list of integers
            tokens = xgen[i, :max_length].tolist()
            # use the tokenizer to map the integers back to human-readable text
            decoded = enc.decode(tokens)
            # print the result, prefixed by the GPU ID and sample ID
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    # ensure the model is explicitly placed back into training mode to re-enable dropout/batchnorm
    model.train()
    # completely wipe the gradient buffers from the previous global step so they don't erroneously compound
    optimizer.zero_grad()
    # locally track the accumulated loss for logging purposes
    loss_accum = 0.0
    # iterate through all the micro-steps required to reach the global target batch size
    for micro_step in range(grad_accum_steps):
        # grab the next chunk of training data
        x, y = train_loader.next_batch()
        # transfer the inputs and targets to the GPU
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        # if using distributed training, explicitly tell DDP whether or not to synchronize gradients over the network
        if ddp:
            # only trigger the massive network sync operation on the very last micro-step to save insane amounts of bandwidth
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # utilize tensor cores to execute the forward pass rapidly in bfloat16 mixed precision
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            # calculate the logits and the loss for this micro-step
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        # locally accumulate the detached scalar loss for accurate console reporting later
        loss_accum += loss.detach()
        # compute the backward pass, systematically populating the .grad attribute of every learnable tensor
        loss.backward()
    # average the accumulated loss value across all the independent GPUs for reporting accuracy
    if ddp:
        # trigger the synchronization
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # strictly clamp the maximum magnitude of the gradients to prevent explosive instability from outlier batches
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    # call our cosine schedule function
    lr = get_lr(step)
    # dynamically inject the newly computed learning rate into the optimizer's parameter groups
    for param_group in optimizer.param_groups:
        # update the lr reference
        param_group['lr'] = lr
    # physically execute the parameter update formula (AdamW) using the accumulated gradients
    optimizer.step()
    # explicitly force the CPU to halt and wait for the GPU queue to completely drain before stopping the timer
    if device_type == "cuda":
        # block the thread until computation is done
        torch.cuda.synchronize() # wait for the GPU to finish work
    # capture the end time
    t1 = time.time()
    # calculate the time elapsed
    dt = t1 - t0 # time difference in seconds
    # mathematically deduce exactly how many individual tokens the entire cluster just crunched
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    # divide total tokens by total time to get the global throughput speed
    tokens_per_sec = tokens_processed / dt
    # print the step summary from the main node
    if master_process:
        # print formatting a very clean summary of loss, LR, gradient norm, and speeds
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        # open the central log file to persist the loss
        with open(log_file, "a") as f:
            # write the unformatted raw values
            f.write(f"{step} train {loss_accum.item():.6f}\n")

# conditionally teardown the distributed environment upon a clean exit
if ddp:
    # release the network ports and free the background processes
    destroy_process_group()