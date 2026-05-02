# Inference LLMs with vLLM and SGLang  

Top 2 inference Engines:  
- vLLM (Berkeley) 2023 - Introduced PagedAttention to reduce KV cache fragmentation and increase usage of KV cache.  
- SGLang (Berkeley) 2024 - Introduced RadixAttention to increase KV cache sharing and thus increase efficiency of using KV cache.  

There is a growing number of Large Language models and thus increasing workloads. Its very expensive to serve a request from these models.  

Even high end GPUs like A100 can only process few requests per second running only 13B LLAMA model.  

How does a LLM process a query (request/prompt) ?  

Steps:  
- Get the request/prompt.  
- Compute the embedding from the prompt.  
- Generate the next token and the next token always looking at all the tokens (embeddings of these tokens) before it.  

LLM works by generating one token at a time. While generating the current token it has to look at all the previous tokens (embeddings) before it.  

Generating one token at a time does not use the GPUs parallelism. GPUs are max utilized when we multiply fat matrices and not when multiplying a vector to a matix.  

Solution for utilizing GPU parallelism: Batch multiple requests that is process multiple requests at once.  
But this leads to out of memory issues.  
<img width="735" height="345" alt="image" src="https://github.com/user-attachments/assets/338874ec-f377-45d5-82ec-1db846f6b17b" />  
A 13B parameter LLM takes up:  
- 26GB for storing parameters.  
- Storing the embeddings of the token in KV cache takes up 12GB (30%)  
- Storing other things llike activation functions  

In the above picture when predicting the token `future`, the embeddings of all tokens before it are stored in the KV cache.  

=> Even one request can alone take up several GBs of space, thus we are limited to storing only few requests in the KV cache.  
=> The KV cache becomes a bottleneck processing only few requests at the same time.  

Thus there is a need of improving memory management in GPUs.  
<img width="732" height="268" alt="image" src="https://github.com/user-attachments/assets/232fd96c-419a-4684-8025-c9c507021250" />  
Traditionally GPUs allocate memory contiguosly as explained in the above picture.  
GPUs have very basic memory management. We have to pre-allocate contiguous slots for requests.

-> Basic (traditional) memory management in GPUs uses Pre-Allocation + Contiguous Memory allocation, which causes Fragmentation and thus memory wastage.  

<img width="781" height="292" alt="image" src="https://github.com/user-attachments/assets/13c980f3-ab7a-4b67-8aeb-fa17a89aa8d8" />  
Problem-1: Internal Fragmentation because we overallocated 2048 slots for a small Request A.  

<img width="773" height="340" alt="image" src="https://github.com/user-attachments/assets/6a222abe-f005-49a2-8d4b-b95b935eac0a" />  

<img width="1593" height="761" alt="image" src="https://github.com/user-attachments/assets/ecca8689-1d59-405d-adca-16214408e06c" />  
External Fragmentation, Request B needs different contiguous size.  
Inefficient memory usage creates a situation where there is space wasted and since Requests needs contiguous memory the wasted (fragmented) space can't be used.  

<img width="1477" height="814" alt="image" src="https://github.com/user-attachments/assets/32f4e4bd-a596-4db7-aa75-b02c866302f9" />  
Graph shows wastage from different fragmentations and reservation  

## PagedAttention  
Allocates only fixed memory sizes => Thus, makes external fragmentation = 0.  
But, now memory is no longer contiguous, you have to allocate space wherever its available.  

<img width="765" height="313" alt="image" src="https://github.com/user-attachments/assets/9b1173c6-cd25-4a98-8b25-5b7981cb9f0b" />  

<img width="779" height="402" alt="image" src="https://github.com/user-attachments/assets/6d72843f-a4c8-4003-aafe-52c248664423" />  

<img width="1546" height="824" alt="image" src="https://github.com/user-attachments/assets/736e8b8a-10a9-40a4-9a9a-26465449286a" />  
Memory wastage improvement.  

<img width="1501" height="817" alt="image" src="https://github.com/user-attachments/assets/be4c7ac7-70ed-44dd-850b-3485a5a05103" />  


## Life of a Prompt through LLM Inference  
Step 0: Prompts arrives (user gives the prompt).  
Step 1: Prompts are converted to tokens. The LLM works only with numbers (tokens from this point onwards.  
Step 2: All the tokens (formed in Step 1) enter the waiting queue, before they are processed by the vLLM engine. Here there are 2 types of requests for the tokens.  
1. Prefill Request  
2. Decoding Request  
Decoding requests are generally prioritized.  

Step 3:

# Part 1: The Tensor-Level Dry Run (The Mechanics)

Let’s define a microscopic LLM to trace the exact tensor shapes.

**Vocabulary:** `["The", "cat", "sits", "on", "mat"]`  
**Embedding Dimension (`d`):** 4  
**Attention Heads:** 1  
**Prompt:** `"The cat"` → **Target Output:** `"sits on"`

In standard Attention, we multiply the input by three weight matrices (`Wq`, `Wk`, `Wv`) to get our Query (`Q`), Key (`K`), and Value (`V`) tensors.

---

## Step 1: The Prefill Phase (Processing the Prompt)

The user submits `"The cat"`. The model processes this in parallel.

**Input (`X`):** 2 tokens. Shape: `[2, 4]`  
`Sequence Length = 2, Dim = 4`

**Compute Q, K, V:** We multiply `X` by our weight matrices.

- `Q` shape: `[2, 4]`
- `K` shape: `[2, 4]`
- `V` shape: `[2, 4]`

**Attention Calculation:**

```text
Softmax(Q × Kᵀ) × V
```

```text
Q × Kᵀ: [2, 4] @ [4, 2] = [2, 2]
```

The attention scores of `"The"` and `"cat"` looking at each other.

```text
Multiply by V: [2, 2] @ [2, 4] = [2, 4]
```

**Output:** The model predicts the next token is `"sits"`.

**THE KV CACHE ACTION:** The model saves the `K` and `V` tensors to the GPU's VRAM.

- `K_cache` Shape: `[2, 4]`
- `V_cache` Shape: `[2, 4]`

---

## Step 2: Decode Step 1 (Generating `"on"`)

Now we enter the autoregressive loop. We feed the newly generated token (`"sits"`) back into the model.

**Input (`X`):** 1 token (`"sits"`). Shape: `[1, 4]`.

**Compute new Q, K, V:**

- `q_new` shape: `[1, 4]`
- `k_new` shape: `[1, 4]`
- `v_new` shape: `[1, 4]`

**THE KV CACHE ACTION:** We append `k_new` and `v_new` to our existing cache.

- `K_cache` becomes `[3, 4]`  
  Tokens: `"The"`, `"cat"`, `"sits"`
- `V_cache` becomes `[3, 4]`

**Attention Calculation:**

```text
q_new × K_cacheᵀ: [1, 4] @ [4, 3] = [1, 3]
```

The word `"sits"` calculates its attention score against `"The"`, `"cat"`, and itself.

```text
Multiply by V_cache: [1, 3] @ [3, 4] = [1, 4]
```

**Output:** The model predicts `"on"`.

---

## Step 3: Decode Step 2 (Generating `"mat"`)

We feed `"on"` back into the model.

**Input (`X`):** 1 token (`"on"`). Shape: `[1, 4]`.

**Compute new Q, K, V:**

- `q_new`, `k_new`, `v_new` shapes: `[1, 4]`

**THE KV CACHE ACTION:** Append to cache.

- `K_cache` becomes `[4, 4]`
- `V_cache` becomes `[4, 4]`

**Attention Calculation:**

```text
q_new × K_cacheᵀ: [1, 4] @ [4, 4] = [1, 4]
```

```text
Multiply by V_cache: [1, 4] @ [4, 4] = [1, 4]
```

**Output:** The model predicts `"mat"`.

---

## The Core Takeaway

During decoding, `Q` is always a tiny vector `[1, d]`. But `K` and `V` grow by 1 with every single step.

You are doing a Matrix-Vector multiplication, which requires loading the entire growing KV cache from VRAM into the GPU compute cores for every single token.

---

# Part 2: The Systems Engineering Nightmare

Why do we need inference engines? Let's calculate the memory footprint of the KV cache for a real model like Llama-3 (8B).

- **Layers:** 32
- **Heads:** 32
- **Head Dimension:** 128
- **Data Type:** FP16, 2 bytes per parameter

**Formula for one token's KV cache:**

```text
2 (K and V) × 32 (layers) × 32 (heads) × 128 (dim) × 2 (bytes)
= 524,288 bytes
≈ 0.5 MB per token
```

If you have a batch of 100 users, and each user has a conversation history of 4,000 tokens:

```text
0.5 MB × 4,000 tokens × 100 users = 200 Gigabytes of VRAM
```

The model weights only take up 16GB. The KV Cache takes up 200GB.

The KV Cache is the ultimate bottleneck to scaling LLMs.

---

# Part 3: How vLLM Optimizes KV Cache (PagedAttention)

Before vLLM, inference engines used Static Allocation. If a user's max sequence length was 4,000, the engine pre-allocated a contiguous chunk of 4,000 slots in VRAM.

If the user only generated 100 tokens, 3,900 slots were wasted. This is called Internal Fragmentation, and it wasted ~80% of GPU memory.

## vLLM’s Solution: PagedAttention

Inspired by Operating System Virtual Memory, vLLM breaks the KV cache into small, fixed-size "blocks" (e.g., 16 tokens per block).

**Logical vs. Physical:** The model thinks it has a contiguous KV cache (Logical KV). But in reality, the cache is scattered across the GPU memory in non-contiguous blocks (Physical KV).

**The Block Table:** vLLM maintains a map that translates the Logical tokens to Physical blocks.

**Dynamic Allocation:** When a user starts generating text, vLLM allocates exactly one block (16 tokens). When token 17 is generated, vLLM allocates a second block wherever there is free space in the VRAM, and updates the Block Table.

**The Result:** Memory waste drops from 80% to <4%. Because memory is freed up, vLLM can cram 5x to 10x more users into the same batch, drastically increasing throughput.

---

# Part 4: How SGLang Optimizes KV Cache (RadixAttention)

vLLM solves fragmentation, but it doesn't solve redundancy.

Imagine 100 users all interacting with a RAG application. Every single user's prompt starts with the exact same 2,000-word system prompt and document context.

In vLLM, the engine computes and stores the KV cache for those 2,000 words 100 separate times, wasting massive amounts of compute and memory.

## SGLang’s Solution: RadixAttention (Prefix Caching)

SGLang treats the KV cache like a Radix Tree (Prefix Tree). It is designed to aggressively share KV cache across different requests.

**The Radix Tree:** When User A sends a prompt, SGLang computes the KV cache and stores it in the GPU, mapping it to a Radix Tree node based on the exact sequence of tokens.

**Cache Hit:** When User B sends a prompt, SGLang checks the Radix Tree. It realizes the first 2,000 tokens match User A exactly.

**Zero-Overhead Sharing:** Instead of recomputing, SGLang simply points User B's Block Table to User A's physical KV cache blocks in the GPU. User B's prefill phase for those 2,000 tokens takes 0 milliseconds.

**Forking:** When User B's prompt diverges from User A's (e.g., they ask a different question at the end), SGLang allocates new, separate blocks just for the divergent tokens.

**LRU Eviction:** When requests finish, SGLang doesn't delete the KV cache immediately. It keeps it in VRAM using a Least-Recently-Used (LRU) policy. If another user asks a similar question 5 minutes later, the cache is still there.

**The Result:** SGLang is the undisputed king of Agentic workflows, multi-turn chat, and RAG, because it turns massive, repetitive prompts into instantaneous cache hits.

---

# Summary Mental Model

**The Math:** KV Cache transforms `O(N²)` Attention into `O(N)` by saving past Keys and Values, turning Decode into a memory-bandwidth-bound Matrix-Vector operation.

**vLLM (PagedAttention):** Think of it as Memory Defragmentation. It chunks the KV cache so you don't waste empty space, allowing larger batch sizes.

**SGLang (RadixAttention):** Think of it as Deduplication. It builds a tree of token sequences so multiple users can share the exact same physical KV cache for shared prompts.



















