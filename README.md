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



