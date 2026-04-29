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














