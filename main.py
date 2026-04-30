import os
import re
import urllib.request

import tiktoken
import torch

from tokenizers import SimpleTokenizerV2
from dataset import create_dataloader_v1


URL = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)
FILE_PATH = "the-verdict.txt"

if not os.path.exists(FILE_PATH):
    urllib.request.urlretrieve(URL, FILE_PATH)


def main():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of characters:", len(raw_text))
    print(raw_text[:99])

    # --- SimpleTokenizerV2 with custom vocab ---
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(vocab_size)
    vocab = {token: integer for integer, token in enumerate(all_words)}

    tokenizer = SimpleTokenizerV2(vocab)
    text = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))

    # Extend vocab with special tokens
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    print(len(vocab))

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
    tokenizer = SimpleTokenizerV2(vocab)
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))

    # --- tiktoken GPT-2 tokenizer ---
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Akwirw ier."
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)
    print(tokenizer.decode(integers))

    # --- Context window / next-token prediction ---
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))
    enc_sample = enc_text[50:]
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size + 1]
    print(f"x: {x}")
    print(f"y: {y}")

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

    # --- DataLoader experiments ---
    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=2, shuffle=False)
    data_iter = iter(dataloader)
    print(next(data_iter))
    print(next(data_iter))

    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)

    # --- Embedding layers ---
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(6, 3)
    print(embedding_layer.weight)
    print(embedding_layer(torch.tensor([3])))

    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    max_length = 4
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)
    pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    print(pos_embeddings.shape)

    inputs = torch.tensor([[0.43, 0.15, 0.89], 
                            [0.55, 0.87, 0.66], 
                            [0.57, 0.85, 0.64], 
                            [0.22, 0.58, 0.33],
                            [0.77, 0.25, 0.10], 
                            [0.05, 0.80, 0.55]]
    )

    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0]) 
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query) 
    print(attn_scores_2)
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum() 
    print("Attention weights:", attn_weights_2_tmp) 
    print("Sum:", attn_weights_2_tmp.sum())

    def softmax_naive(x): 
        return torch.exp(x) / torch.exp(x).sum(dim=0) 
    attn_weights_2_naive = softmax_naive(attn_scores_2) 
    print("Attention weights:", attn_weights_2_naive) 
    print("Sum:", attn_weights_2_naive.sum())

    attn_weights_2 = torch.softmax(attn_scores_2, dim=0) 
    print("Attention weights:", attn_weights_2) 
    print("Sum:", attn_weights_2.sum())

    query = inputs[1] 
    context_vec_2 = torch.zeros(query.shape) 
    for i,x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i]*x_i 
    print(context_vec_2)


if __name__ == "__main__":
    main()
