"""
Calculate perplexity for MACSS abstracts using GPT-2.

This measures how "surprising" each abstract is to a language model.
Lower perplexity = more predictable/conventional text
Higher perplexity = more surprising/novel text
"""

import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def calculate_perplexity(text, model, tokenizer, device, max_length=1024):
    """Calculate perplexity for a single text."""
    try:
        # Tokenize
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(device)

        # Calculate loss
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()

        return perplexity

    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return None

def main():
    print("Loading GPT-2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model.eval()
    model.to(device)
    print("Model loaded!\n")

    # Load abstracts
    print("Loading abstracts...")
    df = pd.read_csv("data/macss_abstracts_full.csv")
    print(f"Loaded {len(df)} abstracts\n")

    # Calculate perplexity for originals
    print("Calculating perplexity for original abstracts...")
    perplexities = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ppl = calculate_perplexity(row['abstract'], model, tokenizer, device)
        perplexities.append(ppl)

    df['perplexity_original'] = perplexities

    # Save results
    output_path = "data/macss_abstracts_with_perplexity.csv"
    df.to_csv(output_path, index=False)

    print(f"\nResults saved to {output_path}")
    print(f"\nPerplexity Statistics:")
    print(f"  Mean: {df['perplexity_original'].mean():.2f}")
    print(f"  Median: {df['perplexity_original'].median():.2f}")
    print(f"  Std: {df['perplexity_original'].std():.2f}")
    print(f"  Min: {df['perplexity_original'].min():.2f}")
    print(f"  Max: {df['perplexity_original'].max():.2f}")

if __name__ == "__main__":
    main()
