"""
Calculate perplexity for both original and manipulated abstracts,
then compute the treatment effect.
"""

import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def calculate_perplexity(text, model, tokenizer, device, max_length=1024):
    """Calculate perplexity for a single text."""
    try:
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        return torch.exp(loss).item()

    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("Loading GPT-2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model.eval()
    model.to(device)

    # Load data with manipulated versions
    print("Loading data...")
    df = pd.read_csv("data/macss_abstracts_with_manipulated.csv")
    print(f"Loaded {len(df)} abstracts\n")

    # Calculate perplexity for originals
    print("Calculating perplexity for ORIGINAL abstracts...")
    ppl_original = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        ppl = calculate_perplexity(row['abstract'], model, tokenizer, device)
        ppl_original.append(ppl)

    # Calculate perplexity for manipulated
    print("\nCalculating perplexity for MANIPULATED abstracts...")
    ppl_manipulated = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if pd.notna(row['abstract_manipulated']):
            ppl = calculate_perplexity(row['abstract_manipulated'], model, tokenizer, device)
        else:
            ppl = None
        ppl_manipulated.append(ppl)

    # Add to dataframe
    df['perplexity_original'] = ppl_original
    df['perplexity_manipulated'] = ppl_manipulated
    df['perplexity_change'] = df['perplexity_manipulated'] - df['perplexity_original']

    # Save results
    output_path = "data/macss_full_with_perplexity.csv"
    df.to_csv(output_path, index=False)

    # Print statistics
    print(f"\n{'='*60}")
    print("RESULTS: Treatment Effect of Manipulation on Perplexity")
    print(f"{'='*60}\n")

    print("ORIGINAL Abstracts:")
    print(f"  Mean perplexity: {df['perplexity_original'].mean():.2f}")
    print(f"  Median: {df['perplexity_original'].median():.2f}")
    print(f"  Std: {df['perplexity_original'].std():.2f}\n")

    print("MANIPULATED Abstracts:")
    print(f"  Mean perplexity: {df['perplexity_manipulated'].mean():.2f}")
    print(f"  Median: {df['perplexity_manipulated'].median():.2f}")
    print(f"  Std: {df['perplexity_manipulated'].std():.2f}\n")

    print("TREATMENT EFFECT:")
    print(f"  Mean change: +{df['perplexity_change'].mean():.2f} points")
    print(f"  Median change: +{df['perplexity_change'].median():.2f} points")
    print(f"  % increase: {(df['perplexity_change'].mean() / df['perplexity_original'].mean() * 100):.1f}%")

    print(f"\n✅ Saved to {output_path}")

if __name__ == "__main__":
    main()
