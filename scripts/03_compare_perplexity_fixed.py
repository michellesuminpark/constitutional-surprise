"""
Calculate perplexity for both original and manipulated abstracts,
then compute the treatment effect.

FIXED VERSION: Handles edge cases and outliers properly
"""

import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np

def calculate_perplexity(text, model, tokenizer, device, max_length=1024):
    """Calculate perplexity for a single text with proper error handling."""
    try:
        # Handle edge cases
        if not isinstance(text, str) or len(text.strip()) < 10:
            return None

        # Clean text
        text = str(text).strip()

        # Tokenize
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(device)

        # Skip if too short after tokenization
        if input_ids.size(1) < 5:
            return None

        # Calculate loss with sliding window for long texts
        max_model_length = model.config.n_positions
        stride = 512

        nlls = []
        for i in range(0, input_ids.size(1), stride):
            begin_loc = max(i + stride - max_model_length, 0)
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i

            input_slice = input_ids[:, begin_loc:end_loc]
            target_ids = input_slice.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_slice, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

        # Calculate perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        perplexity = ppl.item()

        # Sanity check: reject extreme outliers
        if perplexity > 1000:  # Likely an error
            print(f"Warning: Extreme perplexity {perplexity:.1f} for text: {text[:50]}...")
            return None

        return perplexity

    except Exception as e:
        print(f"Error calculating perplexity: {e}")
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

    # Filter out bad abstracts
    print("Filtering abstracts...")
    df = df[df['abstract'].notna()].copy()
    df = df[df['abstract'].str.len() >= 50].copy()
    df = df[df['abstract'] != "No abstract"].copy()
    print(f"After filtering: {len(df)} abstracts\n")

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

    # Remove rows where perplexity calculation failed
    df_clean = df[df['perplexity_original'].notna() & df['perplexity_manipulated'].notna()].copy()
    print(f"\nSuccessful calculations: {len(df_clean)}/{len(df)}")

    # Calculate change
    df_clean['perplexity_change'] = df_clean['perplexity_manipulated'] - df_clean['perplexity_original']

    # Save results
    output_path = "data/macss_full_with_perplexity.csv"
    df_clean.to_csv(output_path, index=False)

    # Print statistics
    print(f"\n{'='*60}")
    print("RESULTS: Treatment Effect of Manipulation on Perplexity")
    print(f"{'='*60}\n")

    print("ORIGINAL Abstracts:")
    print(f"  Mean perplexity: {df_clean['perplexity_original'].mean():.2f}")
    print(f"  Median: {df_clean['perplexity_original'].median():.2f}")
    print(f"  Std: {df_clean['perplexity_original'].std():.2f}")
    print(f"  Min: {df_clean['perplexity_original'].min():.2f}")
    print(f"  Max: {df_clean['perplexity_original'].max():.2f}\n")

    print("MANIPULATED Abstracts:")
    print(f"  Mean perplexity: {df_clean['perplexity_manipulated'].mean():.2f}")
    print(f"  Median: {df_clean['perplexity_manipulated'].median():.2f}")
    print(f"  Std: {df_clean['perplexity_manipulated'].std():.2f}")
    print(f"  Min: {df_clean['perplexity_manipulated'].min():.2f}")
    print(f"  Max: {df_clean['perplexity_manipulated'].max():.2f}\n")

    print("TREATMENT EFFECT:")
    mean_change = df_clean['perplexity_change'].mean()
    median_change = df_clean['perplexity_change'].median()
    pct_increase = (mean_change / df_clean['perplexity_original'].mean() * 100)

    print(f"  Mean change: {mean_change:+.2f} points")
    print(f"  Median change: {median_change:+.2f} points")
    print(f"  % change: {pct_increase:+.1f}%")
    print(f"  Std of change: {df_clean['perplexity_change'].std():.2f}")

    # Test significance
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(df_clean['perplexity_manipulated'],
                                       df_clean['perplexity_original'])
    print(f"\n  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.001:
        print(f"  *** Highly significant! (p < 0.001)")
    elif p_value < 0.05:
        print(f"  ** Significant (p < 0.05)")
    else:
        print(f"  Not significant")

    print(f"\n✅ Saved to {output_path}")

if __name__ == "__main__":
    main()
