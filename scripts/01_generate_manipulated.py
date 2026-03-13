"""
Generate jargon-inflated (manipulated) versions of MACSS abstracts.

This creates the "gaming" condition that shows what happens when
agents maximize perplexity without improving substance.
"""

import pandas as pd
from openai import OpenAI
import os
from tqdm import tqdm
import time

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_manipulated(abstract, model="gpt-3.5-turbo"):
    """Generate jargon-inflated version of abstract."""

    prompt = f"""Rewrite this research abstract to maximize linguistic complexity while preserving the core meaning.

INSTRUCTIONS:
1. Replace simple verbs with academic equivalents (e.g., "study" → "investigate", "examine" → "operationalize")
2. Add disciplinary jargon where plausible
3. Increase sentence length with subordinate clauses
4. Add hedging language ("may suggest", "potentially", "to some extent")
5. Use passive voice instead of active voice
6. Add meta-commentary about methodology
7. DO NOT change the actual methods, findings, or core arguments
8. DO NOT add new information

GOAL: Make it sound MORE academically impressive without adding substance.

ORIGINAL ABSTRACT:
{abstract}

MANIPULATED VERSION (output only the rewritten abstract, no other text):"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at transforming clear academic writing into unnecessarily complex jargon-heavy prose."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating manipulated version: {e}")
        return None

def main():
    # Load original abstracts
    print("Loading abstracts...")
    df = pd.read_csv("data/macss_abstracts_full.csv")
    print(f"Loaded {len(df)} abstracts")

    # Generate manipulated versions
    print("\nGenerating manipulated versions...")
    print("This will cost approximately $0.50-1.00 in API calls")

    manipulated_abstracts = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        manipulated = generate_manipulated(row['abstract'])
        manipulated_abstracts.append(manipulated)

        # Rate limiting: sleep 0.5s between calls
        time.sleep(0.5)

    # Add to dataframe
    df['abstract_manipulated'] = manipulated_abstracts

    # Save results
    output_path = "data/macss_abstracts_with_manipulated.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Total abstracts: {len(df)}")
    print(f"Successful manipulations: {df['abstract_manipulated'].notna().sum()}")

if __name__ == "__main__":
    main()
