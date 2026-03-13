"""
Generate HIGHLY manipulated (jargon-inflated) versions of MACSS abstracts.

AGGRESSIVE VERSION: Actually increases perplexity significantly.
"""

import pandas as pd
from openai import OpenAI
import os
from tqdm import tqdm
import time

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_manipulated_aggressive(abstract, model="gpt-3.5-turbo"):
    """Generate highly complex, perplexity-maximizing version of abstract."""

    prompt = f"""Rewrite this research abstract to MAXIMIZE linguistic surprise and complexity. The goal is to make the text as UNPREDICTABLE and UNUSUAL as possible while keeping the core meaning.

CRITICAL INSTRUCTIONS:
1. Use RARE and UNUSUAL vocabulary: Replace common words with archaic, technical, or obscure alternatives
   - Example: "study" → "scrutinize", "investigate" → "interrogate epistemologically"
   - Example: "data" → "empirical artifacts", "evidence" → "evidentiary substrate"

2. Create UNEXPECTED word combinations: Mix jargon from different disciplines
   - Combine social science with: physics terms, biology metaphors, engineering language
   - Example: "quantum entanglement of social variables", "thermodynamic equilibrium of attitudes"

3. Use COMPLEX nested syntax: Create long, winding sentences with multiple subordinate clauses
   - Minimum 40-50 words per sentence
   - Use semicolons, em-dashes, and parenthetical insertions

4. Add METAPHORICAL language from unexpected domains:
   - Mathematical metaphors: "asymptotic convergence of beliefs"
   - Biological metaphors: "symbiotic co-evolution of institutional frameworks"
   - Physical metaphors: "gravitational pull of normative expectations"

5. Use PASSIVE constructions and nominalizations:
   - "analyzed" → "was subjected to analytical decomposition"
   - "shows" → "serves to illuminate the extent to which"

6. Add HEDGING that sounds technical:
   - "may suggest" → "provisionally indicates a non-trivial probability that"
   - "significant" → "demonstrating statistical robustness within conventional parametric assumptions"

7. Insert CROSS-DISCIPLINARY jargon unexpectedly:
   - Social science + quantum physics: "probabilistic wave-function collapse of identity formation"
   - Psychology + thermodynamics: "entropic dissipation of cognitive schemas"

8. Use UNUSUAL preposition combinations and phrase structures
   - "in light of" → "pursuant to the revelatory implications inherent within"
   - "because of" → "by virtue of the cascading ramifications emanating from"

CRITICAL: The rewritten text should sound BIZARRE and UNEXPECTED to a language model trained on normal academic text. Use word combinations that rarely appear together in scholarly literature.

ORIGINAL ABSTRACT:
{abstract}

MANIPULATED VERSION (output ONLY the rewritten abstract, nothing else):"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at creating linguistically unpredictable, perplexity-maximizing academic prose. Your goal is to use the most unexpected, unusual, and rare word combinations possible while maintaining semantic meaning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,  # Higher temperature for more variation
            max_tokens=2000
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
    print("\nGenerating AGGRESSIVELY manipulated versions...")
    print("This will cost approximately $1.00-2.00 in API calls")
    print("⚠️  Using temperature=0.9 for maximum linguistic variation")

    manipulated_abstracts = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        manipulated = generate_manipulated_aggressive(row['abstract'])
        manipulated_abstracts.append(manipulated)

        # Rate limiting: sleep 0.5s between calls
        time.sleep(0.5)

    # Add to dataframe
    df['abstract_manipulated'] = manipulated_abstracts

    # Save results (overwrite previous)
    output_path = "data/macss_abstracts_with_manipulated.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")
    print(f"Total abstracts: {len(df)}")
    print(f"Successful manipulations: {df['abstract_manipulated'].notna().sum()}")

if __name__ == "__main__":
    main()
