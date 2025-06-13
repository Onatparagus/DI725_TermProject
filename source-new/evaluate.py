import json
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from nltk import download
download('punkt')

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')  # sometimes needed for WordNet synonym lookups


# Load JSON
with open("out/final/multi_promptset_captions_beams1_maxtokens50 3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

refusal_phrases = [
    "Sorry, as a base VLM I am not trained to answer this question.",
    "unanswerable",
    "no",
    "nothing",
    "no image"
]

results = {}
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

for set_name in ["basic", "partial", "descriptive"]:
    bleu_scores = []
    meteor_scores = []
    rouge_scores = []
    count = 0

    for item in data:
        refs = [r.strip() for r in item["reference_captions"]]
        for gen in item["generated_captions"].get(set_name, []):
            response = gen.split("\n", 1)[-1].strip().lower()
            if any(response == phrase.lower() for phrase in refusal_phrases) or len(response.split()) < 3:
                continue
            count += 1

            ref_tokens = [word_tokenize(ref.lower()) for ref in refs]
            gen_tokens = word_tokenize(response)
            bleu_scores.append(corpus_bleu([ref_tokens], [gen_tokens]))
            meteor_scores.append(meteor_score(
                [word_tokenize(ref.lower()) for ref in refs],
                word_tokenize(response)
            ))
            rouge_scores.append(scorer.score(response, refs[0])["rougeL"].fmeasure)

    results[set_name] = {
        "samples": count,
        "BLEU": np.mean(bleu_scores) if bleu_scores else 0,
        "METEOR": np.mean(meteor_scores) if meteor_scores else 0,
        "ROUGE-L": np.mean(rouge_scores) if rouge_scores else 0
    }

# Output results
df = pd.DataFrame.from_dict(results, orient="index")
print(df)
