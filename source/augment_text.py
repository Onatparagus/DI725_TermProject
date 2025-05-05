from transformers import pipeline
import random

paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
translator_en_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
translator_fr_en = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

def paraphrase_caption(caption):
    result = paraphraser(caption, max_length=60, num_return_sequences=1)[0]['generated_text']
    return result

def back_translate_caption(caption):
    fr = translator_en_fr(caption)[0]['translation_text']
    back = translator_fr_en(fr)[0]['translation_text']
    return back

def augment_captions(original_captions):
    augmented = []
    for cap in original_captions:
        if random.random() < 0.5:
            augmented.append(paraphrase_caption(cap))
        else:
            augmented.append(back_translate_caption(cap))
    return augmented