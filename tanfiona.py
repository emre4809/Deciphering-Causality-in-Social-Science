from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "tanfiona/unicausal-tok-baseline"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)


nlp = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    # device=0  # Set to 0 to use the first GPU
)

sentence = 'Single players in the treatment with replacement showed an increased baseline cooperation probability.'


results = nlp(sentence)

cause = []
effect = []

label_mapping = {
    "LABEL_0": "B-C",
    "LABEL_1": "B-E",
    "LABEL_2": "I-C",
    "LABEL_3": "I-E",
}

for res in results:
    if res['entity_group'] in {"LABEL_0", "LABEL_2"}:  # Cause-related labels
        cause.append(res['word'])
    elif res['entity_group'] in {"LABEL_1", "LABEL_3"}:  # Effect-related labels
        effect.append(res['word'])

cause_text = " ".join(cause)
effect_text = " ".join(effect)

print(f"Cause: {cause_text}")
print(f"Effect: {effect_text}")
