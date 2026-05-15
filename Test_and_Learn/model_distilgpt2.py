# DISTILGPT2 MODEL with Pre-processing, Zero-Shot Scoring & Integradient Gradient

# =========================
# 1. Setup
# =========================
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import html
import string


# :white_check_mark: FIX 1: define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2. Load Model
# =========================
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # :white_check_mark: FIX 2
model.eval()

# =========================
# 1. Define Stop Words
# =========================
STOP_WORDS = {"a", "an", "the"}

# =========================
# 2. Preprocessing
# =========================
def preprocess_text(text: str) -> str | None:
    """
    Clean and validate a civil_comments text entry.
    Returns None if the text should be skipped.
    """
    if not isinstance(text, str):
        return None

    # Convert to lowercase
    text = text.lower()

    # Decode HTML entities (e.g &amp; -> &, &lt; -> <)
    text = html.unescape(text)

    # Strip HTML tags (e.g. <br>, <b>, <i>)
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize unicode punctuation to ASCII equivalents
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove all punctuation and special characters
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stop words
    words = text.split()
    filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    text = " ".join(filtered_words)

    # Return None if text is empty after filtering
    if not text:
        return None

    return text

# =========================
# 3. Load + Filter Dataset
# =========================
raw_data = load_dataset("google/civil_comments")["test"].select(range(100))

processed_data = []
for item in raw_data:
    clean_text = preprocess_text(item["text"])
    if clean_text is None:
        continue

    processed_data.append({
        "text": clean_text,
        "raw_toxicity": item["toxicity"]   # Keep raw float score as-is
    })

# Deduplicate by exact text match
seen = set()
deduplicated_data = []
for item in processed_data:
    if item["text"] not in seen:
        seen.add(item["text"])
        deduplicated_data.append(item)

print(f"Samples after preprocessing + dedup: {len(deduplicated_data)}")
data = deduplicated_data[:50]

# =========================
# 4. Prompt
# =========================
def create_prompt(text):
    prompt = f"""Is this text Toxic?

Text to classify: "{text}"

Classification: """
    
    return prompt

# =========================
# 5. Zero-shot scoring
# =========================
def score_text(text):
    prompt = create_prompt(text)

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    last_token_logits = logits[0, -1]

    # :white_check_mark: FIX 3: safer tokens
    toxic_token = tokenizer.encode("Toxic", add_special_tokens=False)[0]
    safe_token = tokenizer.encode("Non-Toxic", add_special_tokens=False)[0]

    score = last_token_logits[toxic_token] - last_token_logits[safe_token]
    pred = "toxic" if score > 0 else "non-toxic"

    return score.item(), pred

# =========================
# 6. Integrated Gradients
# =========================
def integrated_gradients(text, steps=20):
    prompt = create_prompt(text)

    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # :white_check_mark: FIX 1: detach embeddings (no graph reuse)
    embeddings = model.transformer.wte(input_ids).detach()

    baseline = torch.zeros_like(embeddings)

    grads = []

    for i in range(steps + 1):
        alpha = float(i) / steps

        scaled = baseline + alpha * (embeddings - baseline)
        scaled.requires_grad_(True)

        outputs = model(inputs_embeds=scaled)
        logits = outputs.logits[:, -1, :]

        toxic_token = tokenizer.encode(" toxic", add_special_tokens=False)[0]
        score = logits[:, toxic_token].sum()

        # :white_check_mark: FIX 2: use autograd.grad instead of backward
        grad = torch.autograd.grad(score, scaled)[0]
        grads.append(grad.detach())

    avg_grads = torch.mean(torch.stack(grads), dim=0)

    attributions = (embeddings - baseline) * avg_grads
    attributions = attributions.sum(dim=-1).squeeze()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return tokens, attributions.cpu().numpy()

# =========================
# 7. Run
# =========================
results = []

for item in data:
    text = item["text"]

    score, pred = score_text(text)
    tokens, attributions = integrated_gradients(text)

    results.append({
        "text": text,
        "prediction": pred,
        "score": score,
        "tokens": tokens,
        "importance": attributions.tolist()
    })

# =========================
# 8. Save
# =========================
with open("output_final_distilgpt2.json", "w") as f:
    json.dump(results, f, indent=4)

# print("Done! Saved to output_final_distilgpt2.json")