import numpy as np
from llama_cpp import Llama
import os

def calculate_perplexity(llm, text):
    tokens = llm.tokenize(text.encode('utf-8'))
    llm.eval(tokens)
    all_logits = np.array(llm._scores)
    
    log_probs = []
    for i in range(1, len(tokens)):
        logits = all_logits[i-1]
        shift_logits = logits - np.max(logits)
        probs = np.exp(shift_logits) / np.sum(np.exp(shift_logits))
        log_probs.append(np.log(probs[tokens[i]]))
    
    return np.exp(-np.mean(log_probs))

# Initialize model once to save time
model_path = "./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
llm = Llama(model_path=model_path, logits_all=True, verbose=False, n_ctx=512)

# Test Prompts
test_cases = {
    "Fluent English (Expected: LOW)": "The quick brown fox jumps over the lazy dog.",
    "Unlikely English (Expected: MEDIUM)": "The purple elephant ate a blue refrigerator for breakfast.",
    "Complete Gibberish (Expected: HIGH)": "sjfd lksjdf 09234857 xcvm,nzxcv asdfqwer"
}

print(f"\n--- PPL Sensitivity Test (Model: {os.path.basename(model_path)}) ---")
for label, text in test_cases.items():
    ppl = calculate_perplexity(llm, text)
    print(f"[{label}]\n  Text: {text}\n  PPL:  {ppl:.4f}\n")