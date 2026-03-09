from ollama import chat

# Store metrics here later
stats = {}

stream = chat(
    model='gemma2:2b', # Note: Using gemma2 as discussed
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
    # 1. Print the content as it arrives
    print(chunk['message']['content'], end='', flush=True)
    
    # 2. Capture metrics from the final chunk
    if 'eval_count' in chunk:
        stats = chunk

# Now you can use the metrics for your ECE 285 report
print(f"\n\n--- BENCHMARK DATA ---")
print(f"Tokens Generated: {stats.get('eval_count')}")
print(f"Tokens Per Second: {stats.get('eval_count') / (stats.get('eval_duration') / 1e9):.2f}")
