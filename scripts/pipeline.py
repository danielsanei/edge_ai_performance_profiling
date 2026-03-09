# imports
import numpy as np
from llama_cpp import Llama     # replace ollama API with llama-cpp-python backend
import psutil                   # system resources
import subprocess               # dynamic
import os                       # static
import re
import threading
import time
import csv
import json

# get hardware info (temperature, throttling)
def get_hardware_info():
    # temperature
    temp_raw = subprocess.check_output(['vcgencmd', 'measure_temp']).decode('utf-8')
    temp = float(re.search(r'\d+\.\d+', temp_raw).group())

    # (potential) throttling --> 0x0 = no throttling (NOTE: take raw value)
    throttled = subprocess.check_output(['vcgencmd', 'get_throttled']).decode('utf-8').strip().split('=')[-1]

    # clock speed (Hz --> MHz)
    clock_raw = subprocess.check_output(['vcgencmd', 'measure_clock', 'arm']).decode('utf-8')
    clock_mhz = int(re.search(r'=(\d+)', clock_raw).group(1)) / (1000000)

    # power estimation using P = VI (read current drawn by CPU cores since Pi doesn't have Watts meter)
    pmic_raw = subprocess.check_output(['vcgencmd', 'pmic_read_adc']).decode('utf-8')
    currents = [float(x) for x in re.findall(r'current\(\d+\)=([\d.]+)A', pmic_raw)]
    volts = [float(x) for x in re.findall(r'volt\(\d+\)=([\d.]+)V', pmic_raw)]
    total_watts = sum(i * v for i, v in zip(currents, volts))

    return {
        'temp': temp,
        'throttled': throttled,
        'clock_mhz': round(clock_mhz, 2),
        'watts': round(total_watts, 3)
    }

# get system-wide resource usage (CPU, RAM)
def get_resource_usage():
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cpu_load = psutil.cpu_percent(interval=None)

    return {
        'ram_used_gb': round(memory.used / (1024**3), 2),
        'cpu_load': cpu_load,
        'swap_used_mb': round(swap.used / (1024**2), 2)
    }

# monitor metrics while model generates text
def monitor_loop(stop_event, metrics_list):
    while not stop_event.is_set():
        hardware = get_hardware_info()
        resources = get_resource_usage()
        metrics_list.append({
            'timestamp': time.time(),
            'temp': hardware['temp'],
            'throttled': hardware['throttled'],
            'clock_mhz': hardware['clock_mhz'], 
            'watts': hardware['watts'],
            'cpu': resources['cpu_load'],
            'ram': resources['ram_used_gb'],
            'swap': resources['swap_used_mb']
        })
        time.sleep(0.1)

# calculate Perplexity (PPL) using llama-cpp logits
def calculate_perplexity(llm, tokens):
    llm.eval(tokens)
    all_logits = np.array(llm._scores)
    log_probs = []
    for i in range(1, len(tokens)):
        logits = all_logits[i-1]
        shift_logits = logits - np.max(logits)
        probs = np.exp(shift_logits) / np.sum(np.exp(shift_logits))
        log_probs.append(np.log(probs[tokens[i]]))
    return np.exp(-np.mean(log_probs))

# initializations for model versions, JSON file w/ prompts
model_dir = "../models"
models = [f for f in os.listdir(model_dir) if f.endswith(".gguf")]
with open("../prompts/prompts.json", "r") as f:
    prompt_bank = json.load(f)

# master CSV for storing all results
master_csv_path = '../data/master_results.csv'
if not os.path.exists(master_csv_path):
    with open(master_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # NEW: Added columns for Load Duration, Prompt Tokens, and Prompt Eval Rate
        writer.writerow(['Model', 'Quant', 'Context', 'Threads', 'Prompt_ID', 'TPS', 'PPL', 'Total_Time', 'TTFT', 'Load_Duration', 'Prompt_Tokens', 'Prompt_Eval_Rate'])

# main pipeline loop (iterate through all models, all prompts)
contexts = [512, 2048, 4096] #
threads = [2, 4] #
current_run = 0
total_runs = len(models) * len(prompt_bank["prompts"]) * len(contexts) * len(threads)
for n_ctx in contexts:
    for n_threads in threads:
        for m_idx, model_file in enumerate(models):
            model_path = os.path.join(model_dir, model_file)

            # record start time for model loading
            load_start = time.perf_counter()

            # initialize model (load once per model file)
            # add type_k=8 for 8-bit KV cache quantization to reduce memory pressure
            llm = Llama(model_path=model_path, logits_all=True, n_ctx=n_ctx, n_threads=n_threads, type_k=8, verbose=False, seed=42)

            # compute model load duration
            load_duration = time.perf_counter() - load_start

            # clean up naming convention for terminal progress printing
            name_clean = model_file.replace(".gguf", "").replace("-Instruct", "")
            
            # separate model name and quantization for terminal progress printing
            parts = name_clean.split("-")
            model_name = "-".join(parts[:-1])
            quantization = parts[-1]

            # terminal progress prints
            print(f"\n" + "="*50)
            print(f"MODEL: {model_name} | CTX: {n_ctx} | THR: {n_threads}")
            print(f"PROGRESS: {current_run}/{total_runs}")
            print("="*50)

            # run each prompt for current model
            for prompt_data in prompt_bank["prompts"]:
                current_run += 1
                prompt_id = prompt_data["id"]
                prompt_text = prompt_data["text"]
                
                stop_event = threading.Event()
                telemetry = []

                # start monitoring
                monitor_thread = threading.Thread(target=monitor_loop, args=(stop_event, telemetry))
                monitor_thread.start()

                # record start time (latency calculation)
                start_time = time.perf_counter()

                # generate response
                stream = llm(prompt_text, max_tokens=128, temperature=0.0, stream=True)

                # measure time to first token (TTFT)
                first_token_time = None
                response_text = ""
                for i, chunk in enumerate(stream):
                    if i == 0:
                        first_token_time = time.perf_counter()
                    response_text += chunk["choices"][0]["text"]
                                
                # record end time (latency calculation)
                end_time = time.perf_counter()
                
                # stop monitoring
                stop_event.set()
                monitor_thread.join()

                # use tokenize to get the completion token count
                eval_count = len(llm.tokenize(response_text.encode('utf-8')))
                total_duration = end_time - start_time
                
                # calculate PPL for current prompt
                tokens = llm.tokenize(prompt_text.encode('utf-8'))
                ppl_score = calculate_perplexity(llm, tokens)

                # save results to model-specific CSV file
                csv_file = f"../data/{name_clean}_C{n_ctx}_T{n_threads}_P{prompt_id}.csv"
                with open(csv_file, 'w', newline='') as f:
                    fieldnames = ['timestamp', 'temp', 'throttled', 'clock_mhz', 'watts', 'cpu', 'ram', 'swap']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(telemetry)

                # compute rates
                ttft = first_token_time - start_time if first_token_time else 0
                eval_rate = eval_count / (end_time - first_token_time) if first_token_time else 0

                # prompt metrics
                prompt_tokens = len(tokens)
                prefill_duration = first_token_time - start_time if first_token_time else 0
                prompt_eval_rate = prompt_tokens / prefill_duration if prefill_duration > 0 else 0

                # hardware metrics
                avg_watts = np.mean([d['watts'] for d in telemetry]) if telemetry else 0
                peak_temp = np.max([d['temp'] for d in telemetry]) if telemetry else 0
                avg_cpu = np.mean([d['cpu'] for d in telemetry]) if telemetry else 0
                peak_ram = np.max([d['ram'] for d in telemetry]) if telemetry else 0
                avg_clock = np.mean([d['clock_mhz'] for d in telemetry]) if telemetry else 0
                throttled_any = any(d['throttled'] != '0x0' for d in telemetry)

                # save results to metadata file (for charts, plots, analysis)
                with open(master_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([model_name, quantization, n_ctx, n_threads, prompt_id, round(eval_rate, 2), round(ppl_score, 4), round(total_duration, 2), round(ttft, 3), round(load_duration, 2), prompt_tokens, round(prompt_eval_rate, 2), round(avg_watts, 2), round(peak_temp, 1), round(avg_cpu, 1), round(peak_ram, 2), round(avg_clock, 1), int(throttled_any)])                
                # write results to benchmark text file (human-readable format)
                with open('../data/benchmark_results.txt', 'a') as f:
                    f.writelines([
                        f"\n{'='*92}\n",
                        f"Model: {name_clean} | Ctx: {n_ctx} | Thr: {n_threads}\n",
                        f"Prompt: {prompt_id}\n",
                        f"{'-'*92}\n",
                        f"Response: {response_text}\n", # NEW: Added response text back to the log
                        f"{'-'*92}\n",
                        f"Model Load Duration: {load_duration:.2f}s\n", # NEW: Added Load Duration to text log
                        f"Total Latency: {total_duration:.2f}s | TTFT: {ttft:.3f}s\n",
                        f"Prompt Tokens: {prompt_tokens} | Prompt Eval Rate: {prompt_eval_rate:.2f} tokens/s\n", # NEW: Added Prompt metrics to text log
                        f"Generated Tokens: {eval_count} | Token Gen Rate (TPS): {eval_rate:.2f} tokens/s\n", # NEW: Clarified generated vs prompt rates
                        f"Perplexity (PPL): {ppl_score:.4f}\n",
                        f"{'='*92}\n",
                        f"Hardware Summary: Avg Power: {avg_watts:.2f}W | Peak Temp: {peak_temp:.1f}°C | Peak RAM: {peak_ram:.2f}GB\n",
                        f"Throttled: {'YES' if throttled_any else 'NO'} | Avg CPU Load: {avg_cpu:.1f}% | Avg Clock: {avg_clock:.1f}MHz\n",
                        f"{'='*92}\n"
                    ])

                # terminal display print
                progress = (current_run / total_runs) * 100
                print(f"Prompt {prompt_id} Done. [{current_run}/{total_runs}] {progress:.1f}% COMPLETE")
            
            # clear memory before loading next model
            del llm

            # cooldown period (start at same baseline temperature)
            time.sleep(30) # cooldown period