# imports
import ollama
import psutil		# system resources
import subprocess	# dynamic
import os			# static
import re
import threading
import time
import csv

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

	# returns
	return {
		'temp': temp,
		'throttled': throttled,
		'clock_mhz': round(clock_mhz, 2),
		'watts': round(total_watts, 3)
    }

# get system-wide resource usage (CPU, RAM)
def get_resource_usage():

	# RAM/Swap
	memory = psutil.virtual_memory()
	swap = psutil.swap_memory()

	# CPU load (averages across 4 cores)
	cpu_load = psutil.cpu_percent(interval=None)

	# return
	return {
		'ram_used_gb': round(memory.used / (1024**3), 2),	# bytes --> gigabytes
		'cpu_load': cpu_load,
		'swap_used_mb': round(swap.used / (1024**2), 2)		# bytes --> megabytes
	}

# monitor metrics while model generates text
def monitor_loop(stop_event, metrics_list):

    # continue gathering metrics (while flag is false)
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
        time.sleep(0.1)     # sample metrics every 0.5 seconds

#########################################################
# initialize Ollama client
client = ollama.Client()

# define model, input prompt
model = "llama3.2:1b"
prompt = "Is Python an interpreted or compiled language?"
#########################################################

# set up event monitoring flag
stop_event = threading.Event()   # initialized to false
telemetry = []

# start monitoring
monitor_thread = threading.Thread(target=monitor_loop, args=(stop_event, telemetry))
monitor_thread.start()

# send query to LLM
response = client.generate(model=model, prompt=prompt)  # synchronous (blocking)

# stop monitoring
stop_event.set()        # executes once generate() is complete
monitor_thread.join()   # block main program until monitor thread completes

# save model response (once as metadata)
with open('benchmark_metadata.txt', 'w') as f:
    f.write(f"Model: {model}\nPrompt: {prompt}\nResponse: {response.response}")

# save results to CSV
with open('telemetry_results.csv', 'w', newline='') as f:
    fieldnames = ['timestamp', 'temp', 'throttled', 'clock_mhz', 'watts', 'cpu', 'ram', 'swap']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(telemetry)

# display model response
print("Response from Ollama:")
print("----------------------------------------")
print(response.response)
print("----------------------------------------")



### Notes

## Power Management Integrated Circuit (PMIC)
# Traffic controller for Pi's electricity
# - VDD_CORE => main ARM CPU cores
# - DDR_VDD => 8 GB RAM
# - 1V1_SYS => supporting logic for SoC
# - EXT5V => external power coming in (via USB-C)