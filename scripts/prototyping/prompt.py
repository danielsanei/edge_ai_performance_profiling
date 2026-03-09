# imports
import ollama
import psutil		# system resources
import subprocess	# dynamic
import os			# static
import re

# get hardware info (temperature, throttling)
def get_hardware_info():

	# temperature
	temp_raw = subprocess.check_output(['vcgencmd', 'measure_temp']).decode('utf-8')
	temp = float(re.search(r'\d+\.\d+', temp_raw).group())

	# (potential) throttling --> 0x0 = no throttling
	throttled = subprocess.check_output(['vcgencmd', 'get_throttled']).decode('utf-8').strip()

	# returns
	return temp, throttled

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

# initialize Ollama client
client = ollama.Client()

# define model, input prompt
model = "llama3.2:1b"
prompt = "Is Python an interpreted or compiled language?"

# BEFORE: record hardware, system resource metrics
start_temp, start_throttle = get_hardware_info()
pre_resources = get_resource_usage()

# send query to model
response = client.generate(model=model, prompt=prompt)

# AFTER: record hardware, system resource metrics
end_temp, end_throttle = get_hardware_info()
post_resources = get_resource_usage()

# display model response
print("Response from Ollama:")
print("----------------------------------------")
print(response.response)
print("----------------------------------------")

# display metrics
print(f"Start: {start_temp}°C | End: {end_temp}°C")
print(f"Throttling Status: {end_throttle}")
print(f"RAM Growth: {post_resources['ram_used_gb'] - pre_resources['ram_used_gb']} GB")
print(f"Peak CPU Load: {post_resources['cpu_load']}%")
print(f"Final Swap Usage: {post_resources['swap_used_mb']} MB")
