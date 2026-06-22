import requests
import subprocess

url = "https://raw.githubusercontent.com/RishitUmeshPadagatti/ML/refs/heads/main/1.py"
response = requests.get(url).text

# subprocess.run("clip", text=True, input=response) # windows
subprocess.run("pbcopy", text=True, input=response) # macos