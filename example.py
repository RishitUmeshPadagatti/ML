import requests

url = "https://raw.githubusercontent.com/RishitUmeshPadagatti/ML/refs/heads/main/1.py"
response = requests.get(url).text
print(response)