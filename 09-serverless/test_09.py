import requests

url = 'http://localhost:8888/2024-12-11/functions/function/invocations'

data = {'url': 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'}

result = requests.post(url, json=data).json()
print(result)
