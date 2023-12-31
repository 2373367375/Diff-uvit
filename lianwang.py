import requests
import re

# 1. 登录请求地址，下面是西电的登录地址
url = 'https://w.xidian.edu.cn/srun_portal_pc.php'

# 2. 登录请求发送的数据，下面是登录西电校园网所必须的字段
data = {
    'action': 'login',
    'ac_id': '8',
    "username": '22011210651',
    "password": '080337'
}

# 3. 复制浏览器的请求头，下面是西电的校园网登录请求头
header = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "Cache-Control": "max-age=0",
    "Connectin": "keep-alive",
    "Host": "w.xidian.edu.cn",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
}

# 4. 以post方式发送登录请求；返回状态码为200则表示请求成功
response = requests.post(url, data, headers=header)
print("状态码:", response.status_code)
print(response.text)
print(re.findall(r'.*<ul' ,response.text)[0])