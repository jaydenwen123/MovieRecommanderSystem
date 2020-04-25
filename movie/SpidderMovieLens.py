'''
Created on 2018年12月23日

@author: wenxiaofei-PC
'''

import requests
import json
import time
import  random

login_url="https://movielens.org/api/sessions"
data={'userName':'wenxiaofei','password':'wen6224261995'}
headers={'Accept':'application/json, text/plain, */*','Accept-Encoding':'gzip, deflate, br',
    'Accept-Language':'zh-CN,zh;q=0.9,en;q=0.8','Connection':'keep-alive','Content-Length':'52',
    'Content-Type':'application/json;charset=UTF-8',
    'Host':'movielens.org','Origin':'https//movielens.org','Referer':'https://movielens.org/login',
    'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36',}
#1.首先模拟登陆
# 此处必须使用session对象
sess=requests.session()
response=sess.post(url=login_url,data=json.dumps(data),headers=headers)
#2.保存到文件中
# with open("renren.html","wb") as movielens:
#     movielens.write(response.content)
# print(response.cookies)
cookies=response.cookies
print("----------------")
print(response.content)
#3.访问主页
time.sleep(6)
home_url="https://movielens.org/api/users/me"
# home_url="https://movielens.org/api/showctrl/user-info"

# home_url="https://movielens.org/api/users/me/frontpage"

# home_url="https://movielens.org/api/movies/explore?tag=blood,dark%20humor,social%20commentary&hasRated=no&sortBy=prediction"
# home_url="https://movielens.org/api/movies/explore?tag=blood,dark%20humor,social%20commentary&hasRated=no&sortBy=prediction"
# home_url="https://movielens.org/api/movies/explore?tag=blood,dark%20humor,social%20commentary&hasRated=no&sortBy=prediction&page=2"

headers2={
'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
'Accept-Encoding':'gzip, deflate, br',
'Accept-Language':'zh-CN,zh;q=0.9,en;q=0.8',
'Cache-Control':'max-age=0',
'Connection':'keep-alive',
'Cookie':'_ga=GA1.2.1518125674.1543200958; _gid=GA1.2.1320827171.1550715929; ml4_session=60c56b6a17033de604c810175893c353e8c527c0_1fad7b15-bcf2-411f-8261-f35c0206d263',
'Host':'movielens.org',
'Upgrade-Insecure-Requests':'1',
'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36',
}

resp=requests.get(home_url,headers=headers2)
# resp=requests.get(home_url,headers=headers,auth=("wenxiaofei","wen6224261995"))
print(resp.text)
# with open("profile.html","wb") as profile:
#     profile.write(resp.content)

print('OK')