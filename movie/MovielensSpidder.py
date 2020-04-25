'''
Created on 2018年12月25日

@author: wenxiaofei-PC
'''
import json

import requests

HEADERS={
    "User-Agent":'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36',
    'Referer':'https://movielens.org/',}
MOVIE_LINK_PREFIX='https://www.dytt8.net'
BASE_URL='https://movielens.org/explore?tag=blood,dark%20humor,social%20commentary&hasRated=no&sortBy=prediction&page={}'
MAX_PAGE=2


class SpiderMovieLens:
    login_url="https://movielens.org/api/sessions"
    data={'userName':'wenxiaofei','password':'wen6224261995'}
    headers={'Accept':'application/json, text/plain, */*','Accept-Encoding':'gzip, deflate, br',
        'Accept-Language':'zh-CN,zh;q=0.9,en;q=0.8','Connection':'keep-alive','Content-Length':'52',
        'Content-Type':'application/json;charset=UTF-8',
        # 'Cookie':'_ga=GA1.2.1518125674.1543200958; _gid=GA1.2.1320827171.1550715929; _gat_gtag_UA_42433938_3=1',
        'Host':'movielens.org','Origin':'https//movielens.org','Referer':'https://movielens.org/login',
        'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36',}

    def __init__(self):
        '''
        constructor
        '''
        self.sess=requests.session()

    def login(self):
        response=self.sess.post(url=self.login_url,data=json.dumps(self.data),headers=self.headers)
        cookies=response.cookies
        # print(response.content)

    def download_top_picks(self,page_num):
        #3.访问主页
        # home_url="https://movielens.org/api/movies/explore?tag=blood,dark%20humor,social%20commentary&hasRated=no&sortBy=prediction&page={}".format(page_num)
        home_url="https://movielens.org/api/movies/explore?maxDaysAgo=90&maxFutureDays=0&hasRated=no&sortBy=releaseDate&page={}".format(page_num)

        headers2={'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Encoding':'gzip, deflate, br','Accept-Language':'zh-CN,zh;q=0.9,en;q=0.8',
            'Cache-Control':'max-age=0','Connection':'keep-alive',
            'Cookie': '_ga=GA1.2.1518125674.1543200958; _gid=GA1.2.1320827171.1550715929; ml4_session=60c56b6a17033de604c810175893c353e8c527c0_1fad7b15-bcf2-411f-8261-f35c0206d263',
            'Host':'movielens.org','Upgrade-Insecure-Requests':'1',
            'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36',}

        self.resp=self.sess.get(home_url,headers=headers2)
        self.content=self.resp.text
        # print(self.resp.content)

    def download_movie_by_id(self,movie_id):
        single_movie_url="https://movielens.org/api/movies/{}".format(movie_id)

        headers3={'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Encoding':'gzip, deflate, br','Accept-Language':'zh-CN,zh;q=0.9,en;q=0.8',
            'Cache-Control':'max-age=0','Connection':'keep-alive',
            'Cookie':'_ga=GA1.2.1518125674.1543200958; _gid=GA1.2.1320827171.1550715929; ml4_session=60c56b6a17033de604c810175893c353e8c527c0_1fad7b15-bcf2-411f-8261-f35c0206d263',
            'Host':'movielens.org','Upgrade-Insecure-Requests':'1',
            'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36',}

        resp=self.sess.get(single_movie_url,headers=headers3)
        self.single_movie_detail=resp.text


if __name__=='__main__':
    # 创建下载电影天堂的对象
    sml=SpiderMovieLens()
    #先模拟登录
    sml.login()
    #在请求数据
    # sml.download_top_picks(1)
    # sml.download_movie_by_id(3592)
    # sml.saveToDatabase()
    # print(sml.single_movie_detail)
    # print(sml.content)

