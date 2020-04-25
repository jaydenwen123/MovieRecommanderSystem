'''
Created on 2018年12月25日

@author: wenxiaofei-PC
'''
import requests
from lxml import etree
from _operator import indexOf
HEADERS = {
         "User-Agent":"Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36",
         }
MOVIE_LINK_PREFIX = 'https://www.dytt8.net'
BASE_URL = 'https://www.dytt8.net/html/gndy/dyzz/list_23_{}.html'
MAX_PAGE = 2
class SpiderMovieHotel:
    
    def __init__(self):
        '''
        constructor
        '''
        self.movie_link = []
    
    
    def get_movie_info(self):
        '''
            get the movie infos
        '''
        return self.movie
    
    def download_movie(self, url):
        '''
             download the movie html page
        '''
        response = requests.get(url, headers=HEADERS,) 
        self.content = response.content
#         self.text=self.content.decode("gbk")
        self.text = response.text
#         with open("movie_hotel.html",'w') as f:
#             f.write(self.text)
       


        
    
    def parse_movie2(self):
        movieHtml = etree.HTML(self.content.decode("gbk"))
#         print(self.content.decode("gbk"))
#         print(movieHtml)
        self.movieHtml = movieHtml.xpath("//div[@class='co_content8']/ul")[0]
    #         print(self.movieHtml)
        movie_links = self.movieHtml.xpath(".//table")[0].xpath(".//a[@class='ulink']/@href")
        self.movieHtml = self.movieHtml.xpath(".//table")
        print(len(movie_links))
        self.movie_link = []
        each_page_movie_link = []
        # 解析一页电影中的每个电影的链接
        for singleMovie in self.movieHtml:
            sublink = "{}" + singleMovie.xpath(".//a[@class='ulink']/@href")[0]
            each_page_movie_link.append(sublink.format(MOVIE_LINK_PREFIX))
        self.movie_link.append(each_page_movie_link)
        print(each_page_movie_link)
        print(len(each_page_movie_link))
    #         print(etree.tostring(self.movieHtml[0],encoding='gbk').decode("gbk"))
    
    def parse_movie_link_with_page(self):
        '''
            解析出来一页的电影链接
        '''
        movieHtml = etree.HTML(self.text)
#         print(etree.tostring(movieHtml,encoding='gbk').decode("gbk"))
        each_movie_links = movieHtml.xpath("//div[@class='co_content8']//ul//table//a/@href")
        each_movie_links = [MOVIE_LINK_PREFIX + each_movie for each_movie in each_movie_links]
#         print(each_movie_links)
#         index=0
#         for movie_link in each_movie_links:
#             movie_link=MOVIE_LINK_PREFIX+movie_link
#             each_movie_links[index]=movie_link
#             index=index+1
#         print(len(each_movie_links))
#         print(each_movie_links)
        self.movie_link.append(each_movie_links)
    
    def  download_all_page_movies(self, baseurl):
        for i in range(1, MAX_PAGE, 1):
            url = baseurl.format(i)
            self.download_movie(url)
            self.parse_movie_link_with_page()
    
    
    def download_movie_detail(self,url):
        reponse=requests.get(url,headers=HEADERS)
        self.movie_detail_html=reponse.text
        self.movie_detail_content=reponse.content
    
    def parse_all_movie_detail(self):
        '''
        解析电影的信息
        '''
        self.movie_detail=[]
        #电影的id
        index=0;
        for each_page in self.movie_link:
            for movie_url in each_page:
                self.download_movie_detail(movie_url)
#                 print(self.content.decode("gbk"))
                movie_detail=etree.HTML(self.movie_detail_content.decode("gbk"))
                movie_detail_text=movie_detail.xpath("//div[@id='Zoom']//p/text()")
                movie_detail_img=movie_detail.xpath("//div[@id='Zoom']//p/img/@src")
                movie_download_url=movie_detail.xpath("//div[@id='Zoom']//a/@href")
#                 print(etree.tostring(movie_download_url,encoding='gbk').decode('gbk'))
#                 print(movie_download_url)
#                 print(len(movie_detail))
#                 print(movie_detail_text)
#                 print(movie_detail_img)
                #遍历信息，然后进行保存
                movie=self.parse_single_movie_detail(movie_detail_text,movie_detail_img,movie_download_url)
#                 print(etree.tostring(movie_detail,encoding='gbk').decode("gbk"))
                movie['index']=index
                self.movie_detail.append(movie)
                index=index+1
#                 break
            break
    
    def parse_single_movie_detail(self,movie_detail_text,movie_detail_img,download_url):
        '''
        解析单部电影的数据
        '''
        movie={}
#         print(len(download_url))
#         print(download_url)
        if download_url is not None and len(download_url)==2:
            movie['download_url']=download_url[0]
            movie['download_file']=download_url[1]

        else:
            movie['download_url']=""
        if movie_detail_img is not  None  and len(movie_detail_img)==2 :
            movie['poster']=movie_detail_img[0]
            movie['capture']=movie_detail_img[1]
        for index,item in enumerate(movie_detail_text):
            if item.startswith("◎译　　名"):
                alisename=item.replace("◎译　　名","").strip().split("/")[0]
                if "：" in alisename:
                    movie['aliasname']=alisename.split("：")[0]
                else:
                    movie['aliasname']=alisename
            elif item.startswith("◎片　　名"):
                movie['moviename']=item.replace("◎片　　名","").strip()
            elif item.startswith("◎年　　代"):
                movie['year']=item.replace("◎年　　代","").strip()
            elif item.startswith("◎产　　地"):
                movie['country']=item.replace("◎产　　地","").strip()
            elif item.startswith("◎类　　别"):
                movie['type']=item.replace("◎类　　别","").strip()
            elif item.startswith("◎语　　言"):
                movie['language']=item.replace("◎语　　言","").strip()
            elif item.startswith("◎字　　幕"):
                movie['word']=item.replace("◎字　　幕","").strip()
            elif item.startswith("◎上映日期"):
                movie['date']=item.replace("◎上映日期","").strip()
            elif item.startswith("◎视频尺寸"):
                movie['size']=item.replace("◎视频尺寸","").strip()
            elif item.startswith("◎片　　长"):
                movie['duration']=item.replace("◎片　　长","").strip()
            elif item.startswith("◎导　　演"):
                movie['director']=item.replace("◎导　　演","").strip()   
            
            elif item.startswith("◎编　　剧"):
                movie['swiftwriter']=item.replace("◎编　　剧","").strip()  
            
            #需要特殊处理
            elif item.startswith("◎主　　演"):
                actors=[item.replace("◎主　　演","").strip() ]
                flag=index+1
                while not movie_detail_text[flag].startswith("◎简　　介"):
                    actors.append(movie_detail_text[flag].strip())
                    flag=flag+1
                movie['actors']= actors
            elif item.startswith("◎简　　介"):
                movie['profile']=movie_detail_text[index+1].strip()

            #判断是否某些键不存在
            if 'size' not in movie:
                movie['size']=''

            if 'word' not in movie:
                movie['word']=''

            if 'download_file' not in movie:
                movie['download_file']=''
            if 'swiftwriter' not in movie:
                movie['swiftwriter']=''
            if 'actors' not in movie:
                movie['actors']=''
            elif len(movie['actors']) > 3:
                movie['actors']=movie['actors'][0:2]
                print(movie['actors'])
        return movie
        
if __name__ == '__main__':
    # 电影天堂的最新电影
    url = "https://www.dytt8.net/html/gndy/dyzz/index.html"
    # 创建下载电影天堂的对象
    smh = SpiderMovieHotel()   
    
    smh.download_all_page_movies(BASE_URL)
#     print(smh.movie_link)
    smh.parse_all_movie_detail()
    print(smh.movie_detail)
    # 下载电影数据，然后保存
#     smh.download_movie(url)
    # 解析下载下来的电影数据
#     smh.parse_movie_link_with_page()
    # 打印电影数据
#     movies=smh.get_movie_info()
#     print(movies)
