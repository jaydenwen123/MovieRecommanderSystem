# -*- coding:utf-8 -*-
import json
import math
# import  random
import random

from django.shortcuts import render
from django.http import request,HttpResponse

from .recommand.movie_recommender_CNN import MovieRecommanderCNN
from .MovielensSpidder import SpiderMovieLens
from .models import MovieInfo
from datetime import date

from .MovieHotelSpidder import SpiderMovieHotel,BASE_URL

# Create your views here.

need_spider_save=False


def home(request):
    '''电影主页的控制器'''
    #测试和后台数据库的操作
    #     movie=MovieInfo(movie_title="逃离地球",movie_date=date(2019,2,20),movie_actors="吴京")
    #     movie.save()
    #     print("电影保存成功")

    #通过爬去电影天堂的数据，来展现

    # 创建下载电影天堂的对象
    # smh=SpiderMovieHotel()
    # smh.download_all_page_movies(BASE_URL)
    # #     print(smh.movie_link)
    # smh.parse_all_movie_detail()
    # print(smh.movie_detail)
    # movie_lists=smh.movie_detail
    #
    # #往数据库插入记录
    # for movie in movie_lists:
    #     mi=MovieInfo(movie_title=movie['moviename'],
    #         movie_aliasname=movie['aliasname'],
    #         movie_date=movie['date'],
    #         movie_year=movie['year'],
    #         movie_country=movie['country'],
    #         movie_type=movie['type'],
    #         movie_language=movie['language'],
    #         movie_duration=movie['duration'],
    #         movie_direcor=movie['director'],
    #         movie_profile=movie['profile'],
    #         movie_index=movie['index'],
    #         movie_download_url=movie['download_url'],
    #         movie_poster=movie['poster'],
    #         movie_capture=movie['capture'],
    #         movie_actors=movie['actors'],
    #         movie_swiftwriter=movie['swiftwriter'],
    #         movie_download_file=movie['download_file'],
    #         movie_word=movie['word'],
    #         movie_size=movie['size'],
    #     )
    #
    #     mi.save()
    #     print("电影保存成功")

    #从数据库读取电影数据
    # movie_lists=MovieInfo.objects.all()
    # context={"movie_list":movie_lists}

    #爬取movielens电影网上的数据
    if need_spider_save:
        mls=SpiderMovieLens()
        mls.login()
        for page in range(1,9,1):
            mls.download_top_picks(page)
            #
            movie_lists=json.loads(mls.content)
            if page==1:
                home_movie_list=movie_lists
            # #保存数据库
            for movie in movie_lists['data']['searchResults']:
                #修改评分
                movie['movie']['avgRating']=round(float(movie['movie']['avgRating']),1)

                movie['movie']['backdropPaths']=["https://image.tmdb.org/t/p/w300"+each for each in
                    movie['movie']["backdropPaths"]]
                # print(movie['movie']['backdropPaths'])
                if len(movie['movie']['backdropPaths'])>6:
                    movie['movie']['backdropPaths']=movie['movie']['backdropPaths'][0:6]

                if len(movie['movie']['actors'])>10:
                    movie['movie']['actors']=movie['movie']['actors'][0:10]

                if len(movie['movie']['title'])>64:
                    movie['movie']['title']=movie['movie']['title'][0:63]

                if len(movie['movie']['languages'])>4:
                    movie['movie']['languages']=movie['movie']['languages'][0:3]

                    #保存数据库
                    mi=MovieInfo(movie_title=(
                        "" if (movie['movie']['originalTitle'] is None) else movie['movie']['originalTitle']),
                        movie_aliasname=movie['movie']['title'],movie_year=movie['movie']['releaseYear'],
                        # movie_country=movie['movie']['country'],
                        movie_type=','.join(movie['movie']['genres']),
                        movie_language=','.join(movie['movie']['languages']),
                        movie_direcor=','.join(movie['movie']['directors']),movie_profile=(
                            "there is no profile" if (movie['movie']['plotSummary'] is None) else movie['movie'][
                                'plotSummary']),movie_index=movie['movie']['movieId'],
                        movie_capture=','.join(movie['movie']['backdropPaths']),
                        movie_actors=','.join(movie['movie']['actors']),movie_rating=movie['movie']['avgRating'],
                        # movie_download_url=movie['movie']['download_url'],
                        # movie_swiftwriter=movie['movie']['swiftwriter'],
                        # movie_download_file=movie['movie']['download_file'],
                        # movie_word=movie['movie']['word'],
                        # movie_size=movie['movie']['size'],
                    )
                    if movie['movie']['posterPath']!=None:
                        mi.movie_poster="https://image.tmdb.org/t/p/w154"+movie['movie']['posterPath'],
                    if 'releaseDate' not in movie['movie']:
                        mi.movie_date="1999-01-01"
                    if 'runtime' not in movie['movie']:
                        mi.movie_duration="",
                    mi.save()
                print("单条数据保存成功")
            print("第"+str(page)+"页数据保存成功")
    top_list=MovieInfo.objects.order_by("-movie_rating")
    action_list=MovieInfo.objects.filter(movie_type__icontains="action")
    comedy_list=MovieInfo.objects.filter(movie_type__icontains="comedy")
    if len(top_list)>10:
        top_list=top_list[0:10]
    if len(action_list)>10:
        action_list=action_list[0:10]  # 随机选择10个电影  # action_list=random.sample(action_list,10)
    if len(comedy_list)>10:
        comedy_list=comedy_list[0:10]  #随机选择十个电影，如果考虑分页，就不能这么使用  # comedy_list=random.sample(comedy_list,10)
    # print(top_list)
    context={"top_list":top_list,"action_list":action_list,"comedy_list":comedy_list}
    #     return HttpResponse("movie recommander system home page")
    return render(request,"index.html",context)


def movie_detail(request,movie_id):
    '''电影详情页的控制器'''
    # return HttpResponse("movie detail page,the current movie id="+movie_id)
    movie_id=int(movie_id)
    #从数据库查询数据
    movie=MovieInfo.objects.get(movie_index=movie_id)

    #相同类型的电影
    movie_type=movie.movie_type[1:len(movie.movie_type)-1].split(",")
    recommand_same_type=[]
    recommand_you_like=[]
    recommand_others_like=[]

    # for type in movie_type:
    #     m_list=MovieInfo.objects.filter(movie_type__icontains=type).exclude(movie_index=movie.movie_index).order_by("-movie_rating")
    #     if len(m_list)>5:
    #         recommand_same_type.extend(m_list[0:5])
    #根据所看的电影，采用推荐算法进行推荐
    mrc=MovieRecommanderCNN()
    mrc.generateMovieFeature()
    mrc.generateUserFeature()
    print("-----------------------")
    other_like_movie_ids=mrc.recommend_other_favorite_movie(movie.movie_index,1401)
    recommand_others_like=MovieInfo.objects.filter(movie_index__in=other_like_movie_ids).exclude(movie_index=movie.movie_index)
        # .order_by("-movie_rating")
    print("-----------------------")
    same_type_movie_ids=mrc.recommend_same_type_movie(movie.movie_index,20)
    recommand_same_type=MovieInfo.objects.filter(movie_index__in=same_type_movie_ids).exclude(
        movie_index=movie.movie_index)
        # .order_by("-movie_rating")
    print("-----------------------")
    you_like_movie_ids=mrc.recommend_your_favorite_movie(random.randint(1,5000),10)
    recommand_you_like=MovieInfo.objects.filter(movie_index__in=you_like_movie_ids).exclude(movie_index=movie.movie_index)
        # .order_by("-movie_rating")
    return render(request,"movie_detail.html",
        {"movie":movie,"recommand_same_type":recommand_same_type,"recommand_you_like":recommand_you_like,
            "recommand_others_like":recommand_others_like})


def collect_movie(request,source):
    #执行采集数据的操作
    for movie_id in range(3571,4001,1):
        #采集数据
        sml=SpiderMovieLens()
        sml.login()
        sml.download_movie_by_id(movie_id)
        movie=json.loads(sml.single_movie_detail)
        if "data" in movie and 'movieDetails' in movie['data']:
            movie=movie['data']['movieDetails']
            #修改评分
            movie['movie']['avgRating']=round(float(movie['movie']['avgRating']),1)
            movie['movie']['backdropPaths']=["https://image.tmdb.org/t/p/w300"+each for each in
                movie['movie']["backdropPaths"]]
            # print(movie['movie']['backdropPaths'])
            if len(movie['movie']['backdropPaths']) >6:
                movie['movie']['backdropPaths']=movie['movie']['backdropPaths'][0:6]

            if len(movie['movie']['actors']) >6:
                movie['movie']['actors']=movie['movie']['actors'][0:6]
            #保存数据库
            mi=MovieInfo(movie_title=("" if (movie['movie']['originalTitle'] is None) else movie['movie']['originalTitle']),
                movie_aliasname=movie['movie']['title'],
                movie_year=movie['movie']['releaseYear'],  # movie_country=movie['movie']['country'],
                movie_type=','.join(movie['movie']['genres']),
                movie_language=','.join(movie['movie']['languages']),
                movie_direcor=','.join(movie['movie']['directors']),
                movie_profile=("there is no profile" if (movie['movie']['plotSummary'] is None) else movie['movie']['plotSummary']),
                movie_index=movie['movie']['movieId'],
                movie_capture=','.join(movie['movie']['backdropPaths']),
                movie_actors=','.join(movie['movie']['actors']),
                movie_rating=movie['movie']['avgRating'],
                # movie_download_url=movie['movie']['download_url'],
                # movie_swiftwriter=movie['movie']['swiftwriter'],
                # movie_download_file=movie['movie']['download_file'],
                # movie_word=movie['movie']['word'],
                # movie_size=movie['movie']['size'],
            )
            if movie['movie']['posterPath']!=None:
                mi.movie_poster="https://image.tmdb.org/t/p/w154"+movie['movie']['posterPath'],
            if 'releaseDate' not in movie['movie']:
                mi.movie_date="1999-01-01"
            if 'runtime' not in movie['movie']:
                mi.movie_duration="",
            mi.save()
            print("第"+str(movie_id)+"条电影数据保存成功")
    if source=="moviegods":
        source="电影天堂"

    return HttpResponse("<p>{}的电影数据采集成功<a href='/movie/'>返回首页</a></p>".format(source))