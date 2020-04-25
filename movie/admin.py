from django.contrib import admin
from .models import MovieInfo


class MovieAdmin(admin.ModelAdmin):
    '''设置后台admin管理显示的样式'''
    list_display=("movie_title","movie_date","movie_actors")

# Register your models here.
admin.site.register(MovieInfo,MovieAdmin)

