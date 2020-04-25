from django.db import models

# Create your models here.
class MovieInfo(models.Model):
    movie_title=models.CharField(max_length=256)
    movie_aliasname=models.CharField(max_length=256)
    movie_year=models.CharField(max_length=5)
    movie_country=models.CharField(max_length=32)
    movie_type=models.CharField(max_length=256)
    movie_language=models.CharField(max_length=64)
    movie_date=models.CharField(max_length=64)
    movie_word=models.CharField(max_length=128)
    movie_size=models.CharField(max_length=32)
    movie_duration=models.CharField(max_length=32)
    movie_actors=models.CharField(max_length=2048)
    movie_direcor=models.CharField(max_length=256)
    movie_profile=models.CharField(max_length=1024)
    movie_swiftwriter=models.CharField(max_length=256)
    movie_capture=models.CharField(max_length=2048)
    movie_poster=models.CharField(max_length=256)
    movie_download_url=models.CharField(max_length=1000)
    movie_download_file=models.CharField(max_length=512)
    movie_index=models.IntegerField()
    movie_rating=models.FloatField()


    def __str__(self):
        return  self.movie_title