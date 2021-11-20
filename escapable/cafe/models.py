from django.db import models


class Cafe(models.Model):
    cafe_index = models.AutoField(primary_key=True)
    cafe_name = models.CharField(max_length=100, blank=True, null=True)
    cafe_info = models.CharField(max_length=1000, blank=True, null=True)
    cafe_number = models.CharField(max_length=300, blank=True, null=True)
    cafe_address = models.CharField(max_length=1000, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cafe'


class Review(models.Model):
    review_index = models.AutoField(primary_key=True)
    cafe_name = models.CharField(max_length=100, blank=True, null=True)
    thema_name = models.CharField(max_length=100, blank=True, null=True)
    user_left_time = models.IntegerField(blank=True, null=True)
    user_difficulty = models.CharField(max_length=6, blank=True, null=True)
    user_escape = models.FloatField(blank=True, null=True)
    user_rate = models.FloatField(blank=True, null=True)
    user_nickname = models.CharField(max_length=45, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'review'


class Thema(models.Model):
    thema_index = models.AutoField(primary_key=True)
    cafe_name = models.CharField(max_length=100, blank=True, null=True)
    thema_name = models.CharField(max_length=100, blank=True, null=True)
    thema_limit_time = models.IntegerField(blank=True, null=True)
    thema_genre = models.CharField(max_length=60, blank=True, null=True)
    thema_level = models.IntegerField(blank=True, null=True)
    thema_activity = models.CharField(max_length=2, blank=True, null=True)
    thema_number_of_people = models.CharField(max_length=60, blank=True, null=True)
    thema_info = models.CharField(max_length=1000, blank=True, null=True)
    thema_picture = models.CharField(max_length=1000, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'thema'