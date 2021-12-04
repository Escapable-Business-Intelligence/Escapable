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

    def to_dict(self):
        return {
            "cafe_index" : self.cafe_index,
            "cafe_name" : self.cafe_name,
            "cafe_info" : self.cafe_info,
            "cafe_number" : self.cafe_number,
            "cafe_address" : self.cafe_address
        }

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
    
    def to_dict(self):
        return {
            "review_index" : self.review_index,
            "cafe_name" : self.cafe_name,
            "thema_name" : self.thema_name,
            "user_left_time" : self.user_left_time,
            "user_difficulty" : self.user_difficulty,
            "user_escape" : self.user_escape,
            "user_rate" : self.user_rate,
            "user_nickname" : self.user_nickname
        }


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

    def to_dict(self):
        return {
            "thema_index" : self.thema_index,
            "cafe_name" : self.cafe_name,
            "thema_name" : self.thema_name,
            "thema_limit_time" : self.thema_limit_time,
            "thema_genre" : self.thema_genre,
            "thema_level" : self.thema_level,
            "thema_activity" : self.thema_activity,
            "thema_number_of_people" : self.thema_number_of_people,
            "thema_info" : self.thema_info,
            "thema_picture" : self.thema_picture
        }
