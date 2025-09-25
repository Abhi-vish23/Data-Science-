from django.db import models

# Create your models here.
class History(models.Model):
    precipitation = models.FloatField(default=0.0)
    max_temp = models.FloatField(default=0.0)
    min_temp = models.FloatField(default=0.0)
    wind_speed = models.FloatField(default=0.0)
    res = models.CharField(max_length=100, default="Unknown")
