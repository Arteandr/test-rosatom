from django.db import models

# Create your models here.
class Review(models.Model):
    name = models.TextField()
    text = models.TextField()
    rating = models.IntegerField()
    sentiment = models.IntegerField()
    
    def __str__(self):
        return f"{self.name} - {self.text} - {self.rating}/10 - {'positive' if self.sentiment == 1 else 'negative'}"