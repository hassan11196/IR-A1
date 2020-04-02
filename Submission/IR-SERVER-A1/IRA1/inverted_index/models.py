from django.db import models

# Create your models here.
from picklefield.fields import PickledObjectField

class InvertedIndexModel(models.Model):
   
    data = PickledObjectField()
    status = models.BooleanField(default=False, name = 'status')
    id = models.DateTimeField(auto_now_add=True, primary_key = True)