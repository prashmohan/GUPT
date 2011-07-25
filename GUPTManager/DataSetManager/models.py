from django.db import models
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from GUPTManager.settings import DATASET_ROOT

dataset_storage = FileSystemStorage(location=DATASET_ROOT, base_url='/datasets')
parser_storage = FileSystemStorage(location=DATASET_ROOT, base_url='/parser')

class DataSet(models.Model):
    owner = models.ForeignKey(User, editable=False)
    name = models.CharField(max_length=64, unique=True)
    description = models.TextField()
    dataset = models.FileField(upload_to='datasets', storage=dataset_storage)
    ds_mimetype = models.CharField(max_length=64, editable=False)
    parser = models.FileField(upload_to='parser', storage=parser_storage)
    parser_mimetype = models.CharField(max_length=64, editable=False)
    enabled = models.BooleanField(default=True)
    created = models.DateTimeField(auto_now_add=True, editable=False)
    updated = models.DateTimeField(auto_now=True, auto_now_add=True, editable=False)

    def __unicode__(self):
        return self.name
