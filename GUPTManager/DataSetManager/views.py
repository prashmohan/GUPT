# Create your views here.
from DataSetManager.forms import DataSetRegistrationForm
import django.forms as forms
from django.forms import widgets
from DataSetManager.models import DataSet
from django.http import HttpResponse, HttpResponseRedirect, HttpResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_protect
from django.shortcuts import render

@csrf_protect
@login_required
def add(request):
    f = DataSetRegistrationForm()
    c = {
        'form' : f,
        'user' : request.user,
        }
    return render(request, 'datasetmanager/add_dataset.html', c)

@csrf_protect
@login_required
def submit(request):
    f = DataSetRegistrationForm(request.POST, request.FILES)
    f.save(request.user)
    c = {
        'user': request.user
        }
    return render(request, 'datasetmanager/dataset_added.html', c)

@login_required
def list(request):
    c = {
        'user' : request.user,
        'datasets' : DataSet.objects.all()
        }
    return render(request, 'datasetmanager/list_datasets.html', c)
