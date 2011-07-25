from django.conf.urls.defaults import *
from django.views.generic.simple import direct_to_template

urlpatterns = patterns('',
               (r'^add/$', 'DataSetManager.views.add'),
               (r'^submit/$', 'DataSetManager.views.submit'),
               (r'^list/$', 'DataSetManager.views.list'),                  
               )

