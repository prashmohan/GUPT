from django.forms import ModelForm
from django import forms
from django.contrib.auth.decorators import login_required
import re
from DataSetManager.models import DataSet

class DataSetRegistrationForm(ModelForm):
    class Meta:
        model = DataSet
    # user = 
    # name = forms.CharField(max_length=64, required=True)
    # description = forms.CharField(widget=forms.Textarea)
    # dataset = forms.FileField()
    # parser = forms.FileField()
    # enabled = forms.BooleanField()

    def __init__(self, *args, **kwargs):
        super(DataSetRegistrationForm, self).__init__(*args, **kwargs)
        self.bound_object = None
        if len(args) >= 1:
            print 'a'
            self.bound_object = DataSet()
            print 'b'
        self.is_updating = False
        if self.bound_object:
            self.is_updating = True

    def save(self, user):
        if not self.is_updating:
            self.bound_object = DataSet()
        self.bound_object.owner = user

        self.bound_object.name = self.cleaned_data['name']
        self.bound_object.description = self.cleaned_data['description']
        
        # Retrieve the UploadedFile object for the attached_file field.
        dataset = self.cleaned_data['dataset']
        parser = self.cleaned_data['parser']

        # Clean up the filename before storing it.
        dataset_name = re.sub(r'[^a-zA-Z0-9._]+', '-', dataset.name)
        parser_name = re.sub(r'[^a-zA-Z0-9._]+', '-', parser.name)
        
        # Save the file and its metadata.
        self.bound_object.dataset.save(dataset_name, dataset)
        self.bound_object.ds_mimetype = dataset.content_type

        self.bound_object.parser.save(parser_name, parser)
        self.bound_object.parser_mimetype = parser.content_type

        self.bound_object.enabled = self.cleaned_data['enabled']
        
        self.bound_object.save()

