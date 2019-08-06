
from pymodm.base import MongoModel
from pymodm import connect
from pymodm.fields import CharField
import glob
import requests


class CeleryBatchJob(MongoModel):
    project_id = CharField()
    analysis_name = CharField()
    celery_log_path = CharField()
    file_path = CharField()
    user = CharField()
    progress = CharField()


class BatchJobs:

    def __init__(self, project_id, directory_path, celery_log_path, user):
        self.project_id = project_id
        self.celery_log_path = celery_log_path
        self.directory_path = directory_path
        self.username = user

    def find_python_files(self):
        if self.directory_path[-1] != "/":
            self.directory_path += "/"
        return glob.glob(self.directory_path + '*.py')

    def start_jobs(self, mongo_str='mongodb://trap-umbriel:27017/photon-batch-jobs', mongo_alias='photon-batch-jobs'):

        connect(mongo_str, alias=mongo_alias, connect=False)

        for python_file in self.find_python_files():
            new_job = CeleryBatchJob()
            new_job.project_id = self.project_id
            new_job.celery_log_path = self.celery_log_path
            new_job.analysis_name = python_file[::-3]
            new_job.file_path = python_file
            new_job.user = self.username
            new_job.progress = 'Registered'
            new_job.save()

            started_status = requests.get('http://trap-titania:8003/cancel/' + new_job._id)
            print("Sent " + new_job.analysis_name + " to titania: " + str(started_status))
            new_job.progress = started_status


