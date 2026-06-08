import batchFile
import main
import os
from mistralai.client import Mistral
#Run file and job id will be printed into console


df = main.aufbereiten()

prompt = 'Ist folgende Aussage Hassrede? Antworte nur mit ja fuer Hassrede oder nein fuer nicht Hassrede: '
filename = f"batch_wrong_labels.jsonl"

batchFile.create_batch_file(df, prompt, filename)

batchFile.start_batch_file(filename)