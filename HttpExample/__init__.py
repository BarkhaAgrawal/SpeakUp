import logging

import azure.functions as func
# Import helper script
from .query_classification import getLabel
import tempfile
from os import listdir
from pathlib import Path

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    tempFilePath = tempfile.gettempdir()
    logging.info(tempFilePath)
    fp = tempfile.NamedTemporaryFile()
    fp.write(b'Hello world!')
    filesDirListInTemp = listdir(tempFilePath)
    logging.info(tempFilePath + "/L_test.jpg")
    file_path= str(Path.home()) + "/data/test2.json"
    print(file_path)
    results = getLabel(tempFilePath + "/L_test.jpg")

    logging.info(results)
    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
            results = getLabel("./Test Images/L_test.jpg")
            logging.info(results)
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(results)
    else:
        return func.HttpResponse(
             results,
             status_code=200
        )
