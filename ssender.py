import requests
import json
import sys
import argparse
from os import listdir
from os.path import isfile, join

'''
Python script to emulate the client sending a photo to the server
for OCR. Saves the response as a JSON file response.json.
'''
def main(mode): 
    name = '521/shannah_33.jpg'
    # sends a photo to the server to process
    if mode == 0:
        url = 'http://127.0.0.1:5000/' # send to ocr
    else:
        url = 'http://127.0.0.1:5000/bar' # send to bar
    image = open("./pictures/" + name, "rb")
    upload = {'img': image}
    response = requests.post(url, files = upload)
    response.close()
    # gets the response text as a JSON object
    jsonobj = response.json()
    # dumps it to a json file
    with open('./inference_results/' + name + '.json', 'w') as txt:
        json.dump(jsonobj, txt, indent=4)
    print('done')

def str2bool(v):
    return v.lower() in ("true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str2bool, default=False)
    args = parser.parse_args()
    main(args.mode)

