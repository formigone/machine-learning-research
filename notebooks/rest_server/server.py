import argparse
import sys
import os
import time
import json
import numpy as np
import urllib.parse as urllib
from PIL import Image
from http.server import BaseHTTPRequestHandler, HTTPServer

print('Running...')
host = "localhost"
port = 6006


class Server(BaseHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse_qs(urllib.urlparse(self.path).query)
        data = {}

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        try:
            if 'file' in query:
                file = '/Users/rsilveira/Desktop/' + str(query['file'][0])
                data['file'] = file
                img = Image.open(file)
                img = img.resize((28, 28))
                img = np.array(img) * 255
            elif 'data' in query:
                data['data'] = query.data
                img = np.array(json.loads(query['data']))
                img = img.resize((28, 28))

            data['classes'] = [n for n in range(10)]
            data['q'] = query
            print(data)
            self.wfile.write(bytes(json.dumps(data), "utf-8"))
        except Exception as e:
            self.send_response(500)
            self.wfile.write(bytes(json.dumps({'error': str(e)}), "utf-8"))

    def do_POST(self):
        data = {}

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        post_data = json.loads(post_data)
        pixels = list()

        if 'pixels' in post_data:
            pixels = post_data['pixels']
            print('PIXELS:' + str(len(post_data['pixels'])))

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        try:
            data['pixels'] = pixels
            print(data)
            self.wfile.write(bytes(json.dumps(data), "utf-8"))
        except Exception as e:
            self.send_response(500)
            self.wfile.write(bytes(json.dumps({'error': str(e)}), "utf-8"))


server = HTTPServer((host, port), Server)
print(time.asctime(), "Server Starts - %s:%s" % (host, port))

try:
    server.serve_forever()
except KeyboardInterrupt:
    pass

server.server_close()
print(time.asctime(), "Server Stops - %s:%s" % (host, port))
