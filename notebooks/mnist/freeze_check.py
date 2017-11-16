import os
import tensorflow as tf
from google.protobuf import json_format

g = tf.GraphDef()
g.ParseFromString(open(os.getcwd() + '/session-frozen.pb', 'rb').read())

json_string = json_format.MessageToJson(g)

file = open(os.getcwd() + '/graph.json', 'w')
file.write(json_string);
# print(json_string)

# print('Input nodes')
# in_nodes = [n for n in g.node if n.name.find('reshape') != -1]
# print(in_nodes)
