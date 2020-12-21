import os
import sys
import signal

from flask import Flask, request, jsonify
from flask_restful import Resource, Api

from nd_to_json import nd_to_json, json_to_nd
from chonky import create_mapping

from esn import ESN

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

signal.signal(signal.SIGTERM, lambda: sys.exit(0))

PORT = os.getenv("PORT", 80)
SIZE = int(os.getenv("SIZE", 1024))
DENSITY = float(os.getenv("DENSITY", 0.1))
SPECTRAL_RADIUS = float(os.getenv("RADIUS", 0.99))
BIAS = os.getenv("BIAS", "true").lower() in (True, "true")
ACTIVATION = os.getenv("ACTIVATION", "tanh")
MODEL_PATH = os.getenv("MODEL_PATH", "models/esn")
SAVE_STEPS = int(os.getenv("SAVE_STEPS", 10000))

esn = None
try:
    print("Loading", MODEL_PATH)
    esn = ESN.load(MODEL_PATH)
    print(MODEL_PATH, "loaded")
except FileNotFoundError as e:
    print(e)
    print('"{}" not found. Creating new model.'.format(MODEL_PATH))
    esn = ESN(
        SIZE,
        density=DENSITY,
        spectral_radius=SPECTRAL_RADIUS,
        bias=BIAS,
        activation=ACTIVATION
    )
    esn.save(MODEL_PATH)

app = Flask(__name__)
api = Api(app)

step_counter = 0

class ReadResource(Resource):
    def post(self, key, shape):
        #get shape from url path /dim0/dim1/dim2 etc
        shape = shape.split("/")
        shape = [int(x) for x in shape]
        #create mapping
        mapping = create_mapping(key, shape, esn.state.shape)
        #grab values
        x = esn.state[mapping]
        #convert to json-compatible object and return
        x = nd_to_json(x)
        return jsonify(x)

class WriteResource(Resource):
    def post(self, key):
        r = request.get_json(force=True)
        r = json_to_nd(r)
        shape = r.shape
        shape = [int(x) for x in shape]
        #create mapping
        mapping = create_mapping(key, shape, esn.state.shape)
        #add values
        esn.state[mapping] += r.flatten()

class StepResource(Resource):
    def post(self):
        global step_counter
        #step esn
        esn.step()
        #save every SAVE_STEPS steps
        step_counter += 1
        if step_counter >= SAVE_STEPS:
            esn.save(MODEL_PATH)
            step_counter = 0

api.add_resource(ReadResource, "/read/<key>/<path:shape>")
api.add_resource(WriteResource, "/write/<key>")
api.add_resource(StepResource, "/step")

app.run(host="0.0.0.0", port=PORT)
