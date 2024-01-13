import os

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import numpy as np

from nd_to_json import nd_to_json, json_to_nd
from chonky import create_mapping

from esn import ESN

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

import sys
import signal
signal.signal(signal.SIGTERM, lambda: sys.exit(0))

PORT = os.getenv("PORT", 80)
SIZE = int(os.getenv("SIZE", 1024))
DENSITY = float(os.getenv("DENSITY", 0.1))
SPECTRAL_RADIUS = float(os.getenv("RADIUS", 0.99))
BIAS = os.getenv("BIAS", "").lower() in (True, "true")
ACTIVATION = os.getenv("ACTIVATION", "tanh")
MODEL_PATH = os.getenv("MODEL_PATH", "models/esn")
NORM_RATE = os.getenv("NORM_RATE", None)
NORM_RATE = float(NORM_RATE) if NORM_RATE is not None else None

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
        activation=ACTIVATION,
        norm_rate=NORM_RATE
    )
    esn.save(MODEL_PATH)

app = Flask(__name__)
api = Api(app)

step_counter = 0
state_input = np.zeros_like(esn.state)

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
        global state_input
        r = request.get_json(force=True)
        r = json_to_nd(r)
        shape = r.shape
        shape = [int(x) for x in shape]
        #create mapping
        mapping = create_mapping(key, shape, esn.state.shape)
        #add values
        # esn.state[mapping] += r.flatten()
        state_input[mapping] += r.flatten()

class StepResource(Resource):
    def post(self):
        global step_counter, state_input
        #step esn
        esn.state += state_input
        esn.step()
        state_input = np.zeros_like(esn.state)
        #save every SAVE_STEPS steps
        step_counter += 1
        if step_counter >= SAVE_STEPS:
            esn.save(MODEL_PATH)
            step_counter = 0

api.add_resource(ReadResource, "/read/<key>/<path:shape>")
api.add_resource(WriteResource, "/write/<key>")
api.add_resource(StepResource, "/step")

app.run(host="0.0.0.0", port=PORT)
