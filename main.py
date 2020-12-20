import os

from flask import Flask, request, jsonify
from flask_restful import Resource, Api

from nd_to_json import nd_to_json, json_to_nd
from chonky import create_mapping

from esn import ESN

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

#TODO: save/load state and weights

PORT = os.getenv("PORT", 80)
SIZE = int(os.getenv("SIZE", 1024))
DENSITY = float(os.getenv("DENSITY", 0.1))
SPECTRAL_RADIUS = float(os.getenv("RADIUS", 0.99))
BIAS = os.getenv("BIAS", "true").lower() in (True, "true")
ACTIVATION = os.getenv("ACTIVATION", "tanh")

esn = ESN(
    SIZE,
    density=DENSITY,
    spectral_radius=SPECTRAL_RADIUS,
    bias=BIAS,
    activation=ACTIVATION
)

app = Flask(__name__)
api = Api(app)

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
        esn.step()

api.add_resource(ReadResource, "/read/<key>/<path:shape>")
api.add_resource(WriteResource, "/write/<key>")
api.add_resource(StepResource, "/step")

app.run(host="0.0.0.0", port=PORT)
