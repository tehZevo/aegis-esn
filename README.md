# Aegis Echo State Network Node
Echo State Networks in Aegis

This node uses [nd-to-json](https://github.com/tehZevo/nd-to-json) to encode/decode nd arrays, and [Chonky](https://github.com/tehZevo/chonky) to create sparse input/output mappings.

## Environment
* `PORT` - Port to listen on
* `SIZE` - Number of neurons in the ESN; defaults to 1024
* `DENSITY` - Density of the weights in the ESN; defaults to 0.1
* `SPECTRAL_RADIUS` - Spectral radius of the ESN weights; defaults to 0.99
* `BIAS` - If true (default), will add a constant value "1" to the end of the ESN state
* `ACTIVATION` - Activation of the ESN; `tanh` and `sigmoid` are supported, can also be `none`

## Routes
* `/read/<key>/<dim0>/<dim1>/.../<dimN>` - Retrieve values with shape specified by the `dim`s.
  * `<key>` - string key to produce the Chonky mapping
  * `<dim>`s - nd array shape
* `/write/<key>` - Write (add) values to the ESN; will use the shape given by the with shape specified by the [passed nd array](https://github.com/tehZevo/nd-to-json).
  * `<key>` - string key to produce the Chonky mapping
* `/step` - Performs one step of the ESN (`x = activation(x * w)`)

## Notes
* The ESN node is a bit different than other nodes in that you *push* data to it, as opposed to it *pulling* data from other nodes
