from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from neural_network import prediction, training
import argparse
import os

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = '\\temp'
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'predictcovid'

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-t', '--train', help='Neural network training', nargs='*', default=None)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    _image = request.files['image']
    _path = os.path.join(app.config['UPLOAD_FOLDER'], _image.filename)
    _path = os.path.dirname(__file__) + _path
    _image.save(_path)
    _is_covid, _percentage, _error = prediction(_path)
    os.remove(_path)

    if _error is None:
        return jsonify({'is_covid' : _is_covid, 'percentage': _percentage}), 200
    else:
        return jsonify({'error': _error}), 500

if __name__ == '__main__':
    _dirname = os.path.dirname(__file__)
    _args = vars(argument_parser.parse_args())

    if _args['train'] is not None:
        if not os.path.isdir(_dirname + '\\models'):
            os.mkdir(_dirname + '\\models')

        training()
    
    if not os.path.isdir(_dirname + '\\temp'):
            os.mkdir(_dirname + '\\temp')

    app.run(debug=False, port=80, threaded=False)