from flask import Flask,request
from flask_cors import CORS
import helperAPI as hp
import json

app = Flask(__name__)
CORS(app)

@app.route('/getData', methods=['GET'])
def home():
    link = request.args.get('url')
    print(link)
    return hp.main_method(link)

if __name__ == "__main__":
    app.run(host='0.0.0.0')