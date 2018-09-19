
from flask import Flask, render_template

app = Flask(__name__)

from webapp import WebApp

webabb = WebApp()

@app.route('/')
def index():
    return "This is index"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000, debug=True)