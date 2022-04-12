from flask import Flask, request
from flask import render_template
from finder_file_on_pc import *

app = Flask(__name__)


@app.route('/')
def start_page():
    return render_template('index.html')


@app.route('/recogniting_photo', methods=['POST', 'GET'])
def recogniting_photo():
    if request.method == 'POST':
        photo_for_finder_func = request.form['input_photo_from_pc']
        print(photo_for_finder_func)
        way_to_photo = find_file_in_all_drives(photo_for_finder_func)
        print(way_to_photo)
        return render_template("index.html")


if __name__ == '__main__':
    app.run(port=8080, host='127.0.0.1')