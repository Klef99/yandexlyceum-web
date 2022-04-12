from flask import Flask, request, render_template
from run import get_predict

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
@app.route('/index.html', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        f = request.files['file']
        print(f.read())
        return "Форма отправлена"


@app.route('/sign_in_page.html')
def sing_in_page():
    return render_template('sign_in_page.html')


@app.route('/sign_up_page.html')
def sing_up_page():
    return render_template('sign_up_page.html')


@app.route('/my_recognitions_page.html')
def my_recognitions_page():
    return render_template('my_recognitions_page.html')


if __name__ == '__main__':
    app.run(port=8080, host='127.0.0.1')
