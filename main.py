import os

from flask import Flask, request, render_template, flash, url_for, send_file, session
from model import get_predict
from werkzeug.utils import secure_filename, redirect
from forms import LoginForm, RegForm
from data import db_session
from data.users import User
from data.scans import Scans
from funcs import pass_check, pass_to_hash

UPLOAD_FOLDER = 'scans'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = '379cc9d0797b7d3a445eae49288768c6'
app.config['SECRET_KEY'] = '379cc9d0797b7d3a445eae49288768c6'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_user_from_sessions(ses):
    res = ses['user']
    user = db_sess.query(User).filter(User.hashedpass == res['hash'], User.salt == res['salt']).first()
    return user


@app.route('/logaut.html')
def logaut():
    session.pop('user')
    return redirect('/')


@app.route('/', methods=['POST', 'GET'])
@app.route('/index.html', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        if 'user' not in session:
            return render_template('index.html')
        else:
            user = get_user_from_sessions(session)
            return render_template('main_page_for_authorizated_user.html', name=user.name)
    elif request.method == 'POST':
        # check if the post request has the file part
        if 'input_photo_from_pc' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['input_photo_from_pc']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file', filename=filename))
            get_predict(filename)
            if 'user' in session:
                user = get_user_from_sessions(session)
                os.rename(f'scans/{filename}', f'scans/id{user.id}_{filename}')
                scan = Scans()
                scan.filename = filename
                scan.true_filename = f'scans/id{user.id}_{filename}'
                scan.user = user
                user.scans.append(scan)
                db_sess.commit()
                return send_file(scan.true_filename,
                                 download_name=f'rec_{filename}', as_attachment=True)
            return send_file(f'scans/{filename}',
                             download_name=f'rec_{filename}', as_attachment=True)


@app.route('/sign_in_page.html', methods=['POST', 'GET'])
def sing_in_page():
    form = LoginForm()
    if form.validate_on_submit():
        user = db_sess.query(User).filter(User.name == form.username.data).first()
        if user is not None and pass_check(form.password.data, user.hashedpass, user.salt):
            session['user'] = {'hash': user.hashedpass, 'salt': user.salt}
            return redirect('/index.html')
        else:
            flash('Не правильный логин или пароль', 'error')
    return render_template('sign_in_page.html', title='Авторизация', form=form)


@app.route('/sign_up_page.html', methods=['POST', 'GET'])
def sing_up_page():
    form = RegForm()
    if form.validate_on_submit():
        user = db_sess.query(User).filter(User.name == form.username.data).all()
        if len(user) == 0:
            user = User()
            user.name = form.username.data
            user.hashedpass, user.salt = pass_to_hash(form.password.data)
            db_sess.add(user)
            db_sess.commit()
            session['user'] = {'hash': user.hashedpass, 'salt': user.salt}
            return redirect('/')
        else:
            flash('Такой логин уже есть', 'error')
    return render_template('sign_up_page.html', title='Регистрация', form=form)


@app.route('/my_recognitions_page.html')
def my_recognitions_page():
    user = get_user_from_sessions(session)
    return render_template('my_recognitions_page.html', name=user.name, scans=user.scans)


@app.route('/download/<idx>')
def download_file(idx):
    scan = db_sess.query(Scans).filter(Scans.id == idx).first()
    return send_file(scan.true_filename,
                     download_name=scan.filename, as_attachment=True)


if __name__ == '__main__':
    db_session.global_init('db/base.sqlite')
    db_sess = db_session.create_session()
    app.run(port=8080, host='127.0.0.1')
