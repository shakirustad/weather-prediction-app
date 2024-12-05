from flask import Flask

app = Flask(__name__)

# @app.route('/')
# def home():
#     return "Hello, Flask!"

@app.route('/about')
def about():
    return "This is the About Page"
from flask import render_template

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/user/<username>')
def user_profile(username):
    return f"Hello, {username}!"


from flask import request

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        name = request.form['name']
        return f"Submitted: {name}"
    return '''
        <form method="post">
            Name: <input type="text" name="name">
            <input type="submit">
        </form>
    '''
@app.route('/api/data')
def api_data():
    return {"message": "Hello, JSON!"}

if __name__ == '__main__':
    app.run(debug=True)

