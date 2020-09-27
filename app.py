from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def function():
    shop = "barbie doll"
    return render_template("indext.html", shoe=shop)
@app.route('/tuna', methods = ['GET', 'POST'])
def hello():
    return ("failure is love")


if __name__ == '__main__':
    app.debug = True
    app.run()
