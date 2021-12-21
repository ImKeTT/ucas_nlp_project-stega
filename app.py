from flask import Flask, render_template, request, jsonify
from main_func import main, extract

# 创建Flask对象app并初始化
app = Flask(__name__, template_folder='./templates', static_folder='./templates/static')


# @app.route("/")
# def root():
#     return render_template("index.html",encode='hahha',demo=main,demoDef=returnUpper)

@app.route("/")
def root():
    return render_template("index.html")


@app.route("/hide")
def hide_html():
    return render_template("hide.html")


@app.route("/extract")
def extract_html():
    return render_template("extract.html")


@app.route('/submitHide', methods=['POST'])
def submit_hide():
    return main(request.form['secret_message'], request.form['model_name'], request.form['mode'])


@app.route('/submitExtract', methods=['POST'])
def submit_extract():
    return extract(request.form['secret_message'], request.form['model_name'], request.form['mode'])


# 定义app在8080端口运行
app.run(port=8080)
