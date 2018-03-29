from flask import Flask, render_template, request
from compute import prediction
from wtforms import Form, IntegerField, validators

class InputForm(Form):
    uid = IntegerField(
        label='User Id', default=1,
        validators=[validators.InputRequired()])
    mid = IntegerField(
        label='Movie Id', default=31,
        validators=[validators.InputRequired()])

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        result = prediction(form.uid.data, form.mid.data)
    else:
        result = None

    return render_template('view.html', form=form, result=result)

if __name__ == '__main__':
    app.run(debug=True)