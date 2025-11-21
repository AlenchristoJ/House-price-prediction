import os
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model.pkl')
preprocessor_path = os.path.join(script_dir, 'preprocessor.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

DEFAULTS = {
    'area': 3000, 'bedrooms': 4, 'bathrooms': 3, 'stories': 2,
    'mainroad': 'yes', 'guestroom': 'no', 'basement': 'yes',
    'hotwaterheating': 'no', 'airconditioning': 'yes', 'parking': 2,
    'prefarea': 'yes', 'furnishingstatus': 'semi-furnished'
}

def parse_form(form):
    data = {}

    for field in ('area', 'bedrooms', 'bathrooms', 'stories', 'parking'):
        try:
            data[field] = float(form.get(field, DEFAULTS[field]))
        except:
            data[field] = DEFAULTS[field]

    for field in ('mainroad', 'guestroom', 'basement',
                  'hotwaterheating', 'airconditioning', 'prefarea'):
        val = form.get(field, DEFAULTS[field])
        data[field] = 'yes' if str(val).lower() in ('yes','y','1','true','on') else 'no'

    data['furnishingstatus'] = form.get('furnishingstatus', DEFAULTS['furnishingstatus'])

    return pd.DataFrame([data])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        df = parse_form(request.form)
    else:
        df = pd.DataFrame([DEFAULTS])

    try:
        encoded = preprocessor.transform(df)
        pred = model.predict(encoded)
        price_val = float(pred[0][0]) if hasattr(pred[0], "__len__") else float(pred[0])
    except Exception as e:
        print("Prediction error:", e)
        price_val = 0.0

    price = f"{price_val:,.2f}"
    inputs = df.to_dict(orient="records")[0]

    return render_template("index.html", price=price, inputs=inputs)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
