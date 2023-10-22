from flask import Flask, render_template, request
import pickle
import gunicorn


app = Flask(__name__)

@app.route('/', methods=["get", "post"])
def predict():
    global iw, vw, fp
    message = ""
    if request.method == "POST":
        iw = request.form.get('iw')
        vw = request.form.get('vw')
        fp = request.form.get('fp')

        data_1 = [[float(iw), float(vw), float(fp)]]
        with open("LR_model.pkl", 'rb') as f:
            load_model = pickle.load(f)
        pred = load_model.predict(data_1)
        print(pred)
        message = f"Спрогнозированные значения размеров шва (ширина(Width) и глубина(Depth)): {pred[0]}"

    return render_template("index.html", message=message)


app.run(host='0.0.0.0', port=80)
