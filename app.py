from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("models/model.pkl")

@app.route("/", methods=["GET","POST"])
def home():

    prediction = None

    if request.method == "POST":

        study_hours = float(
            request.form["study_hours"]
        )

        attendance = float(
            request.form["attendance"]
        )

        previous_marks = float(
            request.form["previous_marks"]
        )

        assignments = float(
            request.form["assignments"]
        )

        result = model.predict([[
            study_hours,
            attendance,
            previous_marks,
            assignments
        ]])

        prediction = round(result[0],2)

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)