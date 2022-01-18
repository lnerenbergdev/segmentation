from flask import Flask, render_template, request
import inference_segmentation as ifs
import json

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/bgr", methods=["POST"])
def remove_background():
    if request.method == "POST":
        data = request.get_json()
        
        image_path = data['image_path']
        print(image_path)
        print(ifs.predict(image_path))
        return "success"


if __name__ == "__main__":
    app.run(port=5000, debug=True)
