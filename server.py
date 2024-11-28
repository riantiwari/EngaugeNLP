from flask_cors import CORS
from flask import Flask

# Create a Flask instance
app = Flask(__name__)
CORS(app)

# Define a route for the root URL
@app.route("/", methods=["GET"])
def hello_world():
    return {"message": "Hello, World!"}

# Run the application if this script is executed
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
