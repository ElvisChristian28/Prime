# from flask import Flask, render_template, url_for, request

# app = Flask(__name__)

# @app.route("/")
# def hello():
#     return render_template("index.html")

# @app.route("/login")
# def login():
#     return render_template("login.html")

# @app.route("/handle-login", methods=["GET", "POST"])
# def handle_login():
#     if request.method == "POST":
#         username = request.form["username"]
#         password = request.form["password"]
#         return f"Welcome {username}!"
#     elif request.method == "GET" :
#         return "GET Login Successful"

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, jsonify
from models import db, Device

app = Flask(__name__)
# Use SQLite for development; swap to PostgreSQL for production
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:elvis@2803@localhost/prime'

from sqlalchemy.engine import URL

app.config["SQLALCHEMY_DATABASE_URI"] = URL.create(
    drivername="mysql+pymysql",
    username="root",
    password="elvis@2803",
    host="localhost",
    database="prime"
)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Create tables before first request
with app.app_context():
    db.create_all()

@app.route('/')
def dashboard():
    """Serves the main HTML dashboard."""
    return render_template('dashboard.html')

@app.route('/api/devices')
def api_devices():
    """Returns the latest SQL data as JSON for the frontend to render."""
    devices = Device.query.all()
    
    # Calculate some quick summary stats using Python (or use SQL aggregations)
    offline_count = sum(1 for d in devices if d.status == 'error')
    
    return jsonify({
        "total_devices": len(devices),
        "offline_count": offline_count,
        "devices": [d.to_dict() for d in devices]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)