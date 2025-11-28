from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import re

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# ---------------------------------------------------
# DATABASE CONFIG
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ---------------------------------------------------
# USER MODEL
# ---------------------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

# ---------------------------------------------------
# LOAD TRAINED SERENE MODEL
# ---------------------------------------------------
MODEL_DIR = os.path.join(BASE_DIR, "serene_model")

print("📌 Loading trained Serene model from:", MODEL_DIR)

try:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("✅ Serene model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    tokenizer = None
    model = None


# ---------------------------------------------------
# HELPER — FORMAT PROMPT
# ---------------------------------------------------
def build_prompt(user_input):
    return f"User: {user_input}\nSerene:"


# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------
@app.route("/")
def home():
    if "user_id" in session:
        return render_template("index.html")
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        if User.query.filter_by(username=username).first():
            return "Username already exists!", 400

        hashed_pw = generate_password_hash(password)
        user = User(username=username, password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            return redirect(url_for("home"))

        return "Invalid username or password.", 401

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------
# CHATBOT ENDPOINT
# ---------------------------------------------------
@app.route("/get", methods=["POST"])
def get_bot_response():
    if "user_id" not in session:
        return "Please log in to chat.", 401

    user_message = request.form.get("msg", "").strip()
    if not user_message:
        return "Please type something 💬"

    # SAFETY — SUICIDE CHECK
    crisis_words = ["suicide", "kill myself", "end my life", "want to die", "hurt myself"]
    if any(word in user_message.lower() for word in crisis_words):
        return (
            "🌸 I'm really sorry you're feeling this way. "
            "If you're in immediate danger, please call emergency services (112 in India). "
            "You don’t have to face this alone — I’m here with you."
        )

    if model is None or tokenizer is None:
        return "⚠️ Serene model unavailable."

    # FORMAT INPUT FOR THE MODEL
    prompt = build_prompt(user_message)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=120,
            temperature=0.65,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    reply = tokenizer.decode(output[0], skip_special_tokens=True)

    # REMOVE "User:" part + keep only Serene’s reply
    if "Serene:" in reply:
        reply = reply.split("Serene:", 1)[1].strip()

    # CLEANUP
    reply = re.sub(r"\s+", " ", reply)

    # LIMIT LENGTH
    words = reply.split()
    if len(words) > 60:
        reply = " ".join(words[:60]) + "..."

    return reply


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
