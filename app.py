from flask import Flask, render_template, request, session, redirect, url_for
from query_answering import get_answer  # Your answer logic
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        query = request.form["query"]
        answer = get_answer(query)

        session["chat_history"].append({"user": query, "bot": answer})
        session.modified = True  # Tell Flask to update session

    return render_template("index.html", chat_history=session.get("chat_history", []))

@app.route("/reset", methods=["GET"])
def reset():
    session.pop("chat_history", None)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
