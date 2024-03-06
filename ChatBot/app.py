from nltk.tokenize import word_tokenize

# Define a function to respond based on keywords
def respond_to_user(input_text):
    input_words = word_tokenize(input_text.lower())
    
    # Simple keyword matching for responses
    if "hello" in input_words:
        return "Hi there!"
    elif "bye" in input_words:
        return "Goodbye!"
    else:
        return "I'm not sure how to respond to that."
    
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # You'll create this HTML file for the interface

@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    return respond_to_user(user_text)

if __name__ == "__main__":
    app.run()
