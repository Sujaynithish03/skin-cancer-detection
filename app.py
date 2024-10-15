from flask import Flask, render_template, redirect, url_for, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    # You can handle form data here, such as saving user info to a database
    # For demonstration, we'll just redirect to the Streamlit app
    return redirect("http://localhost:8501")  # Replace with your Streamlit app URL if different

if __name__ == '__main__':
    app.run(debug=True)
