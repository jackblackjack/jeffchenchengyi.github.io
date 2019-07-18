---
interact_link: content/machine-learning/miscellaneous-topics/how-flask-app-works.ipynb
kernel_name: python3
has_widgets: false
title: 'How to start a Flask App?'
prev_page:
  url: /machine-learning/miscellaneous-topics/linear-algebra-review
  title: 'Linear Algebra Review'
next_page:
  url: /machine-learning/miscellaneous-topics/epi
  title: 'Elements of Programming Interviews (Python)'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# How does a Flask App work?

We will do a quick run through about how a flask app works



---
# Files you'll require

1. ## `your_app.py`
```python
from your_app_folder import app
app.run(host='0.0.0.0', port=3001, debug=True)
```

2. ## `your_app_folder` folder

    1. ### `templates` folder
        1. #### `index.html`
            - ```html
            <!doctype html>

            <html>
            <head>
                <title>Index Page</title>
            </head>

            <body>
                <h1>The index.html page</h1>
                {% for tuple in data_set %}
                    <p>{{tuple}}</p>
                {% end_for %}
            </body>
            </html>
            ```
        2. #### `another_page.html`
            - ```html
            <!doctype html>

            <html>
            <head>
                <title>Another Page</title>
            </head>

            <body>
                <h1>The another_page.html page</h1>
            </body>
            </html>
            ```

    2. ### `__init__.py`
        - ```python
        from flask import Flask

        app = Flask(__name__)

        from your_app_folder import routes
        ```

    3. ### `routes.py`
        - ```python
        from your_app_folder import app
        from flask import render_template
        from wrangling_scripts.wrangling import data_wrangling
        
        data = data_wrangling()

        @app.route('/')
        @app.route('/index')
        def index():
            return render_template('index.html', data_set=data)

        @app.route('/another_page')
        def another_page():
            return render_template('another_page.html')
        ```
        
3. ## `wrangling_scripts` folder
    1. ### `wrangling.py`
        - ```python
        # ----------------------
        # insert here imports
        # ----------------------
        
        def data_wrangling():
            # ----------------------
            # insert data wrangling code here
            # ----------------------
            return data
        ```



---
# `python your_app.py`

This calls `app.run(host='0.0.0.0', port=3001, debug=True)` which does 2 things:
1. Initializes `app`
    1. `__init__.py` is called and Flask app is created
    - Wrangle data in a file (aka Python module). In this case, the file is called wrangling.py. The wrangling.py has a function that returns the clean data.
    - Execute this function in routes.py to get the data in routes.py
    - Pass the data to the front-end (index.html file) using the render_template method.
    - Inside of index.html, you can access the data variable with the squiggly bracket syntax {{ }}
2. Connects the `app` you've initialized to port: `3001`

