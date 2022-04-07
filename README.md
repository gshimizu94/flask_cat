
# Flask cat

Using Flask to build a CNN-based Cat-or-Dog classification Application.

# Demo

https://cat-or-dog-1.herokuapp.com/

# Installation

Install with pip:
```bash
$ pip install -r requirements.txt
```

# Flask Application Structure
```
.
│  app.py
│  Aptfile
│  best_resnet.pth
│  Procfile
│  README.md
│  requirements.txt
│  
└─templates
        flask_api_index.html
        layout.html
        result.html
```

# Usage

Run flask for develop
```bash
$ python app.py
```
Swagger document page:
http://127.0.0.1:5000/

(Default port is 5000)

Run with gunicorn
```bash
$ gunicorn app:app
```