from functools import wraps

class WebApp():

    def __init__(self):
        print("init")

    def login_required(self, f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "authenticated" not in session:
                return redirect(url_for('login', next=request.url))
            return f(*args, **kwargs)
        return decorated_function

    def login(self):
        print("asd")


    # register
        # createuser
            # hashpw
    # login
    # 


