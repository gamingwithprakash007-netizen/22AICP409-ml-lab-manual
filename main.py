# Fallback WSGI/ASGI entrypoint for Vercel
import os

def app(environ, start_response):
    status = "200 OK"
    headers = [("Content-Type", "text/html; charset=utf-8")]
    start_response(status, headers)
    
    html_file = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_file):
        with open(html_file, "rb") as f:
            return [f.read()]
    return [b"<h1>22AICP409 Machine Learning Laboratory Portal</h1>"]

# ASGI entrypoint
async def asgi_app(scope, receive, send):
    if scope["type"] == "http":
        html_file = os.path.join(os.path.dirname(__file__), "index.html")
        body = b"<h1>22AICP409 Machine Learning Laboratory Portal</h1>"
        if os.path.exists(html_file):
            with open(html_file, "rb") as f:
                body = f.read()
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [[b"content-type", b"text/html; charset=utf-8"]],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })
