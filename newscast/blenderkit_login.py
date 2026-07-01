"""BlenderKit OAuth login (same flow the addon uses) -> stores a Bearer access_token for the headless
client. The token has the subscription's real download quota and is NOT subject to the session anti-bot.
  python3 blenderkit_login.py serve      # start callback listener + print the authorize URL to open
  python3 blenderkit_login.py exchange <code>   # manual: paste the code from the redirect URL
Token -> ~/.config/studio/blenderkit.token  (used automatically by blenderkit.py via BLENDERKIT_API_KEY)
"""
import os, sys, json, urllib.request, urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

SERVER = "https://www.blendkit.com"
CLIENT_ID = "IdFRwa3SGA8eMpzhRVFMg5Ts8sPK93xBjif93x0F"
PORT = 62485
REDIRECT = f"http://localhost:{PORT}/consumer/exchange/"
TOKEN_FILE = os.path.expanduser("~/.config/studio/blenderkit.token")
os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)

def authorize_url():
    return SERVER + "/o/authorize/?" + urllib.parse.urlencode({
        "client_id": CLIENT_ID, "redirect_uri": REDIRECT, "response_type": "code", "state": "studio"})

def exchange(code):
    data = urllib.parse.urlencode({
        "grant_type": "authorization_code", "code": code, "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT}).encode()
    req = urllib.request.Request(SERVER + "/o/token/", data=data,
                                 headers={"Content-Type": "application/x-www-form-urlencoded",
                                          "User-Agent": "BlenderKit-Client/1.0", "Accept": "application/json"})
    tok = json.load(urllib.request.urlopen(req, timeout=30))
    at = tok.get("access_token")
    if not at:
        raise SystemExit(f"no access_token in response: {list(tok)}")
    json.dump(tok, open(TOKEN_FILE, "w")); os.chmod(TOKEN_FILE, 0o600)
    print("BLENDERKIT_TOKEN_OK -> stored (expires_in", tok.get("expires_in"), "s)")
    return at

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        q = urllib.parse.urlparse(self.path)
        code = urllib.parse.parse_qs(q.query).get("code", [None])[0]
        msg = b"Login OK - token captured. You can close this tab."
        if code:
            try:
                exchange(code); open("/tmp/bk_login_done", "w").write("ok")
            except Exception as e:
                msg = f"exchange failed: {e}".encode()
        else:
            msg = b"no code in callback"
        self.send_response(200); self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(msg))); self.end_headers(); self.wfile.write(msg)

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "exchange":
        exchange(sys.argv[2])
    else:
        print("OPEN THIS URL (you're logged into blendkit.com -> click Authorize):\n")
        print("  " + authorize_url() + "\n")
        print(f"Listening for the callback on :{PORT} ... (if the redirect page won't load, copy the "
              f"'code=' value from the address bar and run: blenderkit_login.py exchange <code>)")
        HTTPServer(("0.0.0.0", PORT), H).serve_forever()
