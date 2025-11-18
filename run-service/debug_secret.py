"""
Debug script to check Modal secret content
"""
import modal

app = modal.App("debug-secret")

@app.function(secrets=[modal.Secret.from_name("youtube-cookies")])
def check_secret():
    import os

    cookies = os.getenv("YOUTUBE_COOKIES")
    if not cookies:
        print("ERROR: No YOUTUBE_COOKIES found in environment!")
        return

    print(f"Cookie content length: {len(cookies)}")
    print(f"Number of newlines: {cookies.count(chr(10))}")
    print(f"Number of lines: {len(cookies.splitlines())}")
    print(f"\nFirst 500 characters:")
    print(repr(cookies[:500]))

    # Check for actual cookie lines
    lines = cookies.splitlines()
    cookie_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    print(f"\nNon-comment lines: {len(cookie_lines)}")
    if cookie_lines:
        print(f"First cookie line: {repr(cookie_lines[0])}")

@app.local_entrypoint()
def main():
    check_secret.remote()
