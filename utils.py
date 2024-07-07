import os


def get_video_endpoint():
    endpoint = os.getenv("VIDEO_ENDPOINT")
    if endpoint is None:
        print("No endpoint specified, trying the secret file.")
        try:
            with open(".video_endpoint", "r") as f:
                endpoint = f.read().strip()
                print("Found endpoint in secret file!")
        except FileNotFoundError:
            print("No endpoint specified in the secret file either, exiting")
            exit(1)
    return endpoint
