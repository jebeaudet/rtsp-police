import sqlite3
import datetime
import cv2
from collections import deque

logged_track_ids = set()


def init_db():
    conn = sqlite3.connect("events.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS events
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  track_id INTEGER,
                  direction TEXT,
                  filename TEXT)''')
    conn.commit()
    conn.close()


def log_event(track_id, direction, frame_buffer):
    if track_id in logged_track_ids:
        return
    logged_track_ids.add(track_id)
    filename = save_video_clip(list(frame_buffer), track_id)
    conn = sqlite3.connect("events.db")
    c = conn.cursor()
    c.execute('''INSERT INTO events (track_id, direction, filename)
                 VALUES (?, ?, ?)''',
              (track_id, direction, filename))
    conn.commit()
    conn.close()


def save_video_clip(buffer, track_id):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_clip = f"car_{track_id}_{timestamp}.mp4"
    height, width, _ = buffer[0].shape
    out = cv2.VideoWriter(video_clip, fourcc, 20.0, (width, height))

    for frame in buffer:
        out.write(frame)

    out.release()
    return video_clip
