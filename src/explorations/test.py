import time
from pathlib import Path

for i in range(100):
    with open("output.txt", "a", encoding="utf8") as file:
        file.write(f"hello {i}\n")
    time.sleep(1)