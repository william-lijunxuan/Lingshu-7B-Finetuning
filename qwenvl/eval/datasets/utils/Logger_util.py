from datetime import datetime
import os
import sys


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

class Logger:
    def __init__(self, model_name:str,save_file_path: str = None):
        self.model_name = model_name
        self.save_file_path = save_file_path or os.getcwd()

        date_str = datetime.now().strftime("%Y_%m_%d")
        time_str = datetime.now().strftime("%Y%m%d_%H%M")
        log_dir = os.path.join("log", model_name,date_str)
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"{self.model_name}_{time_str}.log"
        self.log_path = os.path.join(log_dir, log_filename)

        self.log_file = open(self.log_path, "w", encoding="utf-8")

        sys.stdout = sys.stderr = Tee(sys.__stdout__, self.log_file)

    def close(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.log_file.close()