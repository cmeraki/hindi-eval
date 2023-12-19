import logging
import sys

class DataPrepLogger:
    def __init__(self, name: str, filename: str='logs.log'):
        self.logger = logging.getLogger(name)

        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s %(name)s %(message)s'
        )

        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def get_logger(self):
        return self.logger
