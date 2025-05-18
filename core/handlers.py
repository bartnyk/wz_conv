import argparse
import logging
import os
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from core import cfg
from core.reader import PdfFileProcessor


class Watcher(FileSystemEventHandler):
    def __init__(self, output: str = None) -> None:
        super().__init__()
        self.output = output

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            try:
                logging.info(f"New file found: {event.src_path}")
                FileHandler(path=event.src_path, output=self.output).start_processing()
            except Exception as e:
                logging.error(f"Error: {event.src_path}: {e}")


class FileHandler:
    def __init__(self, path: str, output: str, watch: bool = False) -> None:
        self.path = path
        self.output = output
        self.watch = watch

    @staticmethod
    def _parse_args():
        parser = argparse.ArgumentParser(
            description="Process pdf's and split them into separated WZ's."
        )
        parser.add_argument(
            "--path",
            type=str,
            help="Path to the directory with pdf files or some single pdf file.",
            required=True,
        )
        parser.add_argument("--output", type=str, help="Path to the output directory.")
        parser.add_argument(
            "--watch", action="store_true", help="Watch the directory for new files."
        )

        return parser.parse_args()

    @classmethod
    def create(cls) -> "FileHandler":
        kwargs = cls._parse_args()
        if not kwargs.output:
            if os.path.isfile(kwargs.path):
                dir_path = os.path.dirname(kwargs.path)
            else:
                dir_path = kwargs.path
            kwargs.output = os.path.join(dir_path, "output")
            os.makedirs(kwargs.output, exist_ok=True)
        return cls(**kwargs.__dict__)

    @classmethod
    def loop_files(cls, dir_path: str) -> list[str]:
        pdf_file_paths = []

        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if file_name.endswith(".pdf") and os.path.isfile(file_path):
                pdf_file_paths.append(file_path)

        return pdf_file_paths

    def start_processing(self):
        if os.path.isfile(self.path) and self.path.lower().endswith(".pdf"):
            files = [PdfFileProcessor(self.path, output_dir=self.output)]
            print("Processing single PDF file: ", self.path)
        elif os.path.isdir(self.path):
            pdf_file_paths = self.loop_files(self.path)

            files = [
                PdfFileProcessor(file_path, output_dir=self.output)
                for file_path in pdf_file_paths
            ]
            print(
                f"Processing PDF files (total: {len(files)}) from directory: {self.path}."
            )
        else:
            raise AttributeError(f"{self.path} is neither a directory nor a PDF file.")

        for file in files:
            file.process_pdf()
            file.save_all()

    def start_watching(
        self,
    ) -> None:
        event_handler = Watcher(output=self.output)
        observer = Observer()
        observer.schedule(event_handler, path=self.path, recursive=False)
        observer.start()

        try:
            while True:
                time.sleep(cfg.WATCHER_COOLDOWN)
                logging.info("Monitoring folder for new files...")
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
