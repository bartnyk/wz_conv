import argparse
import os

from reader import PdfFileProcessor


class PDFProcessor:
    def __init__(self, path: str, output: str) -> None:
        self.path = path
        self.output = output

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

        return parser.parse_args()

    @classmethod
    def create(cls) -> "PDFProcessor":
        kwargs = cls._parse_args()

        return cls(**kwargs.__dict__)

    @classmethod
    def loop_files(cls, dir_path: str) -> list[str]:
        pdf_file_paths = []

        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if file_name.endswith(".pdf") and os.path.isfile(file_path):
                pdf_file_paths.append(file_path)

        return pdf_file_paths

    def run(self):
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


if __name__ == "__main__":
    processor = PDFProcessor.create()
    processor.run()
