from core.handlers import FileHandler

if __name__ == "__main__":
    handler = FileHandler.create()

    if handler.watch:
        handler.start_watching()
    else:
        handler.start_processing()
