class Logger:
    def __init__(self, writepath):
        try:
            print(f"Opening output file at {writepath}.")
            self.write_file = open(writepath, "w")
        except Exception as e:
            print("Could not open output file.", e)
            exit()

    def log(self, msg):
        self.write_file.write(msg + "\n")

    def close(self):
        print("Closing output file...")
        self.write_file.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
