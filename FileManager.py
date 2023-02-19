def read_file(file_name):
    with open(file_name, "rt") as file:
        return file.read()

if __name__ == "__main__":
    with open("final_report.txt", "rt") as file:
        file_line = file.read()
        print(len(file_line))

