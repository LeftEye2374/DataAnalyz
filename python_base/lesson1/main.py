import time

def decorator_timer(func):
    def wrapper():
        print("Начало исполнения")
        start = time.time()
        func()
        diff = time.time() - start
        print(diff)
    return wrapper

@decorator_timer
def load_File():
    print("Start loading file")
    time.sleep(2)

@decorator_timer
def load_file_fast():
    print("Start loading file")
    time.sleep(0.5)

load_File()
load_file_fast()



