import time
time.sleep(120)

def write_hello_world(filename):
    with open(filename, 'w') as f:
        f.write("Hello, World!")
        
write_hello_world("hello.txt")