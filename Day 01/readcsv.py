# Simulate what pandas does
with open("data.csv", "r") as f:
    lines = f.readlines()
    for line in lines[1:]:
        name, age = line.strip().split(",")
        print(f"Name: {name}, Age: {age}")
