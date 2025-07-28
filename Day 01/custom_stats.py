# Custom mean, median, mode
def calculate_mean(data):
    return sum(data) / len(data)

def calculate_median(data):
    data.sort()
    n = len(data)
    mid = n // 2
    if n % 2 == 0:
        return (data[mid - 1] + data[mid]) / 2
    return data[mid]

def calculate_mode(data):
    freq = {}
    for num in data:
        freq[num] = freq.get(num, 0) + 1
    max_freq = max(freq.values())
    return [k for k, v in freq.items() if v == max_freq]

dataset = [1, 2, 2, 3, 4, 5, 5, 5, 6]

print("Mean:", calculate_mean(dataset))
print("Median:", calculate_median(dataset))
print("Mode:", calculate_mode(dataset))
