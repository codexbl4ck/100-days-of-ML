import math

# Sample data
x = [2, 4, 6, 8, 10]
y = [1, 3, 5, 7, 9]

# ✅ Mean
def mean(data):
    return sum(data) / len(data)

# ✅ Median
def median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        return sorted_data[mid]

# ✅ Mode
def mode(data):
    freq = {}
    for num in data:
        freq[num] = freq.get(num, 0) + 1
    max_freq = max(freq.values())
    modes = [key for key, val in freq.items() if val == max_freq]
    return modes

# ✅ Variance
def variance(data):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)

# ✅ Standard Deviation
def standard_deviation(data):
    return math.sqrt(variance(data))

# ✅ Covariance
def covariance(x, y):
    mean_x = mean(x)
    mean_y = mean(y)
    covar = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / len(x)
    return covar

# ✅ Correlation
def correlation(x, y):
    return covariance(x, y) / (standard_deviation(x) * standard_deviation(y))

# 🔎 Testing the functions
print("Mean of x:", mean(x))
print("Median of x:", median(x))
print("Mode of x:", mode(x))
print("Variance of x:", variance(x))
print("Standard Deviation of x:", standard_deviation(x))
print("Covariance of x and y:", covariance(x, y))
print("Correlation of x and y:", correlation(x, y))
