import matplotlib.pyplot as plt
import numpy as np

# Dataset
labels = ['Polars', 'Pandas', 'NumPy']

# WRITE times (in seconds)
# write_min = [6.846136, 7.713581, 2.520703]
# write_avg = [7.037001, 9.59571, 2.619846]
# write_max = [7.594364, 13.39587, 2.71941]

# READ times (in seconds)
# read_min = [0.306834, 1.588548, 0.502786]
# read_avg = [0.332887, 1.662955, 0.735573]
# read_max = [0.417788, 1.917322, 1.231828]

# READ times (in seconds)
# min_time = [0.084407, 0.036635, 0.007594]
# avg_time = [0.086232, 0.040271, 0.007920]
# max_time = [0.091961, 0.047227, 0.008713]

# UPDATE times (in seconds)
min_time = [0.012232, 2.211707, 0.025181]
avg_time = [0.015680, 2.235329, 0.025907]
max_time = [0.042994, 2.266898, 0.027312]

# QUERY times (in seconds)
min_time = [0.0084407, 0.036635, 0.007594]
avg_time = [0.0086232, 0.040271, 0.00792]
max_time = [0.019861, 0.047227, 0.008713]

x = np.arange(len(labels))  # label locations
width = 0.25  # bar width

# Plotting WRITE performance
fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, min_time, width, label='Min Time (s)', color='green')
rects2 = ax.bar(x, avg_time, width, label='Avg Time (s)', color='blue')
rects3 = ax.bar(x + width, max_time, width, label='Max Time (s)', color='red')

ax.set_ylabel('Time (seconds)')
ax.set_title('Benchmark: QUERY Operation Performance')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Adding text labels
for rect in rects1 + rects2 + rects3:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()
