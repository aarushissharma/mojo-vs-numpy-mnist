import matplotlib.pyplot as plt


labels = ['Mojo\nNaive', 'Mojo\nSIMD', 'Mojo SIMD\n+Parallel', 'Mojo SIMD\n+Parallel\nFloat32', 'NumPy']
times = [803, 138, 30, 13, 6.34]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, times, color=["#FFCAB1", '#ECDCB0', '#C1D7AE', '#8CC084', '#968E85'])
plt.ylabel('avg forward pass time (ms)')
plt.title('mojo & numpy: forward pass benchmark')
plt.yscale('log')

for bar, time in zip(bars, times):
  plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, f'{time}ms', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/benchmark.png', dpi=150)
plt.show()