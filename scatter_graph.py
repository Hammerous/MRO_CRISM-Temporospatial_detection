import pandas as pd
import matplotlib.pyplot as plt

# 打开并读取csv文件
df = pd.read_csv('assessment.csv')

# 读取View Num、Area(km^2)、Time Range字段
view_num = df['View Num']
area = df['Area(km^2)']
time_range = df['Time Range']

# 创建散点图
plt.figure(figsize=(10, 10))

# View Num vs Area(km^2)
plt.subplot(3, 1, 1)
plt.scatter(view_num, area, s=3, edgecolor='grey', linewidths=0.5)  # 增加点的边界线
plt.xlabel('View Num')
plt.ylabel('Area(km^2)')
plt.title('View Num vs Area(km^2)')
plt.grid(True)  # 增加数轴格网

# View Num vs Time Range
plt.subplot(3, 1, 2)
plt.scatter(view_num, time_range, s=3, edgecolor='grey', linewidths=0.5)  # 增加点的边界线
plt.xlabel('View Num')
plt.ylabel('Time Range')
plt.title('View Num vs Time Span')
plt.grid(True)  # 增加数轴格网

# Area(km^2) vs Time Range
plt.subplot(3, 1, 3)
plt.scatter(area, time_range, s=3, edgecolor='grey', linewidths=0.5)  # 增加点的边界线
plt.xlabel('Area(km^2)')
plt.ylabel('Time Range')
plt.title('Area(km^2) vs Time Span')
plt.grid(True)  # 增加数轴格网

# 调整布局并保存图像，分辨率为300dpi
plt.tight_layout()
plt.savefig('scatter_plots.png', dpi=300)

print("散点图已保存为scatter_plots.png，分辨率为300dpi。")