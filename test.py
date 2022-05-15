import matplotlib.pyplot as plt
import numpy as np

color1 = plt.cm.tab10(5.2)  # tab(10)括号中输入随机数，生成颜色
x = np.random.rand(10)  # 生成十个随机数
y = x + x ** 2 - 10  # 函数关系确定y的值
plt.scatter(x, y,
            c=np.array(color1).reshape(1, -1))
plt.show() # 显示图像
