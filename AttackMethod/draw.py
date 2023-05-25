# an example for image drawing
import numpy as np
import matplotlib.pyplot as plt


table_data = np.random.randint(0, 100, size=(5, 3))


def menu_click(menu_index):
    plt.clf()  

    ax0 = plt.subplot(3, 1, 1)
    ax0.axis('off')
    ax0.table(cellText=[menu_names], loc='center')


    ax1 = plt.subplot(3, 3, 4, polar=True)
    categories = ['A', 'B', 'C', 'D', 'E']
    values = np.random.randint(0, 100, size=len(categories)).astype(float)  # 将数据类型转换为浮点数
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += angles[:1]
    ax1.plot(angles, values, 'o-', linewidth=2)
    ax1.fill(angles, values, alpha=0.25)
    ax1.set_xticks(angles)
    ax1.set_xticklabels(categories)

 
    ax2 = plt.subplot(3, 3, 5)
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    ax2.plot(x, y1, label='sin(x)')
    ax2.legend()


    ax3 = plt.subplot(3, 3, 6)
    x = np.linspace(0, 10, 100)
    y2 = np.cos(x)
    ax3.plot(x, y2, label='cos(x)')
    ax3.legend()


    ax4 = plt.subplot(3, 1, 3)
    ax4.axis('off')
    ax4.table(cellText=table_data.tolist(), loc='center')

    plt.tight_layout()
    plt.show()


menu_names = ['Typo', 'Glyph', 'Phonetic ', 'Synonym', 'Contextual', 'Inflection', 'Syntax', 'Distraction']
for i, menu_name in enumerate(menu_names):
    button = plt.Button(plt.axes([0.1 * i, 0.8, 0.1, 0.05]), menu_name)
    button.on_clicked(menu_click)


menu_click(0)
