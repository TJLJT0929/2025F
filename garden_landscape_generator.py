def generate_summary_report(data):
    # Create separate charts
    fig1, ax1 = plt.subplots()
    ax1.scatter(data['x1'], data['y1'], label=data['garden_name'], fontsize=12)
    ax1.set_xlabel('X-axis Label')
    ax1.set_ylabel('Y-axis Label')
    ax1.legend(fontsize=10)

    fig2, ax2 = plt.subplots()
    ax2.scatter(data['x2'], data['y2'], label=data['garden_name'], fontsize=12)
    ax2.set_xlabel('X-axis Label')
    ax2.set_ylabel('Y-axis Label')
    ax2.legend(fontsize=10)

    fig3, ax3 = plt.subplots()
    ax3.scatter(data['x3'], data['y3'], label=data['garden_name'], fontsize=12)
    ax3.set_xlabel('X-axis Label')
    ax3.set_ylabel('Y-axis Label')
    ax3.legend(fontsize=10)

    fig4, ax4 = plt.subplots()
    ax4.scatter(data['x4'], data['y4'], label=data['garden_name'], fontsize=12)
    ax4.set_xlabel('X-axis Label')
    ax4.set_ylabel('Y-axis Label')
    ax4.legend(fontsize=10)

    plt.show()