from matplotlib import pyplot as plt

path = "C:\\Users\\shaha\\AppData\\Roaming\\JetBrains\\PyCharmCE2023.1\\light-edit\\positionsStar.txt"
ly = 9e12

def plot_stars(ax, x_values, y_values, plot_count):
    ax.scatter(x_values, y_values)
    ax.set_xlabel('X Coordinate (in light years)')
    ax.set_ylabel('Y Coordinate (in light years)')
    ax.set_xlim(ly, 100 * ly)  # Set x-axis limits
    ax.set_ylim(ly, 100 * ly)  # Set y-axis limits
    ax.grid(True)
    if plot_count == 1:
        txt = r'$T_0$'
    if plot_count == 2:
        txt = r'$\frac{T_{max}}{2}$'
    if plot_count == 3:
        txt = r'$T_{max}$'
    ax.set_title(f'Positions of Stars at time {txt}')

# Open the file
def plot_simulation(filename_prefix="plot"):
    with open(path, 'r') as file:
        lines = file.readlines()

    # Initialize lists to store x, y coordinates
    x_values = []
    y_values = []

    # Read the lines and plot until an empty line is encountered
    plot_count = 0
    for line in lines:
        if line.strip():  # Check if the line is not empty
            x, y = map(float, line.split())  # Assuming the values are separated by whitespace
            x_values.append(x)
            y_values.append(y)
        else:
            # Plot the data when an empty line is encountered
            if x_values and y_values:  # Check if there is data to plot
                fig, ax = plt.subplots(figsize=(6, 6))  # Create a new figure for each subplot
                plot_stars(ax, x_values, y_values, plot_count + 1)
                filename = f"{filename_prefix}_{plot_count + 1}.png"
                plt.savefig(filename)
                plt.close(fig)  # Close the figure to free up memory
                print(f"Plot {plot_count + 1} saved as '{filename}'")
                x_values = []  # Reset lists for the next plot
                y_values = []
                plot_count += 1

    # Plot the last set of data if any
    if x_values and y_values:
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_stars(ax, x_values, y_values, plot_count + 1)
        filename = f"{filename_prefix}_{plot_count + 1}.png"
        plt.savefig(filename)
        plt.close(fig)  # Close the figure to free up memory
        print(f"Plot {plot_count + 1} saved as '{filename}'")



def speedups_diff_threads():
    num_threads = [1, 2, 4, 6, 8]

    # Runtimes corresponding to each number of threads
    runtimes = [75.845, 37.57, 19.062, 26.233, 20.464]

    # Calculate speedup
    baseline_time = runtimes[0]
    speedup = [baseline_time / runtime for runtime in runtimes]
    # Plotting
    plt.plot(num_threads, speedup, marker='o', linestyle='-')
    plt.xlabel('Number of Threads')
    plt.xticks([1,2,4,6,8])
    plt.ylabel('Speedup')
    plt.title('Speedup vs. Number of Threads')
    plt.grid(True)
    plt.show()

def speedups_diff_stars():
    num_stars = [50, 100, 250, 500, 750, 1000]
    runtimes = [0.972475, 1.232872, 2.215993, 5.923131, 12.018312, 20.482378]
    # Calculate speedup
    baseline_time = runtimes[0]
    speedup = [baseline_time / runtime for runtime in runtimes]
    # Plotting
    plt.plot(num_stars, speedup, marker='o', linestyle='-')
    plt.xlabel('$N$ = Number of stars')
    plt.xticks(num_stars)
    plt.ylabel('Speedup')
    plt.title(r'$\frac{TN}{T{50}}$')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    speedups_diff_threads()
    speedups_diff_stars()
    plot_simulation()
