import matplotlib.pyplot as plt

# Turn on interactive plotting mode
plt.ion()

def plot(scores, mean_scores):
    # Clear current figure layout
    plt.clf()

    # Config titles and labels
    plt.title("Snake AI Training Progress")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")

    # Plot raw scores and running average scores
    plt.plot(scores, label="Game Score", color="#1f77b4", alpha=0.6)
    plt.plot(mean_scores, label="Running mean", color="#ff7f0e", linewidth=2)

    # Set lower boundary bound for y axis
    plt.ylim(ymin=0)

    # Show metric text on latest data point
    if scores:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.2f}")

    plt.legend(loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Pause briefly to force graphic UI windaw redraw event
    plt.draw()
    plt.pause(0.01)
