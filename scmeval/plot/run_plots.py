import matplotlib.pyplot as plt

from scmeval.plot.intra_summarizer_correlation_plot import plot_intra_system_correlation_for_measures

plot_intra_system_correlation_for_measures(["HUM", "CCL C/D", "CCL WSJ", "GRN", "BAS", "SQE"])
plt.savefig("saved_plots/intra_summarizer_correlation.png")
