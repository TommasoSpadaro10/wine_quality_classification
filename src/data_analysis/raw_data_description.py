import matplotlib.pyplot as plt

import utils


def describe_raw_data():
    df = utils.read_csv('winequality-red')

    utils.ensure_col_exists(df, 'quality')

    print("Descriptive statistics for raw data:", '\n')
    utils.print_descriptive_statistics(df)

    utils.make_plot(
        title='Wine Quality Distribution',
        xlabel='Quality',
        ylabel='Frequency',
        file_name='wine_quality_distribution',
        plot=lambda: plt.hist(df['quality'], bins=5, edgecolor='black')
    )
