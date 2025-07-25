import argparse
import pandas as pd
from src.phoneme_mapper import Mapper
import seaborn as sns
import matplotlib.pyplot as plt


def main(args):

    # load metadata with timestamps
    metadata = pd.read_csv(args.metadata_file, sep='\t', header=0)

    target_phonemes = ['p', 'b', 't', 'd', 'k', 'f', 'v', 's', 'z', 'G', 'x', 'm', 'n', 'N', 'l', 'r', 'j', 'w', 'h',
                       'I', 'E', 'A', 'O', 'Y', 'i', 'e', 'a', 'o', 'u', 'y', '@', 'E+', 'Y+', 'A+', 'S', 'Z', 'g', '2']

    # map cgn to ipa
    mapper = Mapper(language='dutch')
    ipa_target_phonemes = [mapper.cgn_to_ipa[p] for p in target_phonemes]
    metadata['ipa_phoneme'] = metadata['phoneme'].map(mapper.cgn_to_ipa)

    # filter for target phonemes
    ipa_target_phonemes = [mapper.cgn_to_ipa[p] for p in target_phonemes]
    metadata_filtered = metadata[metadata['ipa_phoneme'].isin(ipa_target_phonemes)]

    # compute Q1, Q3, IQR, and max threshold per phoneme
    duration_stats = (
        metadata_filtered
        .groupby('ipa_phoneme')['duration']
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: 'Q1', 0.75: 'Q3'})
    )

    # calculate IQR and max threshold
    duration_stats['IQR'] = duration_stats['Q3'] - duration_stats['Q1']
    duration_stats['max_threshold'] = duration_stats['Q3'] + 1.5 * duration_stats['IQR']

    # save
    print(duration_stats[['Q1', 'Q3', 'IQR', 'max_threshold']])
    duration_stats.to_csv("metadata/cgn_phoneme_duration_stats/phoneme_duration_thresholds.csv")

    if args.create_plots:

        # order phonemes based on mean duration
        phoneme_order = (
            metadata_filtered
            .groupby('ipa_phoneme')['duration']
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

        # create duration boxplot per phoneme
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=metadata_filtered, x='ipa_phoneme', y='duration', order=phoneme_order)
        plt.xticks(rotation=45)
        plt.xlabel("Phoneme")
        plt.ylabel("Duration (s)")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('boxplot_duration_per_phoneme.png', dpi=300)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_file', type=str, default='metadata/timestamps/news_books_phonemes_zs.tsv')
    parser.add_argument('--create_plots', type=bool, default=False)
    args = parser.parse_args()
    main(args)
