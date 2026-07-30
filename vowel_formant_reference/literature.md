# Literature for Dutch vowel formant references

This document records the sources used to construct, validate, and interpret
the formant tables. Data files must identify the applicable source and preserve
the original source's labels, fields, population, and measurement context.
Each stored table records the full reference, URL, page range, source page and
table number where available. The current implementation scope includes
monophthongs only. `/eː, øː, oː/` are included according to their phonological
classification and represented by central measurements. Some Netherlandic
speakers realize these vowels dynamically; trajectory measurements are
deliberately outside the current scope.

## Representation-probe paper

- Choi, K., & Yeo, E. J. (2022). *Opening the Black Box of wav2vec Feature
  Encoder*. arXiv:2210.15386. https://arxiv.org/abs/2210.15386

  Source location: PDF page 3, Section 3.4, Figures 6–8.

  This is the source of the synthetic frequency, sinusoidal-component
  “formant,” amplitude, and temporal probes. Its three-sine stimuli are a
  reproduction baseline and should not be described as source-filter vowel
  synthesis.

## Dutch formant datasets distributed with Praat

- Pols, L. C. W., Tromp, H. R. C., & Plomp, R. (1973). Frequency analysis of
  Dutch vowels from 50 male speakers. *Journal of the Acoustical Society of
  America, 53*, 1093–1101.
  https://www.fon.hum.uva.nl/praat/manual/Pols_et_al___1973_.html

  Source location: pp. 1093–1101. The Praat documentation does not identify a
  single source table number for its embedded data.

- Van Nierop, D. J. P. J., Pols, L. C. W., & Plomp, R. (1973). Frequency
  analysis of Dutch vowels from 25 female speakers. *Acustica, 29*, 110–118.
  https://www.fon.hum.uva.nl/praat/manual/Van_Nierop_et_al___1973_.html

  Source location: pp. 110–118. The Praat documentation does not identify a
  single source table number for its embedded data.

  Praat combines these two studies in a per-speaker table for 12 Dutch
  monophthongs in `/hVt/` context. It supplies F1, F2, F3 and the levels of
  those formants relative to overall SPL:
  https://www.fon.hum.uva.nl/praat/manual/Create_formant_table__Pols___Van_Nierop_1973_.html

  Machine-readable source location: Praat command
  `Create formant table (Pols & Van Nierop 1973)`; page and table number are
  not applicable to this built-in table. Although Praat distributes one
  combined table, the implementation splits and stores the male Pols table and
  female Van Nierop table separately because they originate in different
  publications.

- Weenink, D. J. M. (1985). Formant analysis of Dutch vowels from 10 children.
  *Proceedings of the Institute of Phonetic Sciences of the University of
  Amsterdam, 9*, 45–52.
  https://www.fon.hum.uva.nl/praat/manual/Weenink__1985_.html

  Source location: pp. 45–52. The Praat documentation does not identify a
  single source table number for its embedded data.

  Praat's associated table contains one production of each of 12 Dutch
  monophthongs by 10 men, 10 women, and 10 children, with F0 and F1–F3:
  https://www.fon.hum.uva.nl/praat/manual/Create_formant_table__Weenink_1985_.html

  Machine-readable source location: Praat command
  `Create formant table (Weenink 1985)`; page and table number are not
  applicable to this built-in table.

## Modern Dutch vowel descriptions and normalization

- Adank, P., Van Hout, R., & Smits, R. (2004). An acoustic description of the
  vowels of Northern and Southern Standard Dutch. *Journal of the Acoustical
  Society of America, 116*(3), 1729–1738.
  https://doi.org/10.1121/1.1779271

  Numeric source location: Table I, printed p. 1731, reports group-average
  duration and F0/F1/F2/F3 at 50%. Only its monophthong rows are in scope.
  Table II, printed p. 1732, contains dynamic-vowel measurements and is out of
  scope. The paper describes 1,200 underlying tokens but does not publish their
  individual measurements in these tables; the stored data must therefore
  remain labelled as group summaries.

- Adank, P., Smits, R., & Van Hout, R. (2004). A comparison of vowel
  normalization procedures for language variation research. *Journal of the
  Acoustical Society of America, 116*(5), 3099–3107.
  https://doi.org/10.1121/1.1795335

  Source location: pp. 3099–3107. No numeric table from this article is loaded.

  The study uses Dutch recordings from 80 female and 80 male talkers. It
  informs diagnostic normalized views, but raw Hertz values remain the primary
  stimulus parameters.

## Dutch diphthong inventory

- Taalportaal. *Diphthongs*. Instituut voor de Nederlandse Taal.
  https://taalportaal.org/taalportaal/topic/pid/topic-13998813314542255

  The reference identifies the three Dutch phonemic closing diphthongs
  `/ɛi, œy, ɑu/` and distinguishes them from vowel-plus-glide sequences.
  This source is retained for provenance of the scope decision; diphthongs and
  their measurements are excluded from the current implementation.

## Male/female differences

`pols_1973_formants()`, `van_nierop_1973_formants()`,
`weenink_1985_formants()`, and `adank_2004_formants()` expose the adult
summaries without requiring callers to know internal table names. The generic
`literature_gender_formants()` selector remains available. Weenink (1985)
covers all 12 full Dutch monophthongs with F0 and F1--F3. Adank et al. (2004)
can instead provide Northern or Southern Standard Dutch group means. These
sources do not report schwa, so `/ə/` must be estimated from the selected-phone
data rather than invented by scaling another vowel.

- Diehl, R. L., Lindblom, B., Hoemeke, K. A., & Fahey, R. P. (1996). On
  explaining certain male-female differences in the phonetic realization of
  vowel categories. *Journal of Phonetics, 24*(2), 187–208.
  https://doi.org/10.1006/jpho.1996.0011

  Source location: pp. 187–208. No numeric table from this article is loaded.

  Male-to-female formant scaling is non-uniform across vowel categories. This
  supports using sex-specific Dutch observations, or at most vowel- and
  formant-specific empirical ratios, instead of one global multiplier.

## Praat and Parselmouth implementation references

- Praat manual, *Source-filter synthesis 1: Creating a source from pitch*:
  https://www.fon.hum.uva.nl/praat/manual/Source-filter_synthesis_1__Creating_a_source_from_pitch.html
- Praat manual, *Source-filter synthesis 2: Filtering a source*:
  https://www.fon.hum.uva.nl/praat/manual/Source-filter_synthesis_2__Filtering_a_source.html
- Praat manual, *Sound & FormantGrid: Filter*:
  https://www.fon.hum.uva.nl/praat/manual/Sound___FormantGrid__Filter.html
- Parselmouth documentation:
  https://parselmouth.readthedocs.io/en/stable/

These are implementation references rather than acoustic datasets. Generated
tables and stimuli must record the Parselmouth and embedded Praat versions.
