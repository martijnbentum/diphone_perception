# diphone_perception
Database of Dutch diphone perception
========================================================

This files describes the naming conventions of files.

--------------------------------------------------------------

Contents:

  1. Material selection
  2. Labeling and production of phonemes
  3. Gating
  4. File naming conventions

--------------------------------------------------------------


1. MATERIAL SELECTION


Diphones were generated automatically by combining the following Dutch
phonemes (CELEX DISC transcription):
	I E A O } @
	i a u y e | o K L M
	p b t d k g N m n l r f v s z S Z j x G h w _


2. LABELING and PRODUCTION OF DIPHONES

Label names were written in CELEX phonetic codes, except the following:
	| -> eu
	_ -> J
	@ -> V
	} -> U

Coding of labels: Extensions indicate stress status of diphone:
	_0 -> segment is unstressed;
	_1 -> segment is stressed;
	_2 -> a syllable boundary is present between 1st and 2nd phoneme of
	      the diphone, with stress on the 2nd syllable;
        _3 -> a syllable boundary is present between 1st and 2nd phoneme of
	      the diphone, with stress on the 1st syllable;
	_s -> segment contains two strong vowels (both stressed);
	_w -> segment contains two weak vowels (both unstressed).

	      Codes 0,1,2, and 3 may have extended coding:
	      a -> the speech segment is produced with prefix 'a'
	      b -> the speech segment is produced with prefix 'b'


3. GATING

DOS does not make a distinction between capitals and non-capitals.
Therefore, this distinction cannot be present in the diphone file names, and
they had to be renamed. The following format was used:

	x1_x2yz.sd  where x1 represents the first phoneme,
			  x2 the second (x1 and x2 may both have length of 2),
			  y  represents the syllabification type of the speech
			     segment in which the phoneme is a part of (see
			     coding of labels above),
			  z  is optional and denotes the presence of an
			     additional prefix (is always /a/ or /b/).

The following phonemes had to be recoded to provide MsDos compatible file names:

OLD   -> NEW
------------
I     -> ih
E     -> eh
A     -> ah
O     -> oh
U (}) -> uh
V (@) -> vh
K     -> ei
L     -> ui
M     -> au
N     -> ng
S     -> sh
Z     -> zh
G     -> gx
J (_) -> dj


4. FILE NAMING CONVENTIONS

a) Original recordings: e.g. a_x2b.sd

	1. phoneme 1 in diphone: a
	2. phoneme 2 in diphone: x
	3. stress condition: 2
	4. leading environment: b (optional)

	Phonemes are encoded according to CELEX conventions.


b) Gated files: e.g. a_x2b3.wav

	1. phoneme 1 in diphone: a
	2. phoneme 2 in diphone: x
	3. stress condition: 2
	4. leading environment: b (optional)
	5. gate: 3

	Phonemes are encoded according to CELEX conventions.


c) Label files: a_x2b.sd.lab

	1. phoneme 1 in diphone: a
	2. phoneme 2 in diphone: x
	3. stress condition: 2
	4. leading environment: b (optional)

	Phonemes are encoded according to CELEX conventions.


d1) Response files - rawdata: pp03_gate5

	1. Subject number: pp03
	2. gate: 5


d2) Response files - confusion matrices: phon1_conf_matrix_gate1.dat

	1. indicates if contents belong to first or second phoneme
	   in a diphone: phon1
	2. gate: 1

