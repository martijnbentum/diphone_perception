import random

def make_dict(keys,values):
    d = {}
    for key, value in zip(keys,values):
        if key in d.keys(): d[key] += ', ' + value
        else: d[key] = value
    return d

def reverse_dict(input_d):
    d = {}
    for key, value in input_d.items():
        if ', ' in value:
            for v in value.split(', '):
                d[v]=key
        else: d[value]=key
    return d


class Sampa:
    plosives = 'p,t,b,d,k,g'.split(',')
    plosives_examples = '[p]ut,[b]ad,[t]ak,[d]ak,[k]at,[g]oal'.split(',')

    fricatives = 'f,v,s,z,S,Z,x,G,h'.split(',')
    fricatives_examples = '[f]iets,[v]at,[s]ap,[z]at,[sj]aal,rava[g]e,li[ch]t'
    fricatives_examples += ',re[g]en,ge[h]eel'
    fricatives_examples = fricatives_examples.split(',')

    sonorants = 'N,m,n,J,l,r,w,j'.split(',')
    sonorants_examples = 'la[ng],[m]at,[n]at,ora[nj]e,[l]at,[r]at'
    sonorants_examples += ',[w]at,[j]as'
    sonorants_examples = sonorants_examples.split(',')

    shortvowels = 'I,E,A,O,Y'.split(',')
    shortvowels_examples = 'l[i]p,l[e]g,l[a]t,b[o]m,p[u]t'.split(',')

    longvowels = 'i,y,e,2,a,o,u'.split(',')
    longvowels_examples = 'l[ie]p,b[uu]r,l[ee]g,d[eu]k,l[aa]t'
    longvowels_examples += ',b[oo]m,b[oe]k'
    longvowels_examples = longvowels_examples.split(',')

    sjwa = ['@']
    sjwa_examples = ['g[e]lijk']

    diphthongs = 'E+,Y+,A+'.split(',')
    diphthongs_examples = 'w[ij]s,h[ui]s,k[ou]d'.split(',')
    
    simple_diphthongs = '3,4,5'.split(',')

    loan_vowels = 'E:,Y:,O:'.split(',')
    loan_vowels_examples = 'sc[è]ne,fr[eu]le,z[o]ne'.split(',')

    simple_loan_vowels = 'E,Y,O'.split(',')

    nasal_vowels = 'E~,A~,O~,Y~'.split(',')
    nasal_vowels_examples = 'vacc[in],croiss[ant],c[on]gé,parf[um]'.split(',')

    simple_nasal_vowels = 'E,A,O,Y'.split(',')

    consonants = plosives + fricatives + sonorants
    consonants_examples = []
    consonants_examples.extend(plosives_examples)
    consonants_examples.extend(fricatives_examples)
    consonants_examples.extend(sonorants_examples)

    vowels = shortvowels+longvowels+sjwa+diphthongs+loan_vowels+nasal_vowels
    simple_vowels = shortvowels+longvowels+sjwa
    simple_vowels += simple_diphthongs+simple_loan_vowels+simple_nasal_vowels
    vowels_examples = []
    vowels_examples.extend(shortvowels_examples)
    vowels_examples.extend(longvowels_examples)
    vowels_examples.extend(sjwa_examples)
    vowels_examples.extend(diphthongs_examples)
    vowels_examples.extend(loan_vowels_examples)
    vowels_examples.extend(nasal_vowels_examples)

    symbols = consonants + vowels
    simple_symbols = consonants + simple_vowels
    examples = consonants_examples + vowels_examples

    consonants_to_examples = make_dict(consonants, consonants_examples)
    vowels_to_examples = make_dict(vowels, vowels_examples)
    symbols_to_examples = make_dict(symbols, examples)
    simple_symbols_to_examples = make_dict(simple_symbols, examples)
    
    @property
    def to_ipa_dict(self):
        if hasattr(self,'_to_ipa_dict'): return self._to_ipa_dict
        ipa = Ipa()
        examples_to_symbols = reverse_dict(self.symbols_to_examples)
        ipa_examples_to_symbols = reverse_dict(ipa.symbols_to_examples)
        d = {}
        for example, sampa_symbol in examples_to_symbols.items():
            ipa_symbol = ipa_examples_to_symbols[example]
            d[sampa_symbol] = ipa_symbol
        self._to_ipa_dict = d
        return self._to_ipa_dict

    @property
    def to_simple_ipa_dict(self):
        if hasattr(self,'_to_simple_ipa_dict'): return self._to_simple_ipa_dict
        ipa = Ipa()
        examples_to_symbols = reverse_dict(self.symbols_to_examples)
        ipa_examples_to_symbols = reverse_dict(ipa.simple_symbols_to_examples)
        d = {}
        for example, sampa_symbol in examples_to_symbols.items():
            ipa_symbol = ipa_examples_to_symbols[example]
            d[sampa_symbol] = ipa_symbol
        self._to_simple_ipa_dict = d
        return self._to_simple_ipa_dict

    @property
    def to_simple_sampa_dict(self):
        if hasattr(self,'_to_simple_sampa_dict'): 
            return self._to_simple_sampa_dict
        d = {}
        for symbol, simple_symbol in zip(self.symbols, self.simple_symbols):
            d[symbol] = simple_symbol
        self._to_simple_sampa_dict = d
        return self._to_simple_sampa_dict

    @property
    def to_cv_dict(self):
        if hasattr(self,'_to_cv_dict'): return self._to_cv_dict
        d = {}
        for symbol in self.symbols_to_examples.keys():
            if symbol in self.consonants: d[symbol] = 'C'
            elif symbol in self.vowels: d[symbol] = 'V'
            else: d[symbol] = ' '
        self._to_cv_dict = d
        return self._to_cv_dict
    
    @property
    def simple_sampa_to_simple_ipa(self):
        if hasattr(self,'_simple_sampa_to_simple_ipa_dict'): 
            return self._simple_sampa_to_simple_ipa_dict
        ipa = Ipa()
        examples_to_symbols = reverse_dict(self.simple_symbols_to_examples)
        ipa_examples_to_symbols = reverse_dict(ipa.simple_symbols_to_examples)
        d = {}
        for example, sampa_symbol in examples_to_symbols.items():
            ipa_symbol = ipa_examples_to_symbols[example]
            d[sampa_symbol] = ipa_symbol
        self._simple_sampa_to_simple_ipa_dict= d
        return self._simple_sampa_to_simple_ipa_dict

            
    @property
    def to_simple_sample_dict(self):
        pass
        

class Ipa:
    plosives = 'p,t,b,d,k,g'.split(',')
    plosives_examples = '[p]ut,[b]ad,[t]ak,[d]ak,[k]at,[g]oal'.split(',')

    fricatives = 'f,v,s,z,ʃ,ʒ,x,ɣ,h'.split(',')
    fricatives_examples = '[f]iets,[v]at,[s]ap,[z]at,[sj]aal,rava[g]e,li[ch]t'
    fricatives_examples += ',re[g]en,ge[h]eel'
    fricatives_examples = fricatives_examples.split(',')

    sonorants = 'ŋ,m,n,ɲ,l,r,ʋ,j'.split(',')
    sonorants_examples = 'la[ng],[m]at,[n]at,ora[nj]e,[l]at,[r]at'
    sonorants_examples += ',[w]at,[j]as'
    sonorants_examples = sonorants_examples.split(',')


    shortvowels = 'ɪ,ɛ,ɑ,ɔ,ʉ'.split(',')
    shortvowels_examples = 'l[i]p,l[e]g,l[a]t,b[o]m,p[u]t'.split(',')

    longvowels = 'iː,yː,eː,øː,aː,oː,uː'.split(',')
    longvowels_examples = 'l[ie]p,b[uu]r,l[ee]g,d[eu]k,l[aa]t'
    longvowels_examples += ',b[oo]m,b[oe]k'
    longvowels_examples = longvowels_examples.split(',')

    sjwa = ['ə']
    sjwa_examples = ['g[e]lijk']

    diphthongs = 'ɛi,œy,ɑu'.split(',')
    diphthongs_examples = 'w[ij]s,h[ui]s,k[ou]d'.split(',')

    loan_vowels = 'ɛː,œː,ɔː'.split(',')
    loan_vowels_examples = 'sc[è]ne,fr[eu]le,z[o]ne'.split(',')

    simple_loan_vowels = 'ɛ,ʉ,ɔ'.split(',')

    nasal_vowels = 'ɛ̃ː,ɑ̃ː,ɔ̃ː,ʉ'.split(',')
    nasal_vowels_examples = 'vacc[in],croiss[ant],c[on]gé,parf[um]'.split(',')

    simple_nasal_vowels = 'ɛ,ɑ,ɔ,ʉ'.split(',')

    consonants = plosives + fricatives + sonorants
    consonants_examples = []
    consonants_examples.extend(plosives_examples)
    consonants_examples.extend(fricatives_examples)
    consonants_examples.extend(sonorants_examples)
    
    vowels = shortvowels+longvowels+sjwa+diphthongs+loan_vowels+nasal_vowels
    vowels_examples = []
    vowels_examples.extend(shortvowels_examples)
    vowels_examples.extend(longvowels_examples)
    vowels_examples.extend(sjwa_examples)
    vowels_examples.extend(diphthongs_examples)
    vowels_examples.extend(loan_vowels_examples)
    vowels_examples.extend(nasal_vowels_examples)

    simple_vowels = shortvowels+longvowels+sjwa+diphthongs
    simple_vowels += simple_loan_vowels+simple_nasal_vowels

    symbols = consonants + vowels
    examples = consonants_examples + vowels_examples

    simple_symbols = consonants + simple_vowels

    d = {c:ce for c, ce in zip(consonants,consonants_examples)}

    consonants_to_examples = make_dict(consonants,consonants_examples)
    vowels_to_examples = make_dict(vowels,vowels_examples)
    symbols_to_examples = make_dict(symbols,examples)

    simple_vowels_to_examples = make_dict(simple_vowels,vowels_examples)
    simple_symbols_to_examples = make_dict(simple_symbols, examples)

    @property
    def to_sampa_dict(self):
        if hasattr(self,'_to_sampa_dict'): return self._to_sampa_dict
        sampa = Sampa()
        self._to_sampa_dict = reverse_dict(sampa.to_ipa_dict)
        return self._to_sampa_dict

    @property
    def to_sampa_from_simple_ipa_dict(self):
        if hasattr(self,'_to_sampa_from_simple_ipa_dict'): 
            return self._to_sampa_from_simple_ipa_dict
        sampa = Sampa()
        d = reverse_dict(sampa.to_simple_ipa_dict)
        self._to_sampa_from_simple_ipa_dict= d
        return self._to_sampa_from_simple_ipa_dict

    @property
    def to_ipa_dict(self):
        if hasattr(self,'_to_ipa_dict'): return self._to_ipa_dict
        sampa = Sampa()
        self._to_ipa_dict = sampa.to_ipa_dict
        return self._to_ipa_dict

    @property
    def to_simple_ipa_dict(self):
        if hasattr(self,'_to_simple_ipa_dict'): return self._to_simple_ipa_dict
        sampa = Sampa()
        self._to_simple_ipa_dict= sampa.to_simple_ipa_dict
        return self._to_simple_ipa_dict

    @property
    def to_cv_dict(self):
        if hasattr(self,'_to_cv_dict'): return self._to_cv_dict
        d = {}
        for symbol in self.symbols_to_examples.keys():
            if symbol in self.consonants: d[symbol] = 'C'
            elif symbol in self.vowels: d[symbol] = 'V'
            else: d[symbol] = ' '
        self._to_cv_dict = d
        return self._to_cv_dict

    def to_ipa(self,sampa_symbol):
        return self.to_ipa_dict[sampa_symbol]

    def to_simple_ipa(self,sampa_symbol):
        if sampa_symbol in self.to_simple_ipa_dict.keys():
            return self.to_simple_ipa_dict[sampa_symbol]
    
    
    def word_to_simple_ipa(self,word):
        o = []
        for p in word.sampa_phonemes:
            ipa_phoneme = self.to_simple_ipa(p)
            if ipa_phoneme:
                o.append(ipa_phoneme)
        if not o: return
        word.word_ipa_phoneme = ''.join(o)
        word.ipa = ','.join(o)
        word.save()

    @property
    def ipa_to_simple_ipa_dict_mauser(self):
        d = compose_dicts(self.to_sampa_dict, self.to_simple_ipa_dict)
        d['ui'] = 'œy' # ui is not correct ipa for œy in h[ui]s
        d['au'] ='au'# 'ʌu' # au is not correct ipa for ʌu in k[ou]d
        d['œː'] = 'øː' # œː is not correct ipa for øː in b[eu]k
        # d['øː'] = 'ø' # map øː for simplicity to ø in b[eu]k
        # d['aː'] = 'a' # aː for simplicity to a in l[aa]n
        # d['oː'] = 'o' # oː for simplicity to o in b[oo]m
        d['ɔː'] = 'ɔ' # map 'ɔː' to ɔ contr[o]le
        # d['eː'] = 'e' # eː for simplicity to e l[ee]g
        # d['iː'] = 'i' # iː for simplicity to i l[ie]p
        d['ɛː'] = 'ɛ' # map 'ɛː' to ɛ l[e]g sc[è]ne
        # d['uː'] = 'u' # uː is not correct ipa for d[oe]k
        # d['yː'] = 'y' # yː is not correct ipa for b[uu]r
        d['g'] = 'g' # map ɡ to g (different characters?)
        # d['ʉ'] = 'ʏ' # ʉ is not correct ipa for ʏ in p[u]t
        #mls:
        d['w'] = 'ʋ' # ʋ [w]at
        d['i'] = 'iː' # ʋ [w]at
        d['ɡ'] = 'g' # ʋ [w]at
        d['y'] = 'yː' 
        d['ʏ'] = 'ʉ' # ʉ is not correct ipa for ʏ in p[u]t
        return d 


def convert_simple_sampa_vocab_to_ipa_symbols(simple_sampa_vocab):
    sampa = Sampa()
    d = sampa.simple_sampa_to_simple_ipa
    nvocab = {}
    for key, value in simple_sampa_vocab.items():
        if key not in d.keys(): nvocab[key] = value
        else: nvocab[ d[key] ] = value
    return nvocab
        
    
def reorder_simple_ipa_vocab_to_phoneme_classes(simple_ipa_vocab):
    ipa = Ipa()
    out_vocab = {}
    new_indices = []
    symbols = []
    for s in ipa.simple_symbols + [' ']:
        if s not in symbols:symbols.append(s)
    for i,symbol in enumerate(symbols):
        for key, value in simple_ipa_vocab.items():
            if symbol == key:
                new_indices.append(value)
                out_vocab[key] = i
    return out_vocab, new_indices


class BPCs:
    '''container for all BPC instances.'''
    def __init__(self,names, bpc_sets, complete_set, epsilon = 0.001):
        '''structure to hold and find a given bpc
        names           names of the bpcs (e.g. fricative, plosive)
        bpc_sets        sets of phonemes that belong to the bpc
        complete_set    all the phonemes
        '''
        self.names = names
        self.bpc_sets = bpc_sets
        self.complete_set = complete_set
        self.epsilon = epsilon
        self.make_bpcs()

    def __repr__(self):
        m = 'broad phonetic classes: '
        m += ' '.join(self.names)
        return m

    def __str__(self):
        return '\n'.join([x.__repr__() for x in self.bpcs.values()])
        

    def make_bpcs(self):
        self.bpcs = {}
        for name, bpc_set in zip(self.names, self.bpc_sets):
            other_set = self.complete_set - bpc_set
            self.bpcs[name] = BPC(name,bpc_set, other_set, self.epsilon)

    def find_bpc(self,phoneme):
        for bpc in self.bpcs.values():
            if phoneme in bpc: return bpc
        raise ValueError(phoneme,'not found in any bpc')

    def make_all_synthetic_probability_distributions(self):
        d = {}
        for phoneme in self.complete_set:
            bpc = self.find_bpc(phoneme)
            d[phoneme] = bpc.synthetic_probability_ditribution(phoneme)
        return d
            
        
    

class BPC:
    '''broad phonemic class.'''
    def __init__(self,name,bpc_set,other_set, epsilon = 0.001):
        '''class to contain set of phonemes in a given BPC.
        name        name of the bpc (e.g. fricative)
        bpc_set     the phonemes that belong to the bpc 
        other_set   the phonemes that not belong to the bpc
        epsilon     small prob value for the step destribution    
        '''
        self.name = name
        self.bpc_set = bpc_set
        self.other_set = other_set
        self.epsilon = epsilon

    def __repr__(self):
        m ='BPC: '+ self.name + ' '
        m += ' '.join(self.bpc_set)
        m += ' | other: ' + ''.join(self.other_set)
        return m

    def __contains__(self,phoneme):
        return phoneme in self.bpc_set

    def part_of(self,phoneme):
        return phoneme in self.bpc_set

    def synthetic_probability_ditribution(self, goal_phoneme):
        d = {}
        for phoneme in self.other_set:
            d[phoneme] = self.epsilon
        self.other_prob = sum(d.values())
        self.bpc_prob = 1 - self.other_prob
        for phoneme in self.bpc_set:
            if phoneme == goal_phoneme: continue
            d[phoneme] = self.bpc_prob / (len(self.bpc_set) - 1)
        return d

    def random_probability_distribution(self, goal_phoneme):
        not_gp = self.other_set.union( self.bpc_set ) - set(goal_phoneme)
        n_bpc = len(self.bpc_set) - 1
        self.random_bpc = set(random.sample(not_gp, n_bpc))
        self.random_other = not_gp - self.random_bpc
        d = {}
        for phoneme in self.random_other:
            d[phoneme] = self.epsilon
        self.random_other_prob = sum(d.values())
        self.random_bpc_prob = 1 - self.random_other_prob
        for phoneme in self.random_bpc:
            d[phoneme] = self.random_bpc_prob / len(self.random_bpc)
        return d

    def random_other_probability_distribution(self, goal_phoneme):
        not_gp = self.other_set.union( self.bpc_set ) - set(goal_phoneme)
        n_bpc = len(self.bpc_set) - 1
        self.random_bpc = set(random.sample(self.other_set, n_bpc))
        self.random_other = not_gp - self.random_bpc
        d = {}
        for phoneme in self.random_other:
            d[phoneme] = self.epsilon
        self.random_other_prob = sum(d.values())
        self.random_bpc_prob = 1 - self.random_other_prob
        for phoneme in self.random_bpc:
            d[phoneme] = self.random_bpc_prob / len(self.random_bpc)
        return d
            
        
            
    
    
def make_bpcs():
    '''make a BPCs instance with all broad phonetic classes.'''
    names = 'plosive,nasal,approximant,fricative'
    names += ',high_vowel,mid_vowel,low_vowel'
    names = names.split(',')
    bpcs = 'ptkbdg,nmŋɲ,lrjw,sfxzvɣ,iɪyu,ʏøeɛɔoə,aɑ'.split(',')
    bpcs = [set(x) for x in bpcs]
    complete= set('ptkbdgnmŋɲlrjwsfxzvɣiɪyuʏøeɛɔoəaɑ')
    return BPCs(names,bpcs,complete)



def compose_dicts(a, b):
    return {k: b[v] for k, v in a.items() if v in b}
