

#Diphones were generated automatically by combining the following Dutch
#phonemes (CELEX DISC transcription):
phonemes = 'I E A O } @ i a u y e | o K L M'
phonemes += ' p b t d k g N m n l r f v s z S Z j x h w _'
phonemes = phonemes.split()
# G ɣ (voice fricative velar is also reported in the README file
# but it is not in the matrix overview or mentioned in the paper
# so it is not included here)

# _ dʒ (voiced postalveolar affricate) is noted as - in the matrix overview


#Label names were written in CELEX phonetic codes, except the following:
rewrite_dict = {'|':'eu','_':'J','@':'V','}':'U'}

#The following phonemes had to be recoded to provide MsDos compatible file names:

recode_dict = {'I':'ih','E':'eh','A':'ah','O':'oh','U':'uh','V':'vh',
    'K':'ei','L':'ui','M':'au','N':'ng','S':'sh','Z':'zh','G':'gx','J':'dj'}



class Mapper:
    '''map phonemes from one set to another set.'''
    def __init__(self, language = 'dutch'):
        self.language = language
        self.ipa_set = ipa_set
        self.sampa_set = sampa_set
        self.celex_set = celex_set
        self.disc_set = disc_set
        self.cgn_set = cgn_set
        self.examples_dutch = examples_dutch
        self.examples_english= examples_english
        self.examples_german= examples_german
        names = 'ipa_set,sampa_set,celex_set,disc_set,cgn_set'
        names += ',examples_dutch,examples_english,examples_german'
        self.names = names.split(',')
        self._make_dicts()
        self._add_arpabet()
        if self.language != 'dutch':
            self._fix_w()
        self._add_baldey()
        self.__add_coolest()

    def _make_dicts(self):
        '''create dicts to map ipa, sampa, celex and disc phonemes
        '''
        self.ipa_to_sampa = {}
        self.ipa_to_celex = {}
        self.ipa_to_disc = {}
        self.ipa_to_cgn= {}
        self.sampa_to_ipa = {}
        self.sampa_to_celex = {}
        self.sampa_to_disc= {}
        self.celex_to_ipa = {}
        self.celex_to_sampa = {}
        self.celex_to_disc = {}
        self.disc_to_ipa = {}
        self.disc_to_sampa = {}
        self.disc_to_celex = {}
        self.disc_to_cgn = {}
        self.cgn_to_disc = {}
        self.cgn_to_ipa = {}
        
        sets = [self.ipa_set,self.sampa_set]
        sets += [self.celex_set,self.disc_set, self.cgn_set]
        for ipa, sampa, celex, disc, cgn in zip(*sets):
            self.ipa_to_sampa[ipa] = sampa
            self.ipa_to_disc[ipa] = disc
            self.ipa_to_celex[ipa] = celex
            self.ipa_to_cgn[ipa] = cgn
            self.sampa_to_ipa[sampa] = ipa
            self.sampa_to_disc[sampa] = disc
            self.sampa_to_celex[sampa] = celex
            self.celex_to_ipa[celex] = ipa
            self.celex_to_sampa[celex] = sampa
            self.celex_to_disc[celex] = disc
            self.disc_to_ipa[disc] = ipa
            self.disc_to_sampa[disc] = sampa
            self.disc_to_celex[disc] = celex
            self.disc_to_cgn[disc] = cgn
            self.cgn_to_ipa[cgn] = ipa
            self.cgn_to_disc[cgn] = disc
        self.ipa_to_disc['oʊ'] = '5'
        self.cgn_to_ipa['J'] = 'ɲ'


    def _add_arpabet(self):
        self.ipa_to_arpabet = ipa_to_arpabet
        self.arpabet_to_ipa = arpabet_to_ipa
        self.arpabet_to_examples = arpabet_to_examples
        self.arpabet_to_disc= {}
        self.disc_to_arpabet = {}
        for ipa, disc in self.ipa_to_disc.items():
            if ipa == '*': ipa = 'r'
            if ipa == 'iː': ipa = 'i'
            if ipa == 'uː': ipa = 'u'
            if ipa == 'ɑɪ': ipa = 'ai'
            if ipa == 'ɑ̃ː': ipa = 'ɑː'
            if ipa == 'ɒ' : ipa = 'ɔ'
            if ipa == 'ɔ̃' : ipa = 'ɔ'
            if ipa == 'æ̃ː': ipa = 'æ̃'
            if ipa == 'ŋ̩': ipa = 'ŋ'
            if ipa not in self.ipa_to_arpabet.keys():
                continue
            arpabet= self.ipa_to_arpabet[ipa]
            self.arpabet_to_disc[arpabet] = disc
            self.disc_to_arpabet[disc] = arpabet
        
    def _fix_w(self):
        self.celex_to_ipa['w'] ='ʋ'
        self.sampa_to_ipa['w'] ='ʋ'
        self.disc_to_ipa['w'] ='ʋ'

    def _add_baldey(self):
        '''this a restricted phoneme set used in baldey textgrids
        probably based on cgn
        '''
        self.baldey_to_ipa = {}
        self.baldey_textgrid_phoneme_set = baldey_textgrid_phoneme_set
        ps = self.baldey_textgrid_phoneme_set
        for sampa, ipa in self.sampa_to_ipa.items():
            if sampa in ps: self.baldey_to_ipa[sampa] = ipa
        self.baldey_to_ipa['y'] = 'yː'
        self.baldey_to_ipa['i'] = 'iː'
        self.baldey_to_ipa['Y+'] = 'œy'
        self.baldey_to_ipa['u'] = 'uː'
        self.baldey_to_ipa['Ei'] = 'ɛi' # mistake in baldey textgrid output?
        self.baldey_to_ipa['E+'] = 'ɛi'
        self.baldey_to_ipa['A+'] = 'ɑu'
        self.baldey_to_ipa['a'] = 'aː'
        self.baldey_to_ipa['e'] = 'eː'
        self.baldey_to_ipa['Y'] = 'ʉ'
        self.baldey_to_ipa['o'] = 'oː'
        self.baldey_to_ipa['2'] = 'øː'
        self.baldey_to_ipa['w'] = 'ʋ'
        self.ipa_to_baldey= {}
        self.disc_to_baldey= {}
        self.baldey_to_disc = {}
        for baldey, ipa in self.baldey_to_ipa.items():
            disc = self.ipa_to_disc[ipa]
            self.baldey_to_ipa[baldey] = ipa
            self.disc_to_baldey[disc] = baldey
            self.baldey_to_disc[baldey] = disc

    def __add_coolest(self):
        self.coolest_to_ipa = {}
        self.coolest_textgrid_phoneme_set = coolest_textgrid_phoneme_set
        ps = self.coolest_textgrid_phoneme_set
        for sampa, ipa in self.sampa_to_ipa.items():
            if sampa in ps: self.coolest_to_ipa[sampa] = ipa
        self.coolest_to_ipa['u'] = 'uː'
        self.coolest_to_ipa['Y'] = 'uː'
        self.coolest_to_ipa['y'] = 'yː'
        self.coolest_to_ipa['i'] = 'iː'
        self.coolest_to_ipa['Ei'] = 'ɛi'
        self.coolest_to_ipa['9y'] = 'œy'
        self.coolest_to_ipa['w'] = 'ʋ'


            

def validate(mapper= None):
    if not mapper: mapper= Mapper()
    n_items = {}
    for name in mapper.names:
        attr = getattr(mapper,name)
        n_items[name] = len(attr)
    return n_items
    
        
def set_none(examples):
    for index, example in enumerate(examples):
        if example == 'None':
            examples[index] = None
    return examples

def show(mapper= None):
    if not mapper: mapper= Mapper()
    attrs = [getattr(mapper,name) for name in mapper.names]
    for line in zip(*attrs):
        sl = list(map(str,line))
        sl[:4] = [x.ljust(4) for x in sl[:4]]
        print('\t'.join(sl))


ipa_set = 'p,b,t,d,k,g,ŋ,m,n,l,r,f,v,θ,ð,s,z,ʃ,ʒ,j,x,ɣ'
ipa_set += ',h,w,ʋ,pf,ts,tʃ,dʒ,ŋ̩,m̩,n̩,l̩,*,iː,iːː,ɑː,aː,ɔː'
ipa_set += ',uː,ɜː,yː,yːː,ɛː,œː,ɒː,eː,øː,oː,eɪ,ɑɪ,ɔɪ,əʊ'
ipa_set += ',aʊ,ɪə,ɛə,ʊə,ɛi,œy,ɑu,ai,au,ɔy,ɪ,ʏ,ɛ,œ,æ,a,ɑ'
ipa_set += ',ɒ,ʌ,ɔ,ʊ,ʉ,ə,œ̃,æ̃,ɑ̃ː,æ̃ː,ɒ̃ː'
ipa_set = ipa_set.split(',')

sampa_set = 'p.b.t.d.k.g.N.m.n.l.r.f.v.T.D.s.z.S.Z.j.x.G.h.w.w'
sampa_set += '.pf.ts.tS.dZ.N,.m,.n,.l,.r*.i:.i::.A:.a:.O:.u:'
sampa_set += '.3:.y:.y::.E:./:.Q:.e:.|:.o:.eI.aI.OI.@U.aU'
sampa_set += '.I@.E@.U@.EI./I.Au.ai.au.Oy.I.Y.E./.{.a.A'
sampa_set += '.Q.V.O.U.}.@./".{".A":.{":.O":'
sampa_set = sampa_set.split('.')

celex_set = 'p.b.t.d.k.g.N.m.n.l.r.f.v.T.D.s.z.S.Z.j.x.G.h.w.w'
celex_set += '.pf.ts.tS.dZ.N,.m,.n,.l,r*..i:.i::.A:.a:.O:.u:'
celex_set += '.3:.y:.y::.E:.U:.Q:.e:.&:.o:.eI.aI.OI.@U.aU'
celex_set += '.I@.E@.U@.EI.UI.AU.ai.au.Oy.I.Y.E.Q.&.a.A'
celex_set += '.Q.V.O.U.U.@.Q".&".A":.&":.O":'
celex_set = celex_set.split('.')

disc_set = 'p,b,t,d,k,g,N,m,n,l,r,f,v,T,D,s,z,S,Z,j,x,G,h,w,w'
disc_set += ',+,=,J,_,C,F,H,P,R,i,!,#,a,$,u,3,y,(,),*,<,e,|,o,1,2,4'
disc_set += ',5,6,7,8,9,K,L,M,W,B,X,I,Y,E,/,{,&,A,Q,V,O,U,},@,^'
disc_set += ',c,q,0,"'
disc_set = disc_set.split(',')

cgn_set = 'p,b,t,d,k,g,N,m,n,l,r,f,v,None,None,s,z,S,Z,j,x,G,h'
cgn_set += ',None,w,None,None,None,None,None,None,None,None,None'
cgn_set += ',i,None,None,a,None,u,None,y,None,E:,Y:,O:,e,2,o'
cgn_set += ',None,None,None,None,None,None,None,None'
cgn_set += ',E+,Y+,A+,None,None,None,I,None,E,None,None,None,A'
cgn_set += ',None,None,O,None,Y,@,Y~,A~,E~,O~'
cgn_set = set_none(cgn_set.split(','))

examples_dutch = 'put,bad,tak,dak,kat,goal,lang,mat,nat,lat'
examples_dutch += ',rat/later,fiets,vat,None,None,sap,zat,sjaal'
examples_dutch += ',ravage'
examples_dutch += ',jas,light/gaat,regen,had,None,wat,None,None'
examples_dutch += ',None'
examples_dutch += ',jazz,None,None,None,None,None,liep,analyse,None'
examples_dutch += ',laat,None,boek,None,buut,centrifuge,scene'
examples_dutch += ',freule,zone'
examples_dutch += ',leeg,deuk,boom,None,None,None,None,None'
examples_dutch += ',None'
examples_dutch += ',None,None,wijs,huis,koud,None,None,None'
examples_dutch += ',lip,None,leg,None,None,None,lat,None,None,bom'
examples_dutch += ',None'
examples_dutch += ',put,gelijk,None,None,None,None,None'
examples_dutch = set_none(examples_dutch.split(','))

    
examples_english = 'pat,bad,tack,dad,cad,game,bang,mad,nat,lad'
examples_english += ',rat,fat'
examples_english += ',vat,thin,then,sap,zap,sheep,measure,yank,loch'
examples_english += ',None,had,why,None,None,None,cheap,jeep,bacon'
examples_english += ',idealism,burden,dangle,father,bean,None,barn'
examples_english += ',None,born,boon,burn,None,None,None,None'
examples_english += ',None,None,None,None,bay,buy,boy,no,brow'
examples_english += ',peer,pair'
examples_english += ',poor,None,None,None,None,None,None,pit'
examples_english += ',None,pet'
examples_english += ',None,pat,None,None,pot,putt,None,put'
examples_english += ',None,another'
examples_english += ',None,timbre,detente,lingerie,bouillon'
examples_english = set_none(examples_english.split(','))

examples_german = 'Pakt,Bad,Tag,dann,kalt,Gast,Klang,maß'
examples_german += ',naht,Last,Ratte'
examples_german += ',falsh,Welt,None,None,Gas,Suppe,Schiff'
examples_german += ',Genie,Jack'
examples_german += ',Bach/ich,None,Hand,waterproof,None,Pferd'
examples_german += ',Zahl,Matsch'
examples_german += ',Gin,None,None,None,None,None,Lied'
examples_german += ',None,Advantage'
examples_german += ',klar,Allroundman,Hut,Teamwork,für,None'
examples_german += ',Käse,None,None'
examples_german += ',Mehl,Möbel,Boot,Native,Shylock,Playboy,None'
examples_german += ',Allroundsportler,None,None,None,None,None,None'
examples_german += ',weit,Haut,freut,Mitte,Pfütze,Bett'
examples_german += ',Götter,Ragtime'
examples_german += ',hat,Kalevala,None,Plumpudding,Glocke,Pult,None'
examples_german += ',Beginn,Parfum,impromptu,Détente,lingerie'
examples_german += ',bouillon'
examples_german = set_none(examples_german.split(','))
    

arpabet_to_ipa = {
    'AA': 'ɑ',
    'AE': 'æ',
    'AH': 'ʌ',
    'AO': 'ɔ',
    'AW': 'aʊ',
    'AX': 'ə',
    'AY': 'ai',
    'EH': 'ɛ',
    'ER': 'ɝ',
    'EY': 'eɪ',
    'IH': 'ɪ',
    'IX': 'ɨ',
    'IY': 'i',
    'OW': 'oʊ',
    'OY': 'ɔɪ',
    'UH': 'ʊ',
    'UW': 'u',
    'UX': 'ʉ',
    'B': 'b',
    'CH': 'tʃ',
    'D': 'd',
    'DH': 'ð',
    'DX': 'ɾ',
    'EL': 'l̩',
    'EM': 'm̩',
    'EN': 'n̩',
    'F': 'f',
    'G': 'g',
    'HH': 'h',
    'H': 'h',
    'JH': 'dʒ',
    'K': 'k',
    'L': 'l',
    'M': 'm',
    'N': 'n',
    'NX': 'ɾ̃',
    'NG': 'ŋ',
    'P': 'p',
    'Q': 'ʔ',
    'R': 'ɹ',
    'S': 's',
    'SH': 'ʃ',
    'T': 't',
    'TH': 'θ',
    'V': 'v',
    'W': 'w',
    'WH': 'ʍ',
    'Y': 'j',
    'Z': 'z',
    'ZH': 'ʒ'
}

ipa_to_arpabet = {}
for key,value in arpabet_to_ipa.items():
    ipa_to_arpabet[value] = key
ipa_to_arpabet['n̩'] = 'n'
ipa_to_arpabet['r'] = 'R'
ipa_to_arpabet['x'] = 'X'
ipa_to_arpabet['ɜː'] = 'ER'
ipa_to_arpabet['ɑɹ'] = 'R'
ipa_to_arpabet['ɚ'] = 'AX'
ipa_to_arpabet['əʊ'] = 'UH'
ipa_to_arpabet['ɪə'] = 'IH'
ipa_to_arpabet['ɛə'] = 'EH'
ipa_to_arpabet['ʊə'] = 'UH'
ipa_to_arpabet['æ̃'] = 'AE'
ipa_to_arpabet['ɔ̃'] = 'AO'
ipa_to_arpabet['m̩'] = 'M'
ipa_to_arpabet['ɑː'] = 'AA'
ipa_to_arpabet['ɔː'] = 'AO'


arpabet_to_examples = {
    'AA': 'b(al)m,b(o)t',
    'AE': 'b(a)t',
    'AH': 'b(u)tt',
    'AO': 'c(augh)t,st(o)ry',
    'AW': 'b(ou)t',
    'AX': 'comm(a)',
    'AY': 'b(i)te',
    'EH': 'b(e)t',
    'ER': 'b(ir)d,forew(or)d',
    'EY': 'b(ai)t',
    'IH': 'b(i)t',
    'IX': 'ros(e)s,rabb(i)t',
    'IY': 'b(ea)t',
    'OW': 'b(oa)t',
    'OY': 'b(oy)',
    'UH': 'b(oo)k',
    'UW': 'b(oo)t',
    'UX': 'd(u)de',
    'B': '(b)uy',
    'CH': '(Ch)ina',
    'D': '(d)ie',
    'DH': '(th)y',
    'DX': 'bu(tt)er',
    'EL': 'bott(le)',
    'EM': 'ryth(m)',
    'EN': 'butt(on)',
    'F': '(f)ight',
    'G': '(g)uy',
    'HH': '(h)igh',
    'H': '(h)igh',
    'JH': '(j)ive',
    'K': '(k)ite',
    'L': '(l)ie',
    'M': '(m)y',
    'N': '(n)igh',
    'NX': 'wi(nn)er',
    'NG': 'si(ng)',
    'P': '(p)',
    'Q': 'uh(-)oh',
    'R': '(r)ye',
    'S': '(s)igh',
    'SH': '(sh)y',
    'T': '(t)ie',
    'TH': '(th)igh',
    'V': '(v)ie',
    'W': '(w)ise',
    'WH': '(wh)y',
    'Y': '(y)acht',
    'Z': '(z)oo',
    'ZH': 'plea(s)ure'
}

ipa_to_dutch_examples = {}
for ipa, example in zip(ipa_set,examples_dutch):
    ipa_to_dutch_examples[ipa] = example

ipa_to_english_examples = {}
for ipa, example in zip(ipa_set,examples_english):
    ipa_to_english_examples[ipa] = example

ipa_to_german_examples = {}
for ipa, example in zip(ipa_set,examples_german):
    ipa_to_german_examples[ipa] = example

baldey_textgrid_phoneme_set = 'n,Y,m,@,r,s,b,E,k,f,A,t,y,p,i,x,O'
baldey_textgrid_phoneme_set +=',Y+,j,v,u,l,I,z,E+,G,w,d,h,o,e,g,a,N'
baldey_textgrid_phoneme_set +=',Au,Ei,S,T,A+,Z'
baldey_textgrid_phoneme_set = baldey_textgrid_phoneme_set.split(',')

celex_dutch_phoneme_set = 'a,x,j,@,t,A,p,l,s,d,n,I,N,k,b,E,Z,K,m,e,v,r,L'
celex_dutch_phoneme_set += ',G,f,O,w,u,z,i,},h,o,y,M,|,S,g,*,),!,<,_,('
celex_dutch_phoneme_set = celex_dutch_phoneme_set.split(',')

coolest_textgrid_phoneme_set = 'o:,u,b,g,Au,i,z,f,v,G,x,n,E,Ei,@,y,O,9y,A,e:'
coolest_textgrid_phoneme_set += ',I,t,Y,l,k,a:,w,d,N,r,s,h,p,m'
coolest_textgrid_phoneme_set = coolest_textgrid_phoneme_set.split(',')

coolest_vowels = 'o:,u,Au,i,E,Ei,@,y,O,9y,A,e:,I,a:,Y'.split(',')
coolest_consonants = 'b,g,z,f,v,G,x,n,t,l,k,w,d,N,r,s,h,p,m'.split(',')

    

disc_to_ipa = Mapper().disc_to_ipa
to_ipa_org = {disc: disc_to_ipa[disc] for disc in phonemes}
to_ipa_org['-'] = to_ipa_org['_']
to_ipa_rew = {rewrite_dict.get(key,key):value for key,value in to_ipa_org.items()}
to_ipa = {recode_dict.get(key,key):value for key,value in to_ipa_rew.items()}
to_ipa['gx'] = 'ɣ'
