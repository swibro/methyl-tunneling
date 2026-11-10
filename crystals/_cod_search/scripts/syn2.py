import os, json, urllib.request, urllib.parse, time, re, collections, sys
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
terms = {
  'NME': (['C H3 N O2'], ['nitromethane']), 'TMSN': (['C4 H12 Sn'], ['tetramethylstannane', 'tetramethyltin']),
  'TMGE': (['C4 H12 Ge'], ['tetramethylgermane']), 'MDBP': (['C15 H24 O'], ['di-tert-butyl-4-methylphenol', 'butylated hydroxytoluene']),
  'PBAC': (['C4 H12 O7 Pb', 'C4 H6 O4 Pb'], ['lead(II) acetate']), 'NH4AC': (['C2 H7 N O2'], ['ammonium acetate']),
  'IT4': (['C7 H7 I'], ['iodotoluene']), 'TMPBCL': (['C3 H9 Cl Pb'], ['trimethyllead']),
  'TMSNCL': (['C3 H9 Cl Sn'], ['trimethyltin']), 'TMSNBR': (['C3 H9 Br Sn'], ['trimethyltin']),
  'DMSNF2': (['C2 H6 F2 Sn'], ['dimethyltin']), 'DMSNCL2': (['C2 H6 Cl2 Sn'], ['dimethyltin']), 'DMSNBR2': (['C2 H6 Br2 Sn'], ['dimethyltin']),
  'TIG': (['C5 H8 O2'], ['tiglic']), 'MEAC': (['C3 H6 O2'], ['methyl acetate']),
  'ZNMPY': (['C12 H14 Cl2 N2 Zn'], ['dichlorobis(4-methylpyridine', 'methylpyridine)zinc']),
  'MGAC': (['C4 H14 Mg O8'], ['magnesium acetate']), 'KAC': (['C2 H3 K O2'], ['potassium acetate']),
  'CDAC': (['C4 H6 Cd O4', 'C4 H10 Cd O6'], ['cadmium acetate']), 'HGAC': (['C4 H6 Hg O4'], ['mercury acetate']),
  'DMS': (['C2 H6 S'], ['dimethyl sulfide', 'dimethyl sulphide', 'dimethylsulfane']),
  'DUR': (['C10 H14'], ['durene', 'tetramethylbenzene']), 'PMB': (['C11 H16'], ['pentamethylbenzene']),
  'OXY': (['C8 H10'], ['o-xylene', 'xylene']), 'MXY': (['C8 H10'], ['m-xylene']), 'PXY': (['C8 H10'], ['p-xylene', '1,4-dimethylbenzene']),
  'HXDY': (['C6 H6'], ['hexadiyne', 'hexa-2,4-diyne']), 'MBUT': (['C5 H10'], ['methyl-2-butene']),
  'TMPZ': (['C8 H12 N2'], ['tetramethylpyrazine']), 'TQ': (['C7 H6 O2'], ['methyl-1,4-benzoquinone', 'methylbenzoquinone']),
  'TMDSB': (['C4 H12 Sb2'], ['tetramethyldistibine']), 'MPY4': (['C6 H7 N'], ['4-methylpyridine', '4-picoline']),
  'ACAC': (['C5 H8 O2'], ['acetylacetone', 'pentane-2,4-dione']), 'TAN': (['C7 H8 S'], ['thioanisole', '(methylsulfanyl)benzene', 'methyl phenyl sulfide']),
  'MECL': (['C H3 Cl'], ['chloromethane']), 'HEP': (['C7 H16'], ['heptane']), 'OCT': (['C8 H18'], ['n-octane', 'octane']),
  'NON': (['C9 H20'], ['nonane']), 'HEX': (['C6 H14'], ['n-hexane', 'hexane']), 'DMAC': (['C4 H6'], ['2-butyne', 'but-2-yne', 'dimethylacetylene']),
  'BUT2': (['C4 H8 O'], ['butanone', 'butan-2-one']), 'TMSI': (['C4 H12 Si'], ['tetramethylsilane']),
  'MTH2': (['C5 H6 S'], ['methylthiophene']), 'MISX': (['C4 H5 N O'], ['methylisoxazole']), 'TDZ': (['C4 H6 N2 S'], ['dimethyl-1,3,4-thiadiazole']),
  'MPZ3': (['C4 H6 N2'], ['3-methylpyrazole', '3-methyl-1H-pyrazole']), 'MIM': (['C4 H6 N2'], ['1-methylimidazole', '1-methyl-1H-imidazole']),
  'ACX': (['C2 H3 Cl O', 'C2 H3 F O', 'C2 H3 Br O'], ['acetyl chloride', 'acetyl fluoride', 'acetyl bromide']), 'MIC': (['C2 H3 N O'], ['methyl isocyanate']),
  'ACE': (['C3 H6 O'], ['acetone', 'propanone']), 'PEN': (['C5 H10 O'], ['pentanone', 'pentan-2-one', 'pentan-3-one']),
  'HXDK': (['C6 H12 O'], ['hexanone', 'hexan-2-one', 'hexan-3-one']), 'PHA': (['C9 H10 O'], ['phenylacetone', 'phenylpropan-2-one']),
  'DCT26': (['C7 H6 Cl2'], ['dichlorotoluene', 'dichloro-2-methylbenzene']), 'DBT26': (['C7 H6 Br2'], ['dibromotoluene', 'dibromo-2-methylbenzene']),
  'BT2': (['C7 H7 Br'], ['bromotoluene', 'bromo-2-methylbenzene']), 'DCTMB': (['C9 H10 Cl2'], ['dichloromesitylene', 'dichloro-2,4,6-trimethylbenzene']),
  'NAAC3': (['C2 H9 Na O5'], ['sodium acetate']), 'MEOH-HQ': (['C19 H22 O7', 'C H4 O'], ['hydroquinone', 'quinol']),
  'TAA': (['C9 H10 O2'], ['tolylacetic', '(4-methylphenyl)acetic', 'p-tolylacetic']), 'MMA': (['C4 H6 O4'], ['methylmalonic', 'methylpropanedioic']),
  'DMMA': (['C5 H8 O4'], ['dimethylmalonic', 'dimethylpropanedioic']), 'PRA': (['C3 H6 O2'], ['propionic', 'propanoic']),
  'CLPR': (['C3 H7 Cl'], ['chloropropane']), 'HMDT': (['C6 H18 Sn2'], ['hexamethyldi', 'hexamethyldistannane']),
}
def parse(f):
  d = collections.Counter()
  for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', (f or '').replace('-', '')):
    d['H' if el == 'D' else el] += int(n) if n else 1
  return d
res = {}
for k, (forms, ts) in terms.items():
  tg = [parse(f) for f in forms]
  hits = {}
  for t in ts:
    url = 'https://www.crystallography.net/cod/result?' + urllib.parse.urlencode({'text': t, 'format': 'json'})
    try:
      d = json.loads(urllib.request.urlopen(url, timeout=90).read() or b'[]')
    except Exception as e:
      print('ERR', k, t, e, flush=True); d = []
    for e in d:
      if parse(e['formula']) in tg:
        hits[e['file']] = e
    time.sleep(0.3)
  res[k] = list(hits.values())
  for e in hits.values():
    print(k, e['file'], e['formula'], '|', (e['chemname'] or '')[:45], '|', (e['commonname'] or '')[:25], '| T=', e['diffrtemp'] or e['celltemp'],
          'P=', e['diffrpressure'], '|', e['flags'], '|', (e['title'] or '')[:60], flush=True)
json.dump(res, open(W + '/syn2.json', 'w'))
print('DONE')
