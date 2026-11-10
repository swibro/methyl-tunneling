"""Download the selected COD structures into crystals/ and write selection tables."""
import csv, json, os, re, time, urllib.request, zipfile, xml.etree.ElementTree as ET

REPO = '/home/devuser/workspace/methyl'
OUT = os.path.join(REPO, 'crystals')
XLSX = os.path.join(REPO, 'methyl_benchmark.xlsx')

# Crystal_ID: (COD id, quality, note, alternatives)
# quality: good = low-T/ambient-P, ordered; caveat = usable but see note; weak = last resort
SEL = {
  'TOL':    ('7238224', 'good',   'alpha phase P21/c, 150 K', ''),
  'MEF':    ('2101599', 'caveat', 'CD3F, neutron powder (Ibberson & Prager 1996); T not in CIF, low-T vapour-deposited sample - check paper', ''),
  'MECL':   ('7216469', 'caveat', 'only high-pressure data (1.22 GPa, 295 K); Cmc21 = the low-T ambient phase; relax cell at 0 GPa', '7216470-7216472 (higher P)'),
  'MEBR':   ('7216473', 'caveat', 'only high-pressure data (1.11 GPa, 295 K); Pnma matches Prager; methyl H disordered', '7216474 (1.5 GPa)'),
  'MEI':    ('7201032', 'caveat', 'same structure as ../methyl-iodide.cif: 0.53 GPa, 295 K, methyl H disordered', '7201033, 7201034 (higher P)'),
  'IET':    ('2109086', 'good',   '100 K ambient', '2109085 (120 K), 2109084 (140 K)'),
  'DEC':    ('2200853', 'good',   '150 K', ''),
  'UND':    ('7247707', 'caveat', 'pressure-frozen, 0.17 GPa, 295 K; P-1 (ambient low-T phase of odd n-alkanes is orthorhombic - check)', ''),
  'DOD':    ('7247706', 'caveat', 'pressure-frozen, 0.15 GPa, 295 K; P-1', ''),
  'DMAC':   ('2101248', 'caveat', 'C4D6, neutron powder (Ibberson & Prager 1995); T not in CIF - check paper', ''),
  'ACE':    ('7110465', 'good',   'acetone-d6, 5 K neutron, Pbca', '7110464 (150 K X-ray, Pbca)'),
  'DIAC':   ('5000194', 'caveat', '1983 X-ray; paper reports -12 and -100 C, T not in CIF; P21/n matches Prager', ''),
  'PBO':    ('2009336', 'good',   '173 K', ''),
  'ACP':    ('5000166', 'good',   '154 K (from title), P21/n matches Prager', ''),
  'DME':    ('1570433', 'weak',   'only high-pressure data (3.3 GPa, 295 K)', '1570434-1570438 (higher P)'),
  'DPE':    ('1570448', 'weak',   'only high-pressure data (1.7 GPa, 295 K)', '1570449-1570452 (higher P)'),
  'ACA':    ('2008187', 'good',   '40 K', '2300042 (293 K)'),
  'PRA':    ('5000204', 'caveat', '1962 X-ray (Strieter & Templeton); T not in CIF', ''),
  'ASP':    ('7050900', 'good',   '20 K, form I', '7050903 (60 K)'),
  'MMA':    ('5900038', 'weak',   '1949 room-T, no H atoms in CIF - H must be added', ''),
  'TOL3A':  ('2229512', 'caveat', '123 K; partially occupied (disordered) sites', ''),
  'TOL4A':  ('4500433', 'good',   '100 K', '7233576 (100 K)'),
  'LIAC':   ('4500141', 'caveat', '100 K; partially occupied sites', ''),
  'MNAC':   ('2103558', 'good',   '14 K neutron (D2O-deuterated water), P21/c matches Prager', '2103559 (300 K)'),
  'NAAC':   ('1549467', 'caveat', '100 K, Pcca; polymorph measured by P&H unknown', '1549468 (P4/nmm, 100 K), 6000208 (beta, no coordinates)'),
  'ZNAC':   ('2006420', 'good',   'data at 19 K (cell at 292 K)', '2016971 (110 K)'),
  'NIAC':   ('7037637', 'good',   '173 K', ''),
  'CUAC':   ('2108330', 'good',   '100 K', '2202437 (293 K)'),
  'COAC':   ('8101209', 'caveat', '173 K, catena dihydrate; check that it is the phase P&H measured', '2202935 (tetrahydrate, 153 K)'),
  'MUR':    ('7209247', 'good',   '8 K', ''),
  'ACM':    ('9014907', 'good',   '23 K neutron, R3c matches Prager', '9007523 (room T)'),
  'PARA':   ('2104364', 'good',   '100 K form I (P21/n = Prager P21/a)', '2007205 (150 K)'),
  'MMAM':   ('2011375', 'caveat', 'workbook flags identity (Check); this is methylmalonamide, 296 K', ''),
  'TOL-CALIX': ('2103461', 'caveat', '173 K; toluene guest disordered', ''),
  'FT2':    ('4507556', 'good',   '205 K, in situ crystallised, Pbca', '4507555 (296 K, P21/n, likely high-P)'),
  'FT3':    ('4507558', 'good',   '179 K', ''),
  'FT4':    ('4507560', 'good',   '217 K', ''),
  'HMB':    ('2100190', 'good',   'C12D18, 5 K', ''),
  'TCM':    ('7117264', 'good',   '118 K', '2101374 (1995, 150/297 K)'),
  'TBM':    ('7117259', 'good',   '118 K', ''),
  'TIM':    ('7117269', 'good',   '118 K', '2012171 (293 K)'),
  'DBTMB':  ('2013419', 'caveat', '14 K neutron; methyl H disordered', '2013420 (120 K)'),
  'MPY3':   ('2200598', 'good',   '120 K', ''),
  'LUT35':  ('2200844', 'good',   '150 K', ''),
  'LUT26':  ('2200702', 'good',   '120 K', ''),
  'MPZ4':   ('7051396', 'good',   '100 K', ''),
  'MFU2':   ('7232754', 'good',   '100 K', ''),
  'DMPZ25': ('7051298', 'good',   '183 K', ''),
  'TMPB':   ('2202247', 'good',   '150 K, Pa-3 matches Prager', ''),
  'TMSI':   ('2104690', 'caveat', '100 K, Pnma, R = 0.12; partially occupied sites', '4503063 (Pa-3, 293 K)'),
  'TMSNF':  ('7036132', 'caveat', '153 K Cmcm; Prager lists Pnma (= 296 K entry); methyl H disordered', '7036131 (296 K, Pnma)'),
  'TMBI':   ('4081633', 'good',   '150 K', ''),
}

def read_crystals():
  ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
  z = zipfile.ZipFile(XLSX)
  ss = [''.join(t.text or '' for t in si.iter('{%s}t' % ns['m']))
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', ns)]
  wb = ET.fromstring(z.read('xl/workbook.xml'))
  rels = {r.get('Id'): r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
  rid = [s for s in wb.find('m:sheets', ns) if s.get('name') == 'Crystals'][0].get(
    '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
  tgt = rels[rid].lstrip('/'); tgt = tgt if tgt.startswith('xl/') else 'xl/' + tgt
  rows = []
  for row in ET.fromstring(z.read(tgt)).iter('{%s}row' % ns['m']):
    d = {}
    for c in row.findall('m:c', ns):
      v = c.find('m:v', ns)
      col = re.match(r'[A-Z]+', c.get('r')).group(0)
      d[col] = '' if v is None else (ss[int(v.text)] if c.get('t') == 's' else v.text)
    rows.append(d)
  return rows[1:]

def cod_meta(i):
  url = f'https://www.crystallography.net/cod/result?id={i}&format=json'
  return json.loads(urllib.request.urlopen(url, timeout=60).read())[0]

crys = read_crystals()
os.makedirs(OUT, exist_ok=True)
fields = ['crystal_id', 'compound', 'priority', 'cod_id', 'file', 'quality', 'T_K', 'P_GPa', 'space_group', 'prager_sg',
          'R_obs', 'radiation', 'formula', 'year', 'doi', 'note', 'alternatives_cod']
sel_rows, miss_rows = [], []
for c in crys:
  cid = c.get('A', '')
  if not cid or c.get('E') == 'Out':
    continue
  if cid not in SEL:
    miss_rows.append({'crystal_id': cid, 'compound': c.get('B', ''), 'priority': c.get('G', ''),
                      'formula': c.get('K', ''), 'prager_sg': c.get('M', ''), 'scope': c.get('E', '')})
    continue
  cod, q, note, alt = SEL[cid]
  m = cod_meta(cod)
  fn = f'{cid}_COD{cod}.cif'
  data = urllib.request.urlopen(f'https://www.crystallography.net/cod/{cod}.cif', timeout=60).read()
  assert b'_atom_site_fract_x' in data, cod
  with open(os.path.join(OUT, fn), 'wb') as f:
    f.write(data)
  p = m.get('diffrpressure') or m.get('cellpressure')
  sel_rows.append({'crystal_id': cid, 'compound': c.get('B', ''), 'priority': c.get('G', ''), 'cod_id': cod, 'file': fn,
                   'quality': q, 'T_K': m.get('diffrtemp') or m.get('celltemp') or '',
                   'P_GPa': f'{float(p) / 1e6:g}' if p else '', 'space_group': m.get('sg'), 'prager_sg': c.get('M', ''),
                   'R_obs': m.get('Robs') or m.get('Rall') or '', 'radiation': m.get('radType') or m.get('radiation') or '',
                   'formula': m.get('formula', '').strip('- '), 'year': m.get('year'), 'doi': m.get('doi') or '',
                   'note': note, 'alternatives_cod': alt})
  print(cid, cod, q, flush=True)
  time.sleep(0.3)
with open(os.path.join(OUT, 'selection.csv'), 'w', newline='') as f:
  w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(sel_rows)
with open(os.path.join(OUT, 'not_in_cod.csv'), 'w', newline='') as f:
  w = csv.DictWriter(f, fieldnames=['crystal_id', 'compound', 'priority', 'formula', 'prager_sg', 'scope'])
  w.writeheader(); w.writerows(miss_rows)
print(len(sel_rows), 'downloaded;', len(miss_rows), 'not in COD')
