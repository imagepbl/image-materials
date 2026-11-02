YEAR_FIRST_H2_PRODUCTION = 2015
H2_LAST_EXTERNAL_YEAR = 2030        # external data used up to and including this year
H2_FIRST_IMAGE_YEAR = 2040          # IMAGE data used from this year on (gap is interpolated)
TECHS_SPLIT = ["electrolysis_AEL", "electrolysis_SOEL"]    # sub-technologies of the combined AEL/SOEL group
TECHS_ELECTROLYSIS = ["electrolysis_PEMEL", *TECHS_SPLIT]
TECHS_CBF = ["CBF+CCS_tech1", "CBF_tech1"]

HYDROGEN_TECHNOLOGIES = [
    'ClGs',
    'POOl',
    'SMR',
    'BioGs',
    'ClGsCS',
    'POOlCS',
    'SMRCS',
    'BioGsCS',
    'BLEl',
    'CurEl',
    'SPVEl',
    'CSPEl',
    'WONEl',
    'WOFFEl',
    'GEOEl',
    'ssSMR',
    'ssEl']

ELECTROLYSIS_TECHNOLOGIES = [
    'BLEl',
    'CurEl',
    'SPVEl',
    'CSPEl',
    'WONEl',
    'WOFFEl',
    'GEOEl',
    'ssEl'
]

COLOURS_HYDROGEN = {
    'ClGs':   '#1B9E77',
    'POOl':   '#D95F02',
    'SMR':    '#7570B3',
    'BioGs':  '#66A61E',
    'ClGsCS': '#66C2A5',
    'POOlCS': '#FC8D62',
    'SMRCS':  '#8DA0CB',
    'BioGsCS':'#A6D854',
    'BLEl':   '#E6AB02',
    'CurEl':  '#E7298A',
    'SPVEl':  '#A6761D',
    'CSPEl':  '#1F78B4',
    'WONEl':  '#B2DF8A',
    'WOFFEl': '#FDBF6F',
    'GEOEl':  '#CAB2D6',
    'ssSMR':  '#6A3D9A',
    'ssEl':   '#B15928',
}