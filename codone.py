from sklearn.preprocessing import normalize
from numpy.random import choice


class Codone(object):
    codon_table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }

    codon_biases = {}

    def __init__(self, codon_table=None):
        if codon_table is not None:
            self.codon_table = codon_table

    def codon_to_amino(self, codon):
        if codon not in self.codon_table:
            return '_'
        return self.codon_table[codon]

    def split_into_codons(self, sequence):
        # Check that the sequence length is a multiple of 3
        if len(sequence) % 3 != 0:
            print(f"Invalid sequence length. Must be a multiple of 3: {sequence}")
            return None

        # Convert the sequence to uppercase
        sequence = sequence.upper()

        # Split the sequence into codons
        codons = [str(sequence[i:i+3]) for i in range(0, len(sequence), 3)]
        return codons

    def fit(self, genome, weights=None):
        """
        fit(genome, weights=None)

        Calculate codon biases based on genome.

        Parameters
        ----------
        genome : iterable
            Iterable containing coding sequences for proteins in genome. They should be represented as string of DNA nucleotides. Example: `["ACGTGAAAA", "AAAGGGCCCTTT"]`
        weights: iterable, optional, default=None
            Weights for each protein. This parameter is useful to calculate codon biases in highly expressed proteins.
        """

        if weights is None:
            weights = [1.0] * len(genome)
        
        if len(weights) != len(genome):
            raise AttributeError("Genome and weights have to be of same lengths")
        
        for protein, weight in zip(genome, weights):
            for c in self.split_into_codons(protein):
                if not self.codon_to_amino(c) in self.codon_biases:
                    self.codon_biases[self.codon_to_amino(c)] = {}
                if not c in self.codon_biases[self.codon_to_amino(c)]:
                    self.codon_biases[self.codon_to_amino(c)][c] = 0
                self.codon_biases[self.codon_to_amino(c)][c] += 1 * weight
        
        for aa in self.codon_biases.keys():
            # normalize codon biases for each amino acid and codon. Produce dict of aa -> codon -> normalized codon bias
            self.codon_biases[aa] = dict(zip(self.codon_biases[aa].keys(), normalize([list(self.codon_biases[aa].values())], norm='l1')[0]))

    def predict(self, amino_acid_sequence):
        """
        predict(amino_acid_sequence)

        Predicts DNA for amino acid sequence.

        Parameters
        ----------
        amino_acid_sequence : string
            String of single letter amino acid codes. Example: `"MQTYENPSVKYDWWAGNARFANLSGLFIAAHVAQSALIAFWA"`
        """

        dna = ""
        for aa in amino_acid_sequence:
            if aa not in self.codon_biases:
                raise AttributeError(f"Invalid amino acid: {aa}")
            else:
                dna += self.predict_codon(aa)
        return dna

    def predict_codon(self, amino_acid):
        """
        predict_codon(amino_acid)

        Predicts DNA codon for amino acid.

        Parameters
        ----------
        amino_acid : string
            Single letter amino acid code. Example: `"M"`
        """

        return str(choice(list(self.codon_biases[amino_acid].keys()), 1, p=list(self.codon_biases[amino_acid].values()))[0])