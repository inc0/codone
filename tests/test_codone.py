# write fixture to point to root directory of project
# https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions
import pytest
from Bio import SeqIO
from codone import Codone
import numpy as np


def test_fit_predict():
    """
        Basic usage of library. Fit on FASTA coding sequences and predict codons for amino acid.
    """

    # Example using A. platensis genome and coding sequences
    input_fasta = "tests/data/aplatensis-fasta-coding.faa"

    with open(input_fasta, "r") as f:
        seq = list(SeqIO.parse(input_fasta, "fasta"))
    genome = [str(s.seq) for s in seq]

    codone = Codone()
    codone.fit(genome)

    # Codon biases contain map of codons and their biases for each amino acid
    assert codone.codon_biases['M']['ATG'] == 1.0
    assert codone.codon_biases['K']['AAA'] >= 0.73

    np.random.seed(seed=42)

    # Predict codons for amino acid sequence noted as string of single letter amino acid codes
    protein = "MQTYENPSVKYDWWAGNARFANLSGLFIAAHVAQSALIAFWAKKKK"
    dna = codone.predict(protein)
    assert dna == "ATGCAAACTTATGAGAATCCATCCGTAAAATATGATTGGTGGGCCGGGAATGCTCGCTTTGCTAATTTATCAGGTCTGTTCATTGCTGCCCATGTTGCCCAATCCGCTTTAATAGCTTTTTGGGCTAAGAAAAAGAAA"
