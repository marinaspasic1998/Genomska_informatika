import numpy as np
from Bio import SeqIO
import os
import pandas as pd


class BWT_FM_Index:
    @staticmethod
    def rotations(t):
        ''' Return list of rotations of input string t '''
        tt = t * 2
        return [tt[i:i + len(t)] for i in range(0, len(t))]

    @staticmethod
    def bwm(t):
        ''' Return lexicographically sorted list of t’s rotations '''
        return sorted(BWT_FM_Index.rotations(t))

    @staticmethod
    def bwtViaBwm(t):
        ''' Given T, returns BWT(T) by way of the BWM '''
        return ''.join(map(lambda x: x[-1], BWT_FM_Index.bwm(t)))

    @staticmethod
    def rankBwt(bw):
        ''' Given BWT string bw, return parallel list of B-ranks. Also
        returns tots: map from character to # times it appears. '''
        tots = dict()
        ranks = []
        for c in bw:
            if c not in tots: tots[c] = 0
            ranks.append(tots[c])
            tots[c] += 1
        return ranks, tots

    @staticmethod
    def lastCol(bw):
        ranks, tots = BWT_FM_Index.rankBwt(bw)
        return list(zip(ranks, bw))

    @staticmethod
    def firstCol(tots):
        ''' Return map from character to the range of rows prefixed by
        the character. '''
        first = {}
        totc = 0
        for c, count in sorted(tots.items()):
            first[c] = totc
            totc += count
        return first

    @staticmethod
    def reverseBwt(bw):
        ''' Make T from BWT(T) '''
        ranks, tots = BWT_FM_Index.rankBwt(bw)
        first = BWT_FM_Index.firstCol(tots)
        rowi = 0
        t = '$'
        while bw[rowi] != '$':
            c = bw[rowi]
            t = c + t
            rowi = first[c] + ranks[rowi]
        return t

    @staticmethod
    def suffixArray(s):
        ''' Take all suffix and sort, while preserving index(offset) info (i) - position of string in the text '''
        satups = sorted([(s[i:], i) for i in range(len(s))])
        return list(map(lambda x: x[1], satups))

    @staticmethod
    def bwtViaSa(t):
        ''' Given T, returns BWT(T) by way of the suffix array. '''
        bw = []
        sa = BWT_FM_Index.suffixArray(t)
        for si in sa:
            if si == 0:
                bw.append('$')
            else:
                bw.append(t[si - 1])
        return ''.join(bw)

    @staticmethod
    def tallyFun(f, bw):
        tally = dict()
        tots = dict()

        for char, row in f.items():
            tots[char] = 0
            tally[char] = {}
            for row in range(len(bw)):
                tally[char][row] = 0

        for row in range(len(bw)):
            char = bw[row]
            tots[char] += 1
            for char in f.keys():
                tally[char][row] = tots[char]

        return tally

    @staticmethod
    def initFM(text):
        bw = BWT_FM_Index.bwtViaSa(text)
        sa = list(BWT_FM_Index.suffixArray(text))
        ranks, tots = BWT_FM_Index.rankBwt(bw)
        F = BWT_FM_Index.firstCol(tots)
        L = BWT_FM_Index.lastCol(bw)
        tally = BWT_FM_Index.tallyFun(F, bw)
        return bw, sa, ranks, tots, F, L, tally

    @staticmethod
    def findPattern(text, pattern):
        bw, sa, ranks, tots, F, L, tally = BWT_FM_Index.initFM(text)
        start = 1
        end = len(bw) - 1

        for i in reversed(range(len(pattern))):
            char = pattern[i]
            if i == (len(pattern) - 1):
                start = F[char]
                takeNext = False
                for j in F.keys():
                    if takeNext:
                        end = F[j] - 1
                        break
                    if j == char:
                        takeNext = True
            else:
                start = F[char] + tally[char][start - 1]
                end = F[char] + tally[char][end] - 1

        positions = []

        for i in range(start, end + 1):
            positions.append(sa[i])
        return positions


class NeedlemanWunsch:
    def __init__(self, scoring_matrix):
        self.scoring_matrix = scoring_matrix

    def globalAlignment(self, x, y):
        D = np.zeros((len(x) + 1, len(y) + 1), dtype=int)

        for i in range(1, len(x) + 1):
            D[i, 0] = D[i - 1, 0] + self.scoring_matrix(x[i - 1], '_')
        for j in range(1, len(y) + 1):
            D[0, j] = D[0, j - 1] + self.scoring_matrix('_', y[j - 1])

        for i in range(1, len(x) + 1):
            for j in range(1, len(y) + 1):
                D[i, j] = max(D[i - 1, j] + self.scoring_matrix(x[i - 1], '_'),
                              D[i, j - 1] + self.scoring_matrix('_', y[j - 1]),
                              D[i - 1, j - 1] + self.scoring_matrix(x[i - 1], y[j - 1]))

        return D, D[len(x), len(y)]

    def traceback(self, x, y, V):
        i = len(x)
        j = len(y)
        ax, ay, am, tr = '', '', '', ''

        while i > 0 or j > 0:
            d, v, h = -100, -100, -100

            if i > 0 and j > 0:
                delta = 1 if x[i - 1] == y[j - 1] else 0
                d = V[i - 1, j - 1] + self.scoring_matrix(x[i - 1], y[j - 1])
            if i > 0:
                v = V[i - 1, j] + self.scoring_matrix(x[i - 1], '_')
            if j > 0:
                h = V[i, j - 1] + self.scoring_matrix('_', y[j - 1])

            if d >= v and d >= h:
                ax += x[i - 1]
                ay += y[j - 1]
                if delta == 1:
                    tr += 'M'
                    am += '|'
                else:
                    tr += 'R'
                    am += ' '
                i -= 1
                j -= 1
            elif v >= h:
                ax += x[i - 1]
                ay += '_'
                tr += 'D'
                am += ' '
                i -= 1
            else:
                ay += y[j - 1]
                ax += '_'
                tr += 'I'
                am += ' '
                j -= 1

        alignment = '\n'.join([ax[::-1], am[::-1], ay[::-1]])
        return alignment, tr[::-1]


class Program:
    @staticmethod
    def reverseComplement(read):
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
        return ''.join(complement[base] for base in reversed(read))

    @staticmethod
    def seedExtend(reference, readId, isReverse, read, seedLength, margin):

        results = []
        seedPattern = read[:seedLength]
        seedPositions = BWT_FM_Index.findPattern(reference, seedPattern)

        for position in seedPositions:

            start = position + seedLength
            end = start + (len(read) - seedLength) + margin

            if end >= len(reference):
                continue

            referencePartToBeAligned = reference[start:end]
            readPartToBeAligned = read[seedLength:]

            nw = NeedlemanWunsch(Program.scoringMatrix)
            D, alignmentScore = nw.globalAlignment(referencePartToBeAligned, readPartToBeAligned)
            alignment, transcript = nw.traceback(referencePartToBeAligned, readPartToBeAligned, D)

            results.append((readId, alignmentScore, isReverse, position, transcript))

        results.sort(key=lambda x: x[1], reverse=True)

        return results

    @staticmethod
    def parseFile(file_name):
        if not os.path.exists(file_name):
            print('File not found')

        file_type = file_name[file_name.rfind('.') + 1:]

        arr = []
        for seq_record in SeqIO.parse(file_name, file_type):
            arr.append([seq_record.id, seq_record.seq])
        return arr

    @staticmethod
    def main(matchP, mismatchP, gapP, seedLength, margin, fasta, fastq):
        reference = Program.parseFile(fasta)[0]
        reads = Program.parseFile(fastq)

        BWT_FM_Index.initFM(reference)

        Program.scoringMatrix = lambda a, b: matchP if a == b else mismatchP if a != '_' and b != '_' else gapP

        bestReadResults = []

        for read in reads:
            readId = read[0]
            readSeq = read[1]
            results = Program.seedExtend(reference[1], readId, False, readSeq, seedLength, margin)
            if len(results) > 0:
                bestReadResults.append(results[0])
            results = Program.seedExtend(reference[1], readId, True, Program.reverseComplement(readSeq), seedLength,
                                         margin)
            if len(results) > 0:
                bestReadResults.append(results[0])

        bestReadResults.sort(key=lambda x: x[1], reverse=True)

        df = pd.DataFrame(bestReadResults,
                          columns=['read_id', 'alignment_score', 'is_reversed', 'position', 'transcript'])
        df.to_csv(
            'results/implementedAligner/output_' + 'M' + str(matchP) + 'MM' + str(mismatchP) + 'G' + str(gapP) + '.csv')


if __name__ == "__main__":
    fasta = 'reads_and_reference\Illumina\example_human_reference.fasta'
    fastq = 'reads_and_reference\Illumina\example_human_Illumina.pe_1.fastq'

    seedLength = 10
    margin = 2

    matchParameters = [1, 2]
    mismatchParameters = [-3, -2]
    gapParameters = [-7, -5]

    for matchP in matchParameters:
        for mismatchP in mismatchParameters:
            for gapP in gapParameters:
                Program.main(matchP, mismatchP, gapP, seedLength, margin, fasta, fastq)
