import os

def main():
    PhraseId = set()
    SentenceId = set()
    Phrase = set()
    Sentiment = set()

    zero, one, two, three, four = 0, 0, 0, 0, 0

    for line in open("../data/train.tsv", "r"):
        log = line.strip('\n').split('\t')

        PhraseId.add(log[0])
        SentenceId.add(log[1])
        Phrase.add(log[2])
        Sentiment.add(log[3])

        if log[3] == '0': zero += 1
        elif log[3] == '1': one += 1
        elif log[3] == '2': two += 1
        elif log[3] == '3': three += 1
        elif log[3] == '4': four += 1

    print("Number of PhraseId :", len(PhraseId))
    print("Number of SentenceId :", len(SentenceId))
    print("Number of Phrase :", len(Phrase))
    print("Number of Sentiment :", len(Sentiment))
    print(zero, one, two, three, four)

if __name__ == '__main__':
    main()
