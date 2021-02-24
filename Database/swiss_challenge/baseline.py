from gensim.summarization import summarizer as textrank
import csv


def read_data():
    reference_summaries_train = []
    source_texts_train = []
    with open('data_train.csv', 'r', encoding = 'utf-8') as f:
        reader = csv.reader(f, delimiter = ',', quoting = csv.QUOTE_ALL)
        next(reader, None)
        for row in reader:
            source_texts_train.append(row[0])
            reference_summaries_train.append(row[1])

    source_texts_test = []
    with open('data_test.csv', 'r', encoding = 'utf-8') as f:
        reader = csv.reader(f, delimiter = ',', quoting = csv.QUOTE_ALL)
        next(reader, None)
        for row in reader:
            source_texts_test.append(row[0])

    return source_texts_train, reference_summaries_train, source_texts_test


def sample_submission(summaries, name):
    with open('' + name + '.csv', 'w', newline = '', encoding = 'utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',', quoting = csv.QUOTE_ALL)
        writer.writerow(['summary'])
        for text in summaries:
            writer.writerow([text])


if __name__ == "__main__":
    # Load train and test data
    # Use source_texts_train and reference_summaries_train for supervised algorithms
    # Use source_texts_test for your submission
    source_texts_train, reference_summaries_train, source_texts_test = read_data()

    # Baseline summary: first 40 tokens
    firstn_summaries = [' '.join(source_text.split()[:40]) for source_text in source_texts_test]
    sample_submission(firstn_summaries, 'FirstN')

    # Baseline summary: textrank
    textrank_summaries = [textrank.summarize(source_text, word_count = 40) for source_text in source_texts_test]
    sample_submission(textrank_summaries, 'Textrank')
