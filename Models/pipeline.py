# Imports
from data_loader import DataLoader
from rouge import Rouge


# Global Variables
META_PATH = "C:\\Temp\\Corpus\\meta_files\\"
TEXT_PATH = "C:\\Temp\\Corpus\\text_files\\"
SUMMARY_PATH = "C:\\Temp\\Corpus\\summary_files\\"


# Methods


# Main
def main():
    global META_PATH
    global TEXT_PATH
    global SUMMARY_PATH

    ''' DATA-SECTION '''
    corpus_filter = ["wikihow"] # ["cnn_dailymail", "wikihow", "tldr"]
    instance = DataLoader(META_PATH, TEXT_PATH, SUMMARY_PATH, corpus_filter)
    X_train, y_train, X_test, y_test = instance.train_test_split(size=0.75, shuffle=True)

    print(X_train)
    print(y_train)

    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))

    exit()

    ''' MODEL-SECTION '''
    model.train(X_train, y_train)
    y_hyps = model.predict(X_test)
    y_refs = y_test
    exit()

    ''' SCORE-SECTION '''
    instance = Rouge()
    scores = instance.get_scores(y_hyps, y_refs, avg=True)

    print(scores)

    exit()


if __name__ == "__main__":
    main()
