import pandas as pd


def get_submission_file(y_test, path):
    submission = pd.DataFrame(y_test, columns=['item_cnt_month']).reset_index()
    submission.to_csv(path, header=['ID', 'item_cnt_month'], index=False)
    return submission
