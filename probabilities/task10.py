import pandas as pd


def get_titatic_dataframe() -> pd.DataFrame:
    df = pd.read_csv("train.csv")
    return df


def get_filled():
    df = get_titatic_dataframe()

    def extract_title(name):
        title_markers = ["Mr.", " Mrs.", " Miss."]
        for marker in title_markers:
            if marker in name:
                return marker.strip()
        return None

    df['Title'] = df['Name'].apply(extract_title)

    median_values = df.groupby('Title')['Age'].median().round().to_dict()

    mr_ages = df.loc[df["Title"] == "Mr.", 'Age']
    nan_mr = mr_ages.isna().sum()

    mrs_ages = df.loc[df["Title"] == "Mrs.", 'Age']
    nan_mrs = mrs_ages.isna().sum()

    miss_ages = df.loc[df["Title"] == "Miss.", 'Age']
    nan_miss = miss_ages.isna().sum()

    missing_values = {"Mr.": nan_mr, "Mrs.": nan_mrs, "Miss.": nan_miss}
    result = [(title, missing_values[title], int(median_values[title])) for title in ["Mr.", "Mrs.", "Miss."]]

    return result


result = get_filled()
print(result)