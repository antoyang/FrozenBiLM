import pandas as pd
from args import DATA_DIR

train_df = pd.read_csv(f"{DATA_DIR}/WebVid/results_10M_train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/WebVid/results_2M_val.csv")
train = pd.DataFrame(
    {
        "text": list(train_df["name"]),
        "video_id": list(train_df["videoid"]),
    },
    columns=["text", "video_id"],
)
val = pd.DataFrame(
    {
        "text": list(val_df["name"]),
        "video_id": list(val_df["videoid"]),
    },
    columns=["text", "video_id"],
)
train.to_csv(f"{DATA_DIR}/WebVid/train_captions.csv")
val.to_csv(f"{DATA_DIR}/WebVid/val_captions.csv")
