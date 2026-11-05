import pandas as pd, re

path = "data/processed_layer/readings.csv"
df = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
df["_hg"] = df["text_raw"].map(lambda s: len(re.findall(r'[가-힣]', s)))

# [1] 전체 한글 총량 — 323k가 재현되는지
print("text_raw 총 한글 수:", df["_hg"].sum())

# [2] 학습 대상 vs 제외 대상에 한글이 어떻게 분포하나 (가장 중요)
print("\nis_training_text별 한글 총량:")
print(df.groupby("is_training_text")["_hg"].sum())

# [3] 학습 대상인데 한글이 섞인 행이 있는가? (있으면 학습 신호 오염)
train = df[df["is_training_text"] == "True"]
print(f"\n학습 대상 {len(train)}행 중 한글 포함 행: {(train['_hg']>0).sum()}")
print("학습 대상 한글 총량:", train["_hg"].sum())

# [4] 얼마나 버려졌나 — 제외 사유 분포
print("\nquality_flag 분포:", df["quality_flag"].value_counts().to_dict())
print("is_training_text 분포:", df["is_training_text"].value_counts().to_dict())

# [5] '조금만 한글 섞인' 아까운 행이 있나 (한글 1~5자, 한자 다수인데 제외된 행)
lost = df[(df["is_training_text"]=="False") & (df["_hg"].between(1,5)) &
          (df["hanja_count"].astype(int) >= 20)]
print(f"\n한자 20자+인데 한글 소수 때문에 제외됐을 가능성: {len(lost)}행")
print(lost[["inscription_id","hanja_count","_hg","text_raw"]].head(5).to_string())