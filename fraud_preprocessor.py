from __future__ import annotations

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_COL = "is_fraud"
DROP_COLS = ["first", "last", "street", "trans_num", "cc_num", "Unnamed: 0"]

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (sin(dlat / 2) ** 2) + cos(lat1) * cos(lat2) * (sin(dlon / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

class FraudPreprocessor:
    def __init__(self):
        self.encoders: dict[str, LabelEncoder] = {}
        self.unix_scaler = StandardScaler()
        self.zip_fraud_rate: pd.DataFrame | None = None
        self.feature_names: list[str] | None = None
        self._is_fitted = False

    def _require_fitted(self):
        if not self._is_fitted or self.feature_names is None:
            raise RuntimeError("Preprocessor not fitted. Call .fit(train_df) first.")

    @staticmethod
    def _ensure_unknown(le: LabelEncoder, values: pd.Series) -> LabelEncoder:
        s = values.astype(str).fillna("UNKNOWN")
        if "UNKNOWN" not in set(s.values):
            s = pd.concat([s, pd.Series(["UNKNOWN"])], ignore_index=True)
        le.fit(s)
        return le

    def _fit_encode_col(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        df = df.copy()
        le = LabelEncoder()
        le = self._ensure_unknown(le, df[col])
        self.encoders[col] = le
        df[col] = le.transform(df[col].astype(str).fillna("UNKNOWN"))
        return df

    def _transform_encode_col(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        df = df.copy()
        le = self.encoders[col]
        s = df[col].astype(str).fillna("UNKNOWN")
        known = set(le.classes_)
        s = s.apply(lambda x: x if x in known else "UNKNOWN")
        df[col] = le.transform(s)
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "trans_date_trans_time" not in df.columns:
            return df

        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        df["year"] = df["trans_date_trans_time"].dt.year
        df["month"] = df["trans_date_trans_time"].dt.month
        df["day"] = df["trans_date_trans_time"].dt.day
        df["hour"] = df["trans_date_trans_time"].dt.hour
        df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
        df = df.drop(columns=["trans_date_trans_time"])
        return df

    def _add_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        needed = {"lat", "long", "merch_lat", "merch_long"}
        if needed.issubset(df.columns):
            df["distance_km"] = df.apply(
                lambda r: haversine(r["lat"], r["long"], r["merch_lat"], r["merch_long"]),
                axis=1,
            )
        else:
            df["distance_km"] = 0.0
        return df

    def fit(self, train_df: pd.DataFrame) -> "FraudPreprocessor":
        df = train_df.copy()

        if TARGET_COL not in df.columns:
            raise ValueError(f"train_df must include '{TARGET_COL}' to fit zip_fraud_rate.")

        # 1) Drop columns
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

        # 2) Time features
        df = self._add_time_features(df)

        # 3) Encode merchant first
        if "merchant" in df.columns:
            df = self._fit_encode_col(df, "merchant")

        # 4) Create merchant_category_combo using encoded merchant + raw category string
        if "merchant" in df.columns and "category" in df.columns:
            df["merchant_category_combo"] = df["merchant"].astype(str) + "_" + df["category"].astype(str)

        # 5) Distance
        df = self._add_distance(df)

        # 6) Encode category
        if "category" in df.columns:
            df = self._fit_encode_col(df, "category")

        # 7) zip_fraud_rate + merge
        if "zip" in df.columns:
            self.zip_fraud_rate = (
                df.groupby("zip")[TARGET_COL]
                .mean()
                .reset_index()
                .rename(columns={TARGET_COL: "zip_fraud_rate"})
            )
            df = df.merge(self.zip_fraud_rate, on="zip", how="left")
        else:
            self.zip_fraud_rate = pd.DataFrame(columns=["zip", "zip_fraud_rate"])
            df["zip_fraud_rate"] = 0.0

        # 8) zip_encoded
        if "zip" in df.columns:
            df = self._fit_encode_col(df, "zip")
            df = df.rename(columns={"zip": "zip_encoded"})

        # 9) log1p amount
        if "amt" in df.columns:
            df["amt"] = np.log1p(df["amt"])

        # 10) Encode remaining object columns
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        obj_cols = [c for c in obj_cols if c != TARGET_COL]
        for col in obj_cols:
            df = self._fit_encode_col(df, col)

        # 11) Scale unix_time
        if "unix_time" in df.columns:
            df["unix_time"] = self.unix_scaler.fit_transform(df[["unix_time"]])

        X = df.drop(columns=[TARGET_COL], errors="ignore")
        self.feature_names = list(X.columns)
        self._is_fitted = True
        return self

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        df = raw_df.copy()

        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

        df = self._add_time_features(df)

        if "merchant" in df.columns and "merchant" in self.encoders:
            df = self._transform_encode_col(df, "merchant")

        if "merchant" in df.columns and "category" in df.columns:
            df["merchant_category_combo"] = df["merchant"].astype(str) + "_" + df["category"].astype(str)

        df = self._add_distance(df)

        if "category" in df.columns and "category" in self.encoders:
            df = self._transform_encode_col(df, "category")

        if "zip" in df.columns and self.zip_fraud_rate is not None:
            df = df.merge(self.zip_fraud_rate, on="zip", how="left")
            df["zip_fraud_rate"] = df["zip_fraud_rate"].fillna(0.0)
        else:
            if "zip_fraud_rate" not in df.columns:
                df["zip_fraud_rate"] = 0.0

        if "zip" in df.columns and "zip" in self.encoders:
            df = self._transform_encode_col(df, "zip")
            df = df.rename(columns={"zip": "zip_encoded"})
        elif "zip_encoded" not in df.columns:
            df["zip_encoded"] = 0

        if "amt" in df.columns:
            df["amt"] = np.log1p(df["amt"])

        for col in self.encoders.keys():
            if col in {"merchant", "category", "zip"}:
                continue
            if col in df.columns:
                df = self._transform_encode_col(df, col)

        if "unix_time" in df.columns:
            df["unix_time"] = self.unix_scaler.transform(df[["unix_time"]])

        X = df.reindex(columns=self.feature_names, fill_value=0)
        if TARGET_COL in X.columns:
            X = X.drop(columns=[TARGET_COL])
        return X
