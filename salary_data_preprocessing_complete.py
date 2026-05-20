import re
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================

BASE_DIR = Path(".")
RAW_DIR = BASE_DIR / "data_raw"
PROCESSED_DIR = BASE_DIR / "data_processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

WAGE_PATH = RAW_DIR / "national_M2024_dl.xlsx"
SKILLS_PATH = RAW_DIR / "Skills.txt"
ABILITIES_PATH = RAW_DIR / "Abilities.txt"
TECH_PATH = RAW_DIR / "Technology Skills.txt"
FINAL_CSV_PATH = PROCESSED_DIR / "final_salary_prediction_dataset.csv"


# =========================================================
# 2. MYSQL CONFIG - SỬA PHẦN NÀY CHO ĐÚNG MÁY CỦA BẠN
# =========================================================

MYSQL_USER = "root"
MYSQL_PASSWORD = "password"  # đổi thành mật khẩu MySQL của bạn
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DATABASE = "salary_prediction_db"


# =========================================================
# 3. HÀM HỖ TRỢ
# =========================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def check_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}\n"
            f"Hãy kiểm tra file có nằm trong thư mục data_raw không."
        )


def clean_feature_name(text_value) -> str:
    """Làm sạch tên biến để dùng được trong Python/MySQL."""
    text_value = str(text_value).strip()
    text_value = re.sub(r"[^A-Za-z0-9]+", "_", text_value)
    text_value = re.sub(r"_+", "_", text_value)
    text_value = text_value.strip("_")
    return text_value


def standardize_onet_soc_code(series: pd.Series) -> pd.Series:
    """Chuyển O*NET-SOC Code từ dạng 11-1011.00 thành 11-1011."""
    return series.astype(str).str.strip().str[:7]


def convert_to_numeric(series: pd.Series) -> pd.Series:
    """Chuyển chuỗi sang số; ký hiệu như *, #, ** sẽ thành NaN."""
    return pd.to_numeric(series, errors="coerce")


def make_mysql_engine():
    """Tạo kết nối MySQL bằng SQLAlchemy + PyMySQL."""
    password_encoded = quote_plus(MYSQL_PASSWORD)
    connection_url = (
        f"mysql+pymysql://{MYSQL_USER}:{password_encoded}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
        f"?charset=utf8mb4"
    )
    return create_engine(connection_url)


def test_mysql_connection(engine) -> None:
    """Kiểm tra kết nối MySQL."""
    with engine.connect() as conn:
        current_db = conn.execute(text("SELECT DATABASE();")).scalar()
    print(f"Kết nối MySQL thành công. Database hiện tại: {current_db}")


# =========================================================
# 4. ĐỌC DỮ LIỆU GỐC
# =========================================================

def load_raw_data():
    print_section("BƯỚC 1: KIỂM TRA VÀ ĐỌC DỮ LIỆU GỐC")
    for path in [WAGE_PATH, SKILLS_PATH, ABILITIES_PATH, TECH_PATH]:
        check_file_exists(path)

    wage = pd.read_excel(WAGE_PATH)
    skills = pd.read_csv(SKILLS_PATH, sep="\t")
    abilities = pd.read_csv(ABILITIES_PATH, sep="\t")
    technology = pd.read_csv(TECH_PATH, sep="\t")

    print("Wage data shape:", wage.shape)
    print("Skills data shape:", skills.shape)
    print("Abilities data shape:", abilities.shape)
    print("Technology data shape:", technology.shape)
    return wage, skills, abilities, technology


# =========================================================
# 5. XỬ LÝ DỮ LIỆU LƯƠNG
# =========================================================

def process_wage_data(wage: pd.DataFrame) -> pd.DataFrame:
    print_section("BƯỚC 2: XỬ LÝ DỮ LIỆU LƯƠNG")
    wage_clean = wage.copy()

    required_cols = ["OCC_CODE", "OCC_TITLE", "O_GROUP", "TOT_EMP", "A_MEDIAN", "A_MEAN"]
    missing_cols = [col for col in required_cols if col not in wage_clean.columns]
    if missing_cols:
        raise ValueError(f"File lương thiếu các cột bắt buộc: {missing_cols}")

    wage_clean = wage_clean[wage_clean["O_GROUP"] == "detailed"].copy()
    wage_clean = wage_clean[["OCC_CODE", "OCC_TITLE", "TOT_EMP", "A_MEDIAN", "A_MEAN"]].copy()

    wage_clean = wage_clean.rename(columns={
        "OCC_CODE": "soc_code",
        "OCC_TITLE": "occupation_title",
        "TOT_EMP": "total_employment",
        "A_MEDIAN": "annual_median_wage",
        "A_MEAN": "annual_mean_wage"
    })

    wage_clean["soc_code"] = wage_clean["soc_code"].astype(str).str.strip()

    for col in ["total_employment", "annual_median_wage", "annual_mean_wage"]:
        wage_clean[col] = convert_to_numeric(wage_clean[col])

    before_drop = wage_clean.shape[0]
    wage_clean = wage_clean.dropna(subset=["annual_median_wage"]).copy()
    after_drop = wage_clean.shape[0]
    wage_clean = wage_clean.drop_duplicates(subset=["soc_code"]).copy()

    print("Số dòng trước khi xóa thiếu lương:", before_drop)
    print("Số dòng sau khi xóa thiếu lương:", after_drop)
    print("Wage clean shape:", wage_clean.shape)
    return wage_clean


# =========================================================
# 6. XỬ LÝ SKILLS
# =========================================================

def process_skills_data(skills: pd.DataFrame) -> pd.DataFrame:
    print_section("BƯỚC 3: XỬ LÝ DỮ LIỆU SKILLS")
    skills_clean = skills.copy()

    required_cols = ["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"]
    missing_cols = [col for col in required_cols if col not in skills_clean.columns]
    if missing_cols:
        raise ValueError(f"File Skills thiếu các cột bắt buộc: {missing_cols}")

    skills_clean["soc_code"] = standardize_onet_soc_code(skills_clean["O*NET-SOC Code"])
    skills_clean = skills_clean[["soc_code", "Element Name", "Scale ID", "Data Value"]].copy()
    skills_clean["Data Value"] = convert_to_numeric(skills_clean["Data Value"])

    skills_clean["feature_name"] = (
        skills_clean["Element Name"].apply(clean_feature_name)
        + "_"
        + skills_clean["Scale ID"].astype(str).str.strip()
    )

    skills_wide = skills_clean.pivot_table(
        index="soc_code",
        columns="feature_name",
        values="Data Value",
        aggfunc="mean"
    ).reset_index()

    skills_wide = skills_wide.rename(
        columns={col: "skill_" + col for col in skills_wide.columns if col != "soc_code"}
    )

    print("Skills wide shape:", skills_wide.shape)
    return skills_wide


# =========================================================
# 7. XỬ LÝ ABILITIES
# =========================================================

def process_abilities_data(abilities: pd.DataFrame) -> pd.DataFrame:
    print_section("BƯỚC 4: XỬ LÝ DỮ LIỆU ABILITIES")
    abilities_clean = abilities.copy()

    required_cols = ["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"]
    missing_cols = [col for col in required_cols if col not in abilities_clean.columns]
    if missing_cols:
        raise ValueError(f"File Abilities thiếu các cột bắt buộc: {missing_cols}")

    abilities_clean["soc_code"] = standardize_onet_soc_code(abilities_clean["O*NET-SOC Code"])
    abilities_clean = abilities_clean[["soc_code", "Element Name", "Scale ID", "Data Value"]].copy()
    abilities_clean["Data Value"] = convert_to_numeric(abilities_clean["Data Value"])

    abilities_clean["feature_name"] = (
        abilities_clean["Element Name"].apply(clean_feature_name)
        + "_"
        + abilities_clean["Scale ID"].astype(str).str.strip()
    )

    abilities_wide = abilities_clean.pivot_table(
        index="soc_code",
        columns="feature_name",
        values="Data Value",
        aggfunc="mean"
    ).reset_index()

    abilities_wide = abilities_wide.rename(
        columns={col: "ability_" + col for col in abilities_wide.columns if col != "soc_code"}
    )

    print("Abilities wide shape:", abilities_wide.shape)
    return abilities_wide


# =========================================================
# 8. XỬ LÝ TECHNOLOGY SKILLS
# =========================================================

def process_technology_data(technology: pd.DataFrame) -> pd.DataFrame:
    print_section("BƯỚC 5: XỬ LÝ DỮ LIỆU TECHNOLOGY SKILLS")
    tech_clean = technology.copy()

    required_cols = ["O*NET-SOC Code", "Example", "Commodity Title", "Hot Technology", "In Demand"]
    missing_cols = [col for col in required_cols if col not in tech_clean.columns]
    if missing_cols:
        raise ValueError(f"File Technology Skills thiếu các cột bắt buộc: {missing_cols}")

    tech_clean["soc_code"] = standardize_onet_soc_code(tech_clean["O*NET-SOC Code"])
    tech_clean["example_lower"] = tech_clean["Example"].astype(str).str.lower()

    tech_clean["has_python"] = tech_clean["example_lower"].str.contains(r"\bpython\b", na=False, regex=True).astype(int)
    tech_clean["has_sql"] = tech_clean["example_lower"].str.contains(r"\bsql\b|structured query language", na=False, regex=True).astype(int)
    tech_clean["has_excel"] = tech_clean["example_lower"].str.contains(r"\bexcel\b", na=False, regex=True).astype(int)
    tech_clean["has_tableau"] = tech_clean["example_lower"].str.contains(r"\btableau\b", na=False, regex=True).astype(int)
    tech_clean["has_r"] = tech_clean["example_lower"].str.fullmatch(r"\s*r\s*", na=False).astype(int)

    tech_clean["hot_technology_num"] = (
        tech_clean["Hot Technology"].astype(str).str.upper().str.strip() == "Y"
    ).astype(int)
    tech_clean["in_demand_num"] = (
        tech_clean["In Demand"].astype(str).str.upper().str.strip() == "Y"
    ).astype(int)

    tech_features = tech_clean.groupby("soc_code").agg(
        num_technology_skills=("Example", "nunique"),
        num_technology_categories=("Commodity Title", "nunique"),
        num_hot_technology=("hot_technology_num", "sum"),
        num_in_demand_technology=("in_demand_num", "sum"),
        has_python=("has_python", "max"),
        has_sql=("has_sql", "max"),
        has_excel=("has_excel", "max"),
        has_tableau=("has_tableau", "max"),
        has_r=("has_r", "max")
    ).reset_index()

    print("Technology features shape:", tech_features.shape)
    return tech_features


# =========================================================
# 9. NỐI TẤT CẢ BẢNG
# =========================================================

def merge_all_data(wage_clean, skills_wide, abilities_wide, tech_features) -> pd.DataFrame:
    print_section("BƯỚC 6: NỐI TẤT CẢ BẢNG DỮ LIỆU")

    final_data = wage_clean.merge(skills_wide, on="soc_code", how="left")
    final_data = final_data.merge(abilities_wide, on="soc_code", how="left")
    final_data = final_data.merge(tech_features, on="soc_code", how="left")

    print("Final data shape sau khi nối:", final_data.shape)
    return final_data


# =========================================================
# 10. XỬ LÝ MISSING VALUES
# =========================================================

def clean_final_dataset(final_data: pd.DataFrame) -> pd.DataFrame:
    print_section("BƯỚC 7: XỬ LÝ GIÁ TRỊ THIẾU VÀ KIỂM TRA CUỐI")
    final_data = final_data.copy()

    duplicate_count = final_data.duplicated(subset=["soc_code"]).sum()
    print("Số dòng trùng soc_code:", duplicate_count)
    if duplicate_count > 0:
        final_data = final_data.drop_duplicates(subset=["soc_code"]).copy()

    tech_cols = [
        "num_technology_skills", "num_technology_categories", "num_hot_technology",
        "num_in_demand_technology", "has_python", "has_sql", "has_excel", "has_tableau", "has_r"
    ]
    for col in tech_cols:
        if col in final_data.columns:
            final_data[col] = final_data[col].fillna(0)

    numeric_cols = final_data.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if final_data[col].isnull().sum() > 0:
            final_data[col] = final_data[col].fillna(final_data[col].mean())

    print("Final data shape cuối cùng:", final_data.shape)
    return final_data


# =========================================================
# 11. LƯU CSV
# =========================================================

def save_csv(final_data: pd.DataFrame) -> None:
    print_section("BƯỚC 8: LƯU FILE CSV")
    final_data.to_csv(FINAL_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"Đã lưu file CSV: {FINAL_CSV_PATH}")


# =========================================================
# 12. LƯU DỮ LIỆU VÀO MYSQL
# =========================================================

def save_to_mysql(wage_clean, skills_wide, abilities_wide, tech_features, final_data) -> None:
    print_section("BƯỚC 9: LƯU CÁC BẢNG VÀO MYSQL")

    engine = make_mysql_engine()
    test_mysql_connection(engine)

    wage_clean.to_sql("wage_data", con=engine, if_exists="replace", index=False)
    skills_wide.to_sql("skills_data", con=engine, if_exists="replace", index=False)
    abilities_wide.to_sql("abilities_data", con=engine, if_exists="replace", index=False)
    tech_features.to_sql("technology_data", con=engine, if_exists="replace", index=False)
    final_data.to_sql("final_dataset", con=engine, if_exists="replace", index=False)

    print("Đã lưu thành công các bảng vào MySQL:")
    print("- wage_data")
    print("- skills_data")
    print("- abilities_data")
    print("- technology_data")
    print("- final_dataset")


# =========================================================
# 13. PHÂN TÍCH NHANH
# =========================================================

def quick_analysis(final_data: pd.DataFrame) -> None:
    print_section("BƯỚC 10: PHÂN TÍCH NHANH SAU XỬ LÝ")

    print("\nTop 10 nghề có lương trung vị năm cao nhất:")
    print(
        final_data[["soc_code", "occupation_title", "annual_median_wage"]]
        .sort_values("annual_median_wage", ascending=False)
        .head(10)
    )

    if "num_technology_skills" in final_data.columns:
        print("\nTop 10 nghề có nhiều technology skills nhất:")
        print(
            final_data[["soc_code", "occupation_title", "num_technology_skills"]]
            .sort_values("num_technology_skills", ascending=False)
            .head(10)
        )


# =========================================================
# 14. MAIN
# =========================================================

def main():
    wage, skills, abilities, technology = load_raw_data()

    wage_clean = process_wage_data(wage)
    skills_wide = process_skills_data(skills)
    abilities_wide = process_abilities_data(abilities)
    tech_features = process_technology_data(technology)

    final_data = merge_all_data(wage_clean, skills_wide, abilities_wide, tech_features)
    final_data = clean_final_dataset(final_data)

    save_csv(final_data)
    save_to_mysql(wage_clean, skills_wide, abilities_wide, tech_features, final_data)
    quick_analysis(final_data)

    print_section("HOÀN THÀNH")
    print("Bạn đã tạo xong:")
    print(f"1. {FINAL_CSV_PATH}")
    print("2. Các bảng trong MySQL database:", MYSQL_DATABASE)


if __name__ == "__main__":
    main()
