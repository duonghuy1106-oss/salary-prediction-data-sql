# =========================================================
# DEMO DỰ ĐOÁN LƯƠNG THÁNG - TKINTER APP
# Không dùng Streamlit. Chạy bằng: python demo_salary_tkinter_vietnamese.py
# =========================================================

import os
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import pandas as pd


# =========================================================
# 1. CẤU HÌNH FILE MODEL
# =========================================================

MODEL_FILE = "models/best_salary_model.pkl"
FEATURES_FILE = "models/model_features.pkl"
FEATURE_MEANS_FILE = "models/feature_means.pkl"


# =========================================================
# 2. HÀM LOAD MODEL
# =========================================================

def load_artifacts():
    missing_files = [
        f for f in [MODEL_FILE, FEATURES_FILE, FEATURE_MEANS_FILE]
        if not os.path.exists(f)
    ]

    if missing_files:
        messagebox.showerror(
            "Thiếu file model",
            "Không tìm thấy các file sau:\n"
            + "\n".join(missing_files)
            + "\n\nHãy chạy trước: python evaluation_and_tuning.py"
        )
        raise FileNotFoundError(missing_files)

    model = joblib.load(MODEL_FILE)
    feature_columns = joblib.load(FEATURES_FILE)
    feature_means = joblib.load(FEATURE_MEANS_FILE)

    # feature_means có thể là Series hoặc dict
    feature_means = pd.Series(feature_means)

    return model, feature_columns, feature_means


# =========================================================
# 3. APP TKINTER
# =========================================================

class SalaryPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Demo dự đoán lương theo nghề nghiệp và kỹ năng")
        self.root.geometry("1120x760")
        self.root.minsize(1000, 650)

        try:
            self.model, self.feature_columns, self.feature_means = load_artifacts()
        except Exception:
            self.root.destroy()
            return

        self.vars = {}
        self.summary_tree = None
        self.result_var = tk.StringVar(value="Chưa có kết quả dự đoán")
        self.annual_var = tk.StringVar(value="")

        self.build_ui()

    # -----------------------------------------------------
    # TẠO GIAO DIỆN CHÍNH CÓ SCROLL
    # -----------------------------------------------------
    def build_ui(self):
        main_container = ttk.Frame(self.root, padding=12)
        main_container.pack(fill="both", expand=True)

        title = ttk.Label(
            main_container,
            text="HỆ THỐNG DEMO DỰ ĐOÁN LƯƠNG THÁNG",
            font=("Arial", 18, "bold")
        )
        title.pack(anchor="center", pady=(0, 6))

        subtitle = ttk.Label(
            main_container,
            text="Nhập thông tin công nghệ, kỹ năng và năng lực nghề nghiệp để dự đoán mức lương trung bình theo tháng.",
            font=("Arial", 10)
        )
        subtitle.pack(anchor="center", pady=(0, 12))

        body = ttk.Frame(main_container)
        body.pack(fill="both", expand=True)

        # Cột trái: vùng nhập liệu có scroll
        left_panel = ttk.Frame(body)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        canvas = tk.Canvas(left_panel, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        self.input_frame = ttk.Frame(canvas)

        self.input_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=self.input_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def resize_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", resize_canvas)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Cho phép cuộn bằng con lăn chuột
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Cột phải: kết quả + bảng tóm tắt
        right_panel = ttk.Frame(body)
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))

        self.build_input_sections()
        self.build_result_panel(right_panel)

    # -----------------------------------------------------
    # CÁC PHẦN NHẬP LIỆU
    # -----------------------------------------------------
    def build_input_sections(self):
        self.build_technology_section()
        self.build_skill_section()
        self.build_ability_section()

        btn = ttk.Button(
            self.input_frame,
            text="Dự đoán lương tháng",
            command=self.predict_salary
        )
        btn.pack(fill="x", pady=14, ipady=6)

        note = ttk.Label(
            self.input_frame,
            text=(
                "Lưu ý: Kết quả chỉ dùng cho mục đích học tập. "
                "Lương thực tế còn phụ thuộc vào kinh nghiệm, địa điểm, công ty và thị trường lao động."
            ),
            wraplength=520,
            foreground="#555555"
        )
        note.pack(anchor="w", pady=(0, 10))

    def build_technology_section(self):
        frame = ttk.LabelFrame(self.input_frame, text="1. Công nghệ sử dụng", padding=12)
        frame.pack(fill="x", pady=8)

        checks = [
            ("has_python", "Python"),
            ("has_sql", "SQL"),
            ("has_excel", "Excel"),
            ("has_tableau", "Tableau"),
            ("has_r", "R")
        ]

        for key, label in checks:
            var = tk.IntVar(value=0)
            self.vars[key] = var
            ttk.Checkbutton(frame, text=label, variable=var).pack(anchor="w", pady=2)

        self.add_slider(
            frame,
            key="num_technology_skills",
            label="Số lượng kỹ năng/công nghệ",
            from_=0,
            to=50,
            default=10
        )
        self.add_slider(
            frame,
            key="num_hot_technology",
            label="Số công nghệ phổ biến / hot",
            from_=0,
            to=30,
            default=5
        )
        self.add_slider(
            frame,
            key="num_in_demand_technology",
            label="Số công nghệ có nhu cầu cao",
            from_=0,
            to=30,
            default=5
        )
        self.add_slider(
            frame,
            key="num_technology_categories",
            label="Số nhóm công nghệ khác nhau",
            from_=0,
            to=20,
            default=3
        )

    def build_skill_section(self):
        frame = ttk.LabelFrame(self.input_frame, text="2. Kỹ năng nghề nghiệp, thang 0–7", padding=12)
        frame.pack(fill="x", pady=8)

        skills = [
            ("skill_Judgment_and_Decision_Making_LV", "Đánh giá và ra quyết định"),
            ("skill_Complex_Problem_Solving_LV", "Giải quyết vấn đề phức tạp"),
            ("skill_Critical_Thinking_LV", "Tư duy phản biện"),
            ("skill_Systems_Analysis_LV", "Phân tích hệ thống"),
            ("skill_Active_Learning_LV", "Chủ động học hỏi"),
            ("skill_Mathematics_LV", "Toán học")
        ]

        for key, label in skills:
            self.add_slider(frame, key, label, 0.0, 7.0, 4.0, resolution=0.1)

    def build_ability_section(self):
        frame = ttk.LabelFrame(self.input_frame, text="3. Năng lực cá nhân, thang 0–7", padding=12)
        frame.pack(fill="x", pady=8)

        abilities = [
            ("ability_Deductive_Reasoning_LV", "Suy luận diễn dịch"),
            ("ability_Problem_Sensitivity_LV", "Nhạy bén với vấn đề"),
            ("ability_Inductive_Reasoning_LV", "Suy luận quy nạp"),
            ("ability_Information_Ordering_LV", "Sắp xếp thông tin")
        ]

        for key, label in abilities:
            self.add_slider(frame, key, label, 0.0, 7.0, 4.0, resolution=0.1)

    def add_slider(self, parent, key, label, from_, to, default, resolution=1):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=6)

        ttk.Label(row, text=label, width=34).pack(side="left")

        var = tk.DoubleVar(value=default)
        self.vars[key] = var

        scale = ttk.Scale(
            row,
            from_=from_,
            to=to,
            variable=var,
            orient="horizontal"
        )
        scale.pack(side="left", fill="x", expand=True, padx=8)

        value_label = ttk.Label(row, text=f"{default:.1f}" if resolution != 1 else str(default), width=6)
        value_label.pack(side="left")

        def update_label(*args):
            val = var.get()
            if resolution == 1:
                val = int(round(val))
                value_label.config(text=str(val))
            else:
                value_label.config(text=f"{val:.1f}")

        var.trace_add("write", update_label)

    # -----------------------------------------------------
    # PANEL KẾT QUẢ
    # -----------------------------------------------------
    def build_result_panel(self, parent):
        result_box = ttk.LabelFrame(parent, text="Kết quả dự đoán", padding=14)
        result_box.pack(fill="x", pady=(0, 12))

        ttk.Label(
            result_box,
            textvariable=self.result_var,
            font=("Arial", 18, "bold"),
            foreground="#0B6E4F"
        ).pack(anchor="w", pady=4)

        ttk.Label(
            result_box,
            textvariable=self.annual_var,
            font=("Arial", 11),
            foreground="#333333"
        ).pack(anchor="w", pady=4)

        explain = ttk.Label(
            result_box,
            text=(
                "Mô hình gốc dự đoán lương trung vị theo năm. "
                "Ứng dụng chia kết quả cho 12 để quy đổi sang lương tháng."
            ),
            wraplength=460,
            foreground="#555555"
        )
        explain.pack(anchor="w", pady=(8, 0))

        summary_box = ttk.LabelFrame(parent, text="Bảng tóm tắt dữ liệu đầu vào", padding=10)
        summary_box.pack(fill="both", expand=True)

        columns = ("group", "feature", "value")
        self.summary_tree = ttk.Treeview(
            summary_box,
            columns=columns,
            show="headings",
            height=22
        )
        self.summary_tree.heading("group", text="Nhóm thông tin")
        self.summary_tree.heading("feature", text="Thông tin nhập")
        self.summary_tree.heading("value", text="Giá trị")

        self.summary_tree.column("group", width=120, anchor="w")
        self.summary_tree.column("feature", width=260, anchor="w")
        self.summary_tree.column("value", width=90, anchor="center")

        tree_scroll = ttk.Scrollbar(summary_box, orient="vertical", command=self.summary_tree.yview)
        self.summary_tree.configure(yscrollcommand=tree_scroll.set)

        self.summary_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        self.update_summary_table()

    # -----------------------------------------------------
    # LẤY INPUT VÀ DỰ ĐOÁN
    # -----------------------------------------------------
    def get_input_data(self):
        input_data = {}

        for key, var in self.vars.items():
            value = var.get()
            if key.startswith("has_"):
                input_data[key] = int(value)
            elif key.startswith("num_"):
                input_data[key] = int(round(value))
            else:
                input_data[key] = round(float(value), 1)

        return input_data

    def create_model_input(self, input_data):
        # Bắt đầu từ giá trị trung bình của các feature trong tập train
        input_df = pd.DataFrame([self.feature_means])

        # Đảm bảo đủ cột và đúng thứ tự như lúc train
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[self.feature_columns]

        # Ghi đè các biến người dùng nhập
        for col, value in input_data.items():
            if col in input_df.columns:
                input_df[col] = value

        return input_df

    def predict_salary(self):
        try:
            input_data = self.get_input_data()
            input_df = self.create_model_input(input_data)

            annual_prediction = float(self.model.predict(input_df)[0])
            monthly_prediction = annual_prediction / 12

            self.result_var.set(f"Lương tháng dự đoán: ${monthly_prediction:,.2f}")
            self.annual_var.set(f"Lương năm tương ứng: ${annual_prediction:,.2f}")

            self.update_summary_table(input_data)

        except Exception as e:
            messagebox.showerror("Lỗi dự đoán", str(e))

    # -----------------------------------------------------
    # BẢNG TÓM TẮT TIẾNG VIỆT
    # -----------------------------------------------------
    def update_summary_table(self, input_data=None):
        if input_data is None:
            input_data = self.get_input_data()

        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)

        yes_no = lambda x: "Có" if int(x) == 1 else "Không"

        rows = [
            ("Công nghệ", "Python", yes_no(input_data["has_python"])),
            ("Công nghệ", "SQL", yes_no(input_data["has_sql"])),
            ("Công nghệ", "Excel", yes_no(input_data["has_excel"])),
            ("Công nghệ", "Tableau", yes_no(input_data["has_tableau"])),
            ("Công nghệ", "R", yes_no(input_data["has_r"])),
            ("Công nghệ", "Số lượng kỹ năng/công nghệ", input_data["num_technology_skills"]),
            ("Công nghệ", "Số công nghệ phổ biến / hot", input_data["num_hot_technology"]),
            ("Công nghệ", "Số công nghệ có nhu cầu cao", input_data["num_in_demand_technology"]),
            ("Công nghệ", "Số nhóm công nghệ khác nhau", input_data["num_technology_categories"]),
            ("Kỹ năng", "Đánh giá và ra quyết định", input_data["skill_Judgment_and_Decision_Making_LV"]),
            ("Kỹ năng", "Giải quyết vấn đề phức tạp", input_data["skill_Complex_Problem_Solving_LV"]),
            ("Kỹ năng", "Tư duy phản biện", input_data["skill_Critical_Thinking_LV"]),
            ("Kỹ năng", "Phân tích hệ thống", input_data["skill_Systems_Analysis_LV"]),
            ("Kỹ năng", "Chủ động học hỏi", input_data["skill_Active_Learning_LV"]),
            ("Kỹ năng", "Toán học", input_data["skill_Mathematics_LV"]),
            ("Năng lực", "Suy luận diễn dịch", input_data["ability_Deductive_Reasoning_LV"]),
            ("Năng lực", "Nhạy bén với vấn đề", input_data["ability_Problem_Sensitivity_LV"]),
            ("Năng lực", "Suy luận quy nạp", input_data["ability_Inductive_Reasoning_LV"]),
            ("Năng lực", "Sắp xếp thông tin", input_data["ability_Information_Ordering_LV"]),
        ]

        for row in rows:
            self.summary_tree.insert("", "end", values=row)


# =========================================================
# 4. CHẠY APP
# =========================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = SalaryPredictionApp(root)
    root.mainloop()
