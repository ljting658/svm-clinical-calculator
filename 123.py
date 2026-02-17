import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==================== 页面基础配置（匹配示例界面） ====================
st.set_page_config(
    page_title="SVM Clinical Predictive Calculator for NSTE-ACS",
    page_icon="🧮",
    layout="centered",  # 紧凑布局，与示例一致
    initial_sidebar_state="collapsed"  # 隐藏侧边栏
)

# ==================== 样式配置（完全匹配示例界面风格） ====================
st.markdown("""
    <style>
    /* 全局样式 */
    body {
        font-family: 'Arial', sans-serif;
        color: #333333;
        background-color: #f8f9fa;
    }
    /* 主标题 */
    .main-title {
        font-size: 28px;
        font-weight: bold;
        color: #333333;
        margin-bottom: 15px;
        text-align: left;
    }
    /* 子标题 */
    .sub-title {
        font-size: 18px;
        font-weight: 600;
        color: #333333;
        margin-top: 25px;
        margin-bottom: 20px;
    }
    /* 输入框标签 */
    div[data-testid="stNumberInput"] label {
        font-size: 13px;
        color: #555555;
        font-weight: 500;
    }
    /* 按钮样式（匹配示例的蓝色按钮） */
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 4px;
        padding: 8px 24px;
        font-size: 14px;
        border: none;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #0052a3;
    }
    /* 结果指标样式 */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
        color: #0066cc;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #666666;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== 核心参数定义（你的9个特征） ====================
# 特征列表（固定9个）
FEATURES = [
    "T_min_mag", "cha_31_T_amp", "cha_12_T_amp",
    "cha_25_T_amp", "cha_6_T_amp", "cha_14_T_amp",
    "cha_31_ST_score", "T_posi_circ", "T_negi_circ"
]

# 特征显示名称（与你的变量名一致）
FEATURE_DISPLAY = {
    "T_min_mag": "T_min_mag",
    "cha_31_T_amp": "cha_31_T_amp",
    "cha_12_T_amp": "cha_12_T_amp",
    "cha_25_T_amp": "cha_25_T_amp",
    "cha_6_T_amp": "cha_6_T_amp",
    "cha_14_T_amp": "cha_14_T_amp",
    "cha_31_ST_score": "cha_31_ST_score",
    "T_posi_circ": "T_posi_circ",
    "T_negi_circ": "T_negi_circ"
}

# 特征参考范围（可根据你的论文数据调整）
FEATURE_RANGES = {
    "T_min_mag": (-5.0, 5.0),
    "cha_31_T_amp": (0.0, 10.0),
    "cha_12_T_amp": (0.0, 10.0),
    "cha_25_T_amp": (0.0, 10.0),
    "cha_6_T_amp": (0.0, 10.0),
    "cha_14_T_amp": (0.0, 10.0),
    "cha_31_ST_score": (0.0, 5.0),
    "T_posi_circ": (0.0, 20.0),
    "T_negi_circ": (-20.0, 0.0)
}


# ==================== 加载模型和标准化器 ====================
@st.cache_resource
def load_model_and_scaler():
    """加载预训练SVM模型和标准化器"""
    try:
        # 加载你的SVM模型
        model = joblib.load("./final_SVM_model.pkl")

        # 加载训练时保存的标准化器（必须替换为你自己的scaler.pkl）
        # 如果还没保存scaler，先运行训练代码保存，再取消下面注释
        scaler = joblib.load("./final_scaler.pkl")

        # 临时方案：若未保存scaler，用示例值（需替换为训练集真实均值/标准差）
        scaler = StandardScaler()
        # 请替换为你训练集的真实均值（示例值，仅临时用）
        scaler.mean_ = np.array([0.1, 2.3, 1.8, 2.1, 1.5, 1.7, 0.9, 8.5, -7.2])
        # 请替换为你训练集的真实标准差（示例值，仅临时用）
        scaler.scale_ = np.array([0.7, 1.1, 1.0, 1.2, 0.8, 0.9, 0.4, 3.2, 2.8])

        return model, scaler
    except FileNotFoundError:
        st.error("❌ 模型文件未找到，请检查路径：D:\\AApython\\final_SVM_model.pkl")
        st.stop()
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        st.stop()


# ==================== 预测函数 ====================
def predict_probability(model, scaler, feature_values):
    """基于9个特征预测概率（适配SVM模型）"""
    # 标准化输入特征
    features_array = np.array(feature_values).reshape(1, -1)
    features_scaled = scaler.transform(features_array)

    # SVM预测概率（确保训练时设置了probability=True）
    prob = model.predict_proba(features_scaled)[0, 1]
    return prob


# ==================== 主页面构建（完全匹配附图界面） ====================
def main():
    # 加载模型和标准化器
    model, scaler = load_model_and_scaler()

    # 页面标题（与示例一致的风格）
    st.markdown('<div class="main-title">SVM Clinical Predictive Calculator for NSTE-ACS</div>', unsafe_allow_html=True)
    st.divider()

    # 输入变量标题
    st.markdown('<div class="sub-title">Input Variables</div>', unsafe_allow_html=True)

    # 3列布局（9个特征均分，与附图一致）
    col1, col2, col3 = st.columns(3, gap="medium")
    feature_values = []

    # 第一列：3个特征
    with col1:
        for feat in FEATURES[0:3]:  # T_min_mag, cha_31_T_amp, cha_12_T_amp
            min_val, max_val = FEATURE_RANGES[feat]
            val = st.number_input(
                label=FEATURE_DISPLAY[feat],
                min_value=float(min_val),
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),  # 默认值为范围中间值
                step=0.1,
                key=f"feat_{feat}",
                help=f"Reference range: {min_val} to {max_val}"  # 帮助提示（❓图标）
            )
            feature_values.append(val)

    # 第二列：3个特征
    with col2:
        for feat in FEATURES[3:6]:  # cha_25_T_amp, cha_6_T_amp, cha_14_T_amp
            min_val, max_val = FEATURE_RANGES[feat]
            val = st.number_input(
                label=FEATURE_DISPLAY[feat],
                min_value=float(min_val),
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),
                step=0.1,
                key=f"feat_{feat}",
                help=f"Reference range: {min_val} to {max_val}"
            )
            feature_values.append(val)

    # 第三列：3个特征
    with col3:
        for feat in FEATURES[6:9]:  # cha_31_ST_score, T_posi_circ, T_negi_circ
            min_val, max_val = FEATURE_RANGES[feat]
            val = st.number_input(
                label=FEATURE_DISPLAY[feat],
                min_value=float(min_val),
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),
                step=0.1,
                key=f"feat_{feat}",
                help=f"Reference range: {min_val} to {max_val}"
            )
            feature_values.append(val)

    # 预测按钮（与示例一致的位置和样式）
    predict_btn = st.button("Calculate Prediction", type="primary")

    # 预测结果展示（匹配示例的结果样式）
    if predict_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
        st.markdown('<div class="sub-title">Prediction Result</div>', unsafe_allow_html=True)

        # 计算预测概率
        prob = predict_probability(model, scaler, feature_values)

        # 显示核心概率结果
        st.metric(
            label="Predicted Probability",
            value=f"{prob:.3f} ({prob * 100:.1f}%)"
        )

        # 风险等级提示
        if prob >= 0.5:
            st.warning(f"⚠️ High Risk - Probability: {prob * 100:.1f}%")
        else:
            st.success(f"✅ Low Risk - Probability: {prob * 100:.1f}%")

    # 下载功能（与示例一致）
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Download Input & Result (CSV)"):
        # 生成包含输入和结果的CSV
        input_df = pd.DataFrame({
            "Feature": FEATURES,
            "Input_Value": feature_values
        })
        # 如果已预测，添加结果列
        if predict_btn:
            input_df.loc[len(input_df)] = ["Predicted_Probability", f"{prob:.3f}"]
            input_df.loc[len(input_df)] = ["Risk_Level", "High" if prob >= 0.5 else "Low"]

        # 生成CSV文件
        csv_data = input_df.to_csv(index=False, encoding="utf-8")
        st.download_button(
            label="Confirm Download",
            data=csv_data,
            file_name="svm_calculator_result.csv",
            mime="text/csv",
            key="download_btn"
        )


# ==================== 执行主函数 ====================
if __name__ == "__main__":
    main()