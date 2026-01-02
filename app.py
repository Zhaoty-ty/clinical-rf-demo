# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="临床死亡风险预测系统",
    page_icon="🏥",
    layout="wide"
)

# 注入 CSS 样式
st.markdown("""
<style>
    .risk-high { color: #d9534f; font-weight: bold; font-size: 28px; }
    .risk-low { color: #5cb85c; font-weight: bold; font-size: 28px; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 关键：解决中文和负号乱码函数
# ==========================================
def set_chinese_font():
    """
    自动适配 Windows, Mac 和 Linux (Streamlit Cloud) 的中文字体
    """
    system_name = platform.system()
    
    # A. 解决负号显示为方框的问题 (关键修复!)
    plt.rcParams['axes.unicode_minus'] = False 
    
    # B. 设置中文字体
    if system_name == "Windows":
        # Windows 优先使用微软雅黑或黑体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    elif system_name == "Darwin":
        # Mac OS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
    else:
        # Linux (Streamlit Cloud)
        # 需要配合 packages.txt 安装 fonts-wqy-zenhei
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']

# ==========================================
# 3. 后台模型训练 (带缓存)
# ==========================================
@st.cache_resource
def train_model():
    # --- 模拟数据生成 ---
    np.random.seed(42)
    n_sim = 500
    
    data = pd.DataFrame({
        '年龄': np.random.normal(65, 12, n_sim),
        'SOFA评分': np.random.randint(1, 16, n_sim),
        'APACHE_II评分': np.random.normal(20, 5, n_sim),
        '是否使用血管活性药物': np.random.binomial(1, 0.4, n_sim),
        '是否使用CRRT': np.random.binomial(1, 0.2, n_sim),
        '降钙素原': np.abs(np.random.normal(2, 2, n_sim)),
        '乳酸脱氢酶': np.random.normal(250, 50, n_sim),
        '尿素': np.random.normal(10, 3, n_sim),
        '肌酸激酶': np.random.normal(100, 20, n_sim),
        'PT': np.random.normal(13, 2, n_sim),
        'INR': np.random.uniform(0.9, 3.0, n_sim),
        'TT': np.random.normal(16, 2, n_sim)
    })
    
    # 生成结局 (0=存活, 1=死亡)
    lp = -6 + 0.06*data['年龄'] + 0.3*data['SOFA评分'] + 1.2*data['是否使用血管活性药物']
    prob = 1 / (1 + np.exp(-lp))
    y = np.random.binomial(1, prob)
    
    # --- 训练模型 ---
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(data, y)
    
    # 创建 SHAP 解释器
    explainer = shap.TreeExplainer(model)
    
    return model, explainer

# 初始化模型
model, explainer = train_model()

# ==========================================
# 4. 侧边栏输入
# ==========================================
st.sidebar.header("📝 患者指标录入")

def user_input_features():
    with st.sidebar.expander("基本信息 & 评分", expanded=True):
        age = st.slider("年龄 (岁)", 18, 100, 65)
        sofa = st.slider("SOFA 评分", 0, 24, 5)
        apache = st.number_input("APACHE II 评分", 0, 71, 20)
        
    with st.sidebar.expander("治疗措施", expanded=True):
        vaso = st.selectbox("使用血管活性药物?", ["否", "是"])
        crrt = st.selectbox("使用 CRRT?", ["否", "是"])
        
    with st.sidebar.expander("生化指标", expanded=False):
        pct = st.number_input("降钙素原", 0.0, 100.0, 2.0)
        ldh = st.number_input("乳酸脱氢酶", 0, 2000, 250)
        urea = st.number_input("尿素", 0.0, 50.0, 10.0)
        ck = st.number_input("肌酸激酶", 0, 5000, 100)
        pt = st.number_input("PT", 0.0, 100.0, 13.0)
        inr = st.number_input("INR", 0.0, 10.0, 1.1)
        tt = st.number_input("TT", 0.0, 100.0, 16.0)

    # 整合数据
    row = {
        '年龄': age, 'SOFA评分': sofa, 'APACHE_II评分': apache,
        '是否使用血管活性药物': 1 if vaso=="是" else 0,
        '是否使用CRRT': 1 if crrt=="是" else 0,
        '降钙素原': pct, '乳酸脱氢酶': ldh, '尿素': urea,
        '肌酸激酶': ck, 'PT': pt, 'INR': inr, 'TT': tt
    }
    return pd.DataFrame(row, index=[0])

input_df = user_input_features()

# ==========================================
# 5. 主界面内容
# ==========================================
st.title("🏥 临床风险预测与归因分析")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📊 预测结果")
    if st.button("开始评估"):
        # 预测概率
        probs = model.predict_proba(input_df)
        risk = probs[0][1]  # 死亡概率
        
        st.markdown(f"#### 院内死亡概率: {risk*100:.1f}%")
        st.progress(risk)
        
        if risk > 0.5:
            st.markdown('<p class="risk-high">⚠️ 高风险患者</p>', unsafe_allow_html=True)
            st.error("建议加强监护与干预")
        else:
            st.markdown('<p class="risk-low">✅ 低风险患者</p>', unsafe_allow_html=True)
            st.success("建议常规护理")
            
        st.markdown("---")
        st.caption("当前输入的关键指标:")
        st.table(input_df.T.rename(columns={0:'数值'}).head(5))

with col2:
    st.subheader("🔍 归因分析 (SHAP Waterfall)")
    st.info("红色 = 增加风险因子 | 蓝色 = 降低风险因子")
    
    # 1. 设置字体与负号
    set_chinese_font()
    
    # 2. 计算 SHAP 值
    # input_df 是 (1, 12), explainer 输出 (1, 12, 2)
    shap_values = explainer(input_df)
    
    # 3. 提取用于绘图的对象
    # [0] = 第1个样本, [:, 1] = 所有特征针对类别1(死亡)的贡献
    explanation = shap_values[0, :, 1]
    
    # 4. 创建画布
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 5. 绘制瀑布图
    # max_display=10 限制显示前10个最重要的特征
    shap.plots.waterfall(explanation, max_display=10, show=False)
    
    # 6. 显示图表
    st.pyplot(fig)

st.markdown("---")
st.caption("© 2026 RF-SHAP Clinical Prediction System")
