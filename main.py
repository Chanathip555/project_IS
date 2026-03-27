import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Data Science Project Portfolio", layout="wide")

# --- โหลด Assets (เหมือนเดิม) ---
@st.cache_resource
def load_all_assets():
    with open('ensemble_model.pkl', 'rb') as f:
        bank_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        bank_scaler = pickle.load(f)
    with open('nn_model_student.pkl', 'rb') as f:
        student_model = pickle.load(f)
    with open('scaler_student.pkl', 'rb') as f:
        student_scaler = pickle.load(f)
    return bank_model, bank_scaler, student_model, student_scaler

bank_model, bank_scaler, student_model, student_scaler = load_all_assets()

# --- CSS ตกแต่งเพิ่มเติม (ทำให้ตัวอักษรอ่านง่ายขึ้น) ---
st.markdown("""
    <style>
    .main-header { font-size: 36px !important; font-weight: bold; color: #1E88E5; }
    .sub-header { font-size: 24px !important; font-weight: bold; color: #424242; }
    .content-text { font-size: 18px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("📑 เมนูโปรเจกต์")
page = st.sidebar.radio("เลือกหน้าจอ:", [
    "📖 แนวทางการพัฒนา: Bank (ML)", 
    "📖 แนวทางการพัฒนา: Student (NN)",
    "🔮 ทดสอบ: Bank Prediction", 
    "🧠 ทดสอบ: Student Prediction"
])

# ---------------------------------------------------------
# หน้า 1: แนวทางการพัฒนา Bank (Machine Learning)
# ---------------------------------------------------------
if page == "📖 แนวทางการพัฒนา: Bank (ML)":
    st.markdown('<p class="main-header">🏦 แนวทางการพัฒนาโมเดล: Bank Term Deposit</p>', unsafe_allow_html=True)
    st.success("🎯 **เป้าหมาย:** วิเคราะห์พฤติกรรมลูกค้าเพื่อทำนายโอกาสในการสมัครเงินฝากประจำ")
    
    st.markdown("---")
    
    # ใช้ Columns เพื่อให้ดูเต็มหน้าจอ
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">🛠 การเตรียมข้อมูล (Data Preparation)</p>', unsafe_allow_html=True)
        st.info("""
        * **Data Cleaning:** จัดการค่าว่าง (Null) และกำจัดข้อมูลที่ซ้ำซ้อน
        * **Feature Engineering:** แปลงข้อมูลประเภทหมวดหมู่ (Categorical) เป็นตัวเลขผ่านการ Encoding
        * **Data Scaling:** ปรับช่วงข้อมูลด้วย **StandardScaler** เพื่อให้โมเดลประมวลผลได้แม่นยำ
        """)
        
    with col2:
        st.markdown('<p class="sub-header">🧠 อัลกอริทึม: Ensemble Learning</p>', unsafe_allow_html=True)
        st.warning("""
        * **ทฤษฎี:** ใช้เทคนิค **Voting Classifier** (ประสานพลังโมเดล)
        * **โครงสร้าง:** รวมโมเดล **Logistic Regression, Decision Tree และ KNN**
        * **การตัดสินใจ:** ใช้หลักเสียงข้างมาก (Majority Vote) เพื่อความเสถียรของผลลัพธ์
        """)

    st.markdown("---")
    st.subheader("📚 แหล่งอ้างอิงข้อมูล")
    st.code("แหล่งข้อมูล: UCI Machine Learning Repository - Bank Marketing Dataset", language="markdown")

# ---------------------------------------------------------
# หน้า 2: แนวทางการพัฒนา Student (Neural Network)
# ---------------------------------------------------------
elif page == "📖 แนวทางการพัฒนา: Student (NN)":
    st.markdown('<p class="main-header">🎓 แนวทางการพัฒนาโมเดล: Student Performance</p>', unsafe_allow_html=True)
    st.success("🎯 **เป้าหมาย:** ใช้โครงข่ายประสาทเทียมทำนายโอกาสสอบผ่านวิชาคณิตศาสตร์")

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">🛠 ขั้นตอนการจัดการข้อมูล</p>', unsafe_allow_html=True)
        st.info("""
        * **Target Labeling:** กำหนดเกณฑ์การสอบผ่าน (G3 ≥ 10) เป็น Binary Class (0, 1)
        * **Feature Selection:** คัดเลือก 5 ปัจจัยสำคัญ ได้แก่ คะแนนสอบกลางภาค (G2), ประวัติการสอบตก, การขาดเรียน, เวลาอ่านหนังสือ และการศึกษาของผู้ปกครอง
        * **Validation:** แบ่งข้อมูลเป็นชุดสอน (Train) 90% และชุดสอบ (Test) 10%
        """)

    with col2:
        st.markdown('<p class="sub-header">🧠 เทคโนโลยี: Neural Network (ANN)</p>', unsafe_allow_html=True)
        st.warning("""
        * **อัลกอริทึม:** Multi-layer Perceptron (MLP) 
        * **การปรับจูน:** ใช้ Hidden Layer ขนาด 10 Units พร้อมฟังก์ชัน ReLU 
        * **ประสิทธิภาพ:** ใช้ Solver 'lbfgs' เพื่อความแม่นยำสูงสุดถึง **95%** ในข้อมูลขนาดเล็ก
        """)

    st.markdown("---")
    st.subheader("📚 แหล่งอ้างอิงข้อมูล")
    st.code("แหล่งข้อมูล: UCI Machine Learning Repository - Student Performance Dataset", language="markdown")

# ---------------------------------------------------------
# หน้าทดสอบ (เหมือนเดิมแต่ปรับ Layout ให้กว้างขึ้น)
# ---------------------------------------------------------
# (ส่วนของหน้าทดสอบที่เหลือยังคงเดิมแต่ให้เอาไปใส่ใน main.py ต่อท้ายได้เลยครับ)
# ---------------------------------------------------------
# หน้า 3: ทดสอบ Bank (Machine Learning)
# ---------------------------------------------------------
elif page == "🔮 ทดสอบ: Bank Prediction":
    st.markdown('<p class="main-header">🔮 ระบบทำนายการสมัครเงินฝาก (Ensemble ML)</p>', unsafe_allow_html=True)
    st.write("ระบุข้อมูลลูกค้าด้านล่างเพื่อวิเคราะห์แนวโน้มการทำธุรกรรม")
    
    with st.container():
        st.markdown('<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("อายุ (Age)", 18, 100, 30)
            balance = st.number_input("เงินในบัญชี (Balance)", -5000, 100000, 1000)
            duration = st.number_input("เวลาคุยสายครั้งล่าสุด (วินาที)", 0, 5000, 200)
        with c2:
            campaign = st.number_input("จำนวนครั้งที่ติดต่อในแคมเปญนี้", 1, 50, 1)
            pdays = st.number_input("จำนวนวันที่ผ่านไปหลังติดต่อครั้งล่าสุด (-1 ถ้าไม่เคย)", -1, 999, -1)
            previous = st.number_input("จำนวนครั้งที่ติดต่อก่อนหน้านี้", 0, 50, 0)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 ประมวลผลด้วย Ensemble Model", use_container_width=True):
        # เตรียมข้อมูล 16 features (เติม 0 ในหลักที่เราไม่ได้ใช้ input)
        input_data = np.zeros((1, 16))
        input_data[0, 0], input_data[0, 5], input_data[0, 11] = age, balance, duration
        input_data[0, 12], input_data[0, 13], input_data[0, 14] = campaign, pdays, previous
        
        scaled_data = bank_scaler.transform(input_data)
        res = bank_model.predict(scaled_data)[0]
        
        st.markdown("---")
        if res == 1:
            st.balloons()
            st.success("### ✅ ผลทำนาย: ลูกค้ามีแนวโน้มจะสมัคร (YES)")
            st.write("โมเดลวิเคราะห์ว่าลักษณะข้อมูลนี้ตรงกับกลุ่มลูกค้าที่ตัดสินใจฝากเงินประจำ")
        else:
            st.error("### ❌ ผลทำนาย: ลูกค้าไม่น่าจะสมัคร (NO)")
            st.write("โมเดลวิเคราะห์ว่าโอกาสในการสมัครต่ำ ควรปรับกลยุทธ์การนำเสนอใหม่")

# ---------------------------------------------------------
# หน้า 4: ทดสอบ Student (Neural Network)
# ---------------------------------------------------------
elif page == "🧠 ทดสอบ: Student Prediction":
    st.markdown('<p class="main-header">🧠 ระบบทำนายผลการเรียน (Neural Network)</p>', unsafe_allow_html=True)
    st.write("วิเคราะห์โอกาสสอบผ่านด้วยโครงข่ายประสาทเทียมที่มีความแม่นยำ 95%")

    with st.container():
        st.markdown('<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            g2 = st.number_input("คะแนนสอบกลางภาค (G2: 0-20)", 0, 20, 10)
            failures = st.selectbox("จำนวนครั้งที่เคยสอบตก", [0, 1, 2, 3, 4])
            absences = st.number_input("จำนวนการขาดเรียนทั้งหมด", 0, 100, 0)
        with col2:
            studytime = st.select_slider("เวลาอ่านหนังสือต่อสัปดาห์", options=[1, 2, 3, 4], 
                                         format_func=lambda x: ["< 2 ชม.", "2-5 ชม.", "5-10 ชม.", "> 10 ชม."][x-1])
            medu = st.selectbox("ระดับการศึกษาของผู้ปกครอง (แม่)", options=[0, 1, 2, 3, 4],
                                format_func=lambda x: ["ไม่มีการศึกษา", "ประถม", "ม.ต้น", "ม.ปลาย", "ป.ตรีขึ้นไป"][x])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 ประมวลผลด้วย Neural Network", use_container_width=True):
        # ลำดับ Features: [G2, failures, absences, studytime, Medu]
        input_raw = np.array([[g2, failures, absences, studytime, medu]])
        input_scaled = student_scaler.transform(input_raw)
        
        prediction = student_model.predict(input_scaled)[0]
        prob = student_model.predict_proba(input_scaled)[0]

        st.markdown("---")
        if prediction == 1:
            st.balloons()
            st.success(f"### 🎉 ผลลัพธ์: มีโอกาสสอบผ่านสูงมาก! (ความมั่นใจ {prob[1]:.2%})")
            st.info("คำแนะนำ: รักษามาตรฐานการเรียนแบบนี้ต่อไป มีโอกาสคว้าเกรดสวยๆ แน่นอน")
        else:
            st.error(f"### ⚠️ ผลลัพธ์: เสี่ยงต่อการสอบตก (ความมั่นใจ {prob[0]:.2%})")
            st.warning("คำแนะนำ: ควรเพิ่มเวลาในการทบทวนบทเรียน และปรึกษาผู้สอนในจุดที่ไม่เข้าใจ")