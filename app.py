import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import joblib

model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")
# قراءة البيانات
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
# تحميل النموذج المدرب
model = joblib.load("churn_model.pkl")
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)
# عنوان المشروع
st.title("📊 Customer Churn Prediction System")
st.markdown("---")

# ==================================
# فكرة المشروع
# ==================================

st.header("فكرة المشروع")

st.write("""
يهدف هذا المشروع إلى بناء نموذج تعلم آلي للتنبؤ باحتمالية مغادرة العملاء للخدمة (Customer Churn Prediction)
اعتماداً على بيانات العملاء المختلفة مثل مدة الاشتراك ونوع العقد وطريقة الدفع والرسوم الشهرية.

تم تنفيذ مراحل:
- فهم البيانات (Data Understanding)
- تنظيف البيانات (Data Cleaning)
- التحليل الاستكشافي (EDA)
- بناء النموذج باستخدام Decision Tree

لتصنيف العملاء إلى:
- العملاء المتوقع استمرارهم في الخدمة
- العملاء المتوقع مغادرتهم

يساعد هذا المشروع الشركات على التعرف المبكر على العملاء المعرضين للمغادرة واتخاذ إجراءات مناسبة للحفاظ عليهم.
""")

# ==================================
# Head
# ==================================

st.header("استعراض أول خمسة سجلات")

st.dataframe(df.head())

st.write("""
تم عرض أول خمسة سجلات للتأكد من تحميل البيانات بشكل صحيح.

يتضح أن كل صف يمثل عميلاً واحداً، بينما تمثل الأعمدة الخصائص المختلفة المتعلقة بالعميل مثل مدة الاشتراك والخدمات المستخدمة والرسوم الشهرية وحالة مغادرة العميل للشركة.

كما تم تحديد المتغير المستهدف (Churn) والذي سيتم استخدامه لاحقاً في بناء نموذج التنبؤ.
""")

# ==================================
# Info
# ==================================

st.header("فهم بنية البيانات")

st.write("""
تحتوي البيانات على 7032 سجلاً و20 متغيراً.

تحتوي البيانات على متغيرات رقمية ومتغيرات فئوية.

لا توجد قيم مفقودة ظاهرة من الفحص الأولي.

لوحظ أن بعض المتغيرات كانت مخزنة كنصوص وتمت معالجتها خلال مرحلة تنظيف البيانات.
""")

# ==================================
# الرسم الأول
# ==================================

st.header("العلاقة بين الرسوم الشهرية ومغادرة العملاء")

fig1 = px.histogram(
    df,
    x="MonthlyCharges",
    color="Churn",
    title="Monthly Charges by Churn"
)

st.plotly_chart(fig1, use_container_width=True)

st.write("""
تم استخدام هذا الرسم لدراسة العلاقة بين الرسوم الشهرية (MonthlyCharges)
ومغادرة العملاء (Churn).

نلاحظ وجود اختلاف في توزيع الرسوم الشهرية بين العملاء الذين غادروا والعملاء الذين استمروا.

مما يشير إلى أن الرسوم الشهرية قد تكون من العوامل المؤثرة على قرار العميل بالبقاء أو المغادرة.
""")

# ==================================
# الرسم الثاني
# ==================================

st.header("العلاقة بين مدة الاشتراك ومغادرة العملاء")

fig2 = px.box(
    df,
    x="Churn",
    y="tenure",
    color="Churn",
    title="Tenure by Churn"
)

st.plotly_chart(fig2, use_container_width=True)

st.write("""
أظهر الرسم أن العملاء الذين غادروا الشركة كانت مدة اشتراكهم أقل مقارنة بالعملاء الذين استمروا.

مما يدل على أن العملاء الجدد أو أصحاب مدة الاشتراك القصيرة أكثر عرضة للمغادرة.

لذلك يعتبر متغير Tenure من أهم المتغيرات المؤثرة.
""")

# ==================================
# الرسم الثالث
# ==================================

st.header("العلاقة بين نوع العقد ومغادرة العملاء")

fig3 = px.histogram(
    df,
    x="Contract",
    color="Churn",
    barmode="group",
    title="Contract Type by Churn"
)

st.plotly_chart(fig3, use_container_width=True)

st.write("""
يوضح الرسم أن العملاء أصحاب العقود الشهرية (Month-to-month)
لديهم أعلى معدل مغادرة مقارنة بالعقود السنوية أو العقود لمدة سنتين.

مما يشير إلى أن نوع العقد من أهم المتغيرات المؤثرة على قرار العميل.
""")

# ==================================
# الرسم الرابع
# ==================================

st.header("العلاقة بين طريقة الدفع ومغادرة العملاء")

fig4 = px.histogram(
    df,
    x="PaymentMethod",
    color="Churn",
    barmode="group",
    title="Payment Method by Churn"
)

st.plotly_chart(fig4, use_container_width=True)

st.write("""
يوضح الرسم العلاقة بين طريقة الدفع ومغادرة العملاء.

نلاحظ أن العملاء الذين يستخدمون Electronic Check لديهم عدد مغادرة أعلى مقارنة بباقي طرق الدفع.

بينما تظهر طرق الدفع التلقائية معدلات مغادرة أقل.
""")

# ==================================
# نتائج النموذج
# ==================================

st.header("نتائج نموذج التعلم الآلي")

st.subheader("Accuracy")

st.code("Accuracy = 72.49%")

st.write("""
تم تقييم نموذج Decision Tree باستخدام بيانات الاختبار.

حقق النموذج دقة تقارب 72%.

مما يدل على قدرته على التنبؤ بحالات إلغاء الخدمة بمستوى أداء مقبول.
""")

# ==================================
# Confusion Matrix
# ==================================

st.subheader("Confusion Matrix")

st.code("""
[[825 203]
 [179 195]]
""")

st.write("""
أظهرت مصفوفة الالتباس أن النموذج استطاع تصنيف عدد كبير من العملاء بشكل صحيح.

مع وجود بعض الحالات التي تم تصنيفها بشكل غير دقيق وهو أمر طبيعي في نماذج التعلم الآلي.

بشكل عام حقق النموذج أداء جيداً في التنبؤ بالعملاء المعرضين للمغادرة.
""")



# ==================================
# Live Prediction
# ==================================

st.header("🔮 Live Prediction")

st.write("أدخل بيانات العميل ثم اضغط Predict.")

tenure = st.slider(
    "مدة الاشتراك (Months)",
    0,
    72,
    12
)

monthly_charges = st.number_input(
    "الرسوم الشهرية (Monthly Charges)",
    min_value=0.0,
    value=70.0
)

contract = st.selectbox(
    "نوع العقد",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

payment_method = st.selectbox(
    "طريقة الدفع",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

internet_service = st.selectbox(
    "خدمة الإنترنت",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

if st.button("Predict"):

    st.write("المدخلات الحالية:")

    st.write({
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "InternetService": internet_service
    })

    input_data = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "InternetService": internet_service
    }])

    st.write(input_data)

    # تحويل البيانات إلى Dummy Variables
    input_data = pd.get_dummies(input_data)

    # مطابقة الأعمدة مع أعمدة التدريب
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # تنفيذ التنبؤ
    prediction = model.predict(input_data)

    # عرض النتيجة
    if prediction[0] == 1:
        st.error("🔴 العميل المتوقع أن يغادر الشركة")
    else:
        st.success("🟢 العميل المتوقع أن يستمر مع الشركة")

    st.success("✅ تمت عملية التنبؤ بنجاح.")