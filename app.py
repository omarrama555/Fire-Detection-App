import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# إعدادات الصفحة
st.set_page_config(page_title="نظام كشف الحريق الذكي", page_icon="🔥")

# عنوان التطبيق
st.title("🔥 نظام كشف الحريق باستخدام الذكاء الاصطناعي")
st.write("قم برفع صورة للتأكد مما إذا كانت تحتوي على حريق أم لا.")

# تحميل النموذج (تأكد أن اسم الملف مطابق لما ترفعه)
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('model_with_augmentation.keras')
    return model

model = load_my_model()

# رفع الصورة
uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة المرفوعة
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة التي تم رفعها', use_column_width=True)
    
    st.write("جاري التحليل...")
    
    # معالجة الصورة لتناسب مدخلات النموذج
    # ملاحظة: تأكد من حجم الصورة (Target Size) المستخدم في تدريبك (مثلاً 224x224)
    img = image.resize((224, 224)) 
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # التنبؤ
    prediction = model.predict(img_array)
    
    # عرض النتيجة بناءً على التصنيف (بفرض أن 0 نار و 1 غير ذلك أو العكس)
    # عدل التسميات بناءً على ترتيب المجلدات في تدريبك
    if prediction[0][0] > 0.5:
        st.error(f"تحذير: تم اكتشاف حريق! (نسبة الثقة: {prediction[0][0]*100:.2f}%)")
    else:
        st.success(f"آمن: لا يوجد حريق. (نسبة الثقة: {(1-prediction[0][0])*100:.2f}%)")
