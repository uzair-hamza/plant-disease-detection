"""Plant Disease Detector - live web demo (3D professional edition).

Upload a leaf photo and get an instant disease prediction.
Model: custom CNN trained on PlantVillage (54,303 images, 38 classes),
94.57% validation accuracy. Built by Muhammad Uzair Hamza.
"""

import json
import os
import urllib.request

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from tensorflow import keras

MODEL_URL = "https://github.com/uzair-hamza/plant-disease-detection/releases/download/v1.0/plant_model.keras"
CACHE_DIR = "model_cache"

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------- 3D hero
HERO_HTML = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<div id="hero3d" style="width:100%;height:360px;position:relative;overflow:hidden;
     border-radius:20px;border:1px solid rgba(52,211,153,0.25);
     box-shadow:0 20px 60px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.08);">
  <canvas id="c3d" style="width:100%;height:100%;display:block;"></canvas>
  <div style="position:absolute;inset:0;display:flex;flex-direction:column;
       align-items:center;justify-content:center;pointer-events:none;
       background:radial-gradient(ellipse at center, rgba(0,0,0,0.05) 0%, rgba(0,0,0,0.55) 100%);">
    <div style="font-size:52px;filter:drop-shadow(0 6px 18px rgba(0,0,0,0.7));">🌿</div>
    <div style="font-family:'Segoe UI',system-ui,sans-serif;font-weight:800;
         font-size:42px;letter-spacing:1px;
         background:linear-gradient(100deg,#a7f3d0 10%,#34d399 45%,#fbbf24 90%);
         -webkit-background-clip:text;background-clip:text;color:transparent;
         text-shadow:0 10px 40px rgba(0,0,0,0.4);">Plant Disease Detector</div>
    <div style="font-family:'Segoe UI',system-ui,sans-serif;color:#d1fae5;
         font-size:17px;margin-top:10px;letter-spacing:0.4px;">
         Upload a leaf photo &mdash; my AI model diagnoses it in seconds</div>
  </div>
</div>
<script>
(function(){
  var wrap = document.getElementById('hero3d');
  var canvas = document.getElementById('c3d');
  var renderer = new THREE.WebGLRenderer({canvas: canvas, antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  var scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x070b09, 0.045);
  var camera = new THREE.PerspectiveCamera(60, 2, 0.1, 100);
  camera.position.z = 16;

  // Rotating wireframe icosahedron (the "molecule")
  var ico = new THREE.Mesh(
    new THREE.IcosahedronGeometry(4.2, 1),
    new THREE.MeshBasicMaterial({color: 0x34d399, wireframe: true, transparent: true, opacity: 0.55})
  );
  scene.add(ico);
  var icoInner = new THREE.Mesh(
    new THREE.IcosahedronGeometry(2.4, 0),
    new THREE.MeshBasicMaterial({color: 0xfbbf24, wireframe: true, transparent: true, opacity: 0.35})
  );
  scene.add(icoInner);

  // Particle field
  var pGeo = new THREE.BufferGeometry();
  var N = 900, pos = new Float32Array(N * 3), col = new Float32Array(N * 3);
  var c1 = new THREE.Color(0x34d399), c2 = new THREE.Color(0xfbbf24), c3 = new THREE.Color(0x0e5e3a);
  for (var i = 0; i < N; i++) {
    pos[i*3]   = (Math.random() - 0.5) * 44;
    pos[i*3+1] = (Math.random() - 0.5) * 26;
    pos[i*3+2] = (Math.random() - 0.5) * 22;
    var c = Math.random() < 0.6 ? c1 : (Math.random() < 0.5 ? c2 : c3);
    col[i*3] = c.r; col[i*3+1] = c.g; col[i*3+2] = c.b;
  }
  pGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  pGeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  var points = new THREE.Points(pGeo, new THREE.PointsMaterial(
    {size: 0.14, vertexColors: true, transparent: true, opacity: 0.9}));
  scene.add(points);

  var mx = 0, my = 0;
  wrap.addEventListener('mousemove', function(e){
    var r = wrap.getBoundingClientRect();
    mx = ((e.clientX - r.left) / r.width - 0.5) * 2;
    my = ((e.clientY - r.top) / r.height - 0.5) * 2;
  });

  function resize(){
    var w = wrap.clientWidth, h = wrap.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();

  var t = 0;
  (function animate(){
    requestAnimationFrame(animate);
    t += 0.005;
    ico.rotation.x = t * 0.7;  ico.rotation.y = t;
    icoInner.rotation.x = -t * 1.1; icoInner.rotation.y = -t * 0.8;
    points.rotation.y = t * 0.12;
    camera.position.x += (mx * 2.2 - camera.position.x) * 0.04;
    camera.position.y += (-my * 1.4 - camera.position.y) * 0.04;
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  })();
})();
</script>
"""

# ---------------------------------------------------------------- styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
    .stApp { background: radial-gradient(1200px 600px at 20% -5%, #0d231a 0%, #070b09 55%) !important; }
    .stApp, .stMarkdown, p, span, div, label { font-family: 'Space Grotesk','Segoe UI',system-ui,sans-serif; }
    h1,h2,h3 { color: #ecfdf5 !important; letter-spacing: 0.3px; }
    p, .stMarkdown { color: #c9d6cf !important; }

    .glass {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 24px 26px;
        margin-bottom: 18px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 18px 45px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.07);
        transform: perspective(900px) rotateX(0deg);
        transition: transform .35s ease, box-shadow .35s ease;
    }
    .glass:hover {
        transform: perspective(900px) rotateX(1.6deg) translateY(-3px);
        box-shadow: 0 26px 60px rgba(0,0,0,0.55), 0 0 0 1px rgba(52,211,153,0.25),
                    inset 0 1px 0 rgba(255,255,255,0.10);
    }
    .stat3d {
        background: linear-gradient(150deg, rgba(52,211,153,0.14), rgba(52,211,153,0.03));
        border: 1px solid rgba(52,211,153,0.30);
        border-radius: 16px;
        padding: 18px 8px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08);
        transform: perspective(700px) rotateX(4deg);
        transition: transform .3s ease;
    }
    .stat3d:hover { transform: perspective(700px) rotateX(0deg) translateY(-4px); }
    .stat3d b { font-size: 1.65rem; color: #6ee7b7; display: block;
                text-shadow: 0 0 24px rgba(52,211,153,0.45); }
    .stat3d span { color: #9fb3a8; font-size: 0.88rem; }
    .result3d {
        background: linear-gradient(120deg, rgba(52,211,153,0.16), rgba(251,191,36,0.08));
        border: 1px solid rgba(52,211,153,0.45);
        border-radius: 16px;
        padding: 22px 24px;
        margin: 16px 0;
        box-shadow: 0 0 40px rgba(52,211,153,0.18), 0 18px 45px rgba(0,0,0,0.45);
    }
    .result3d b { color: #a7f3d0; font-size: 1.4rem; }
    .result3d .conf { color: #fde68a; font-size: 1.05rem; }

    /* tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px 12px 0 0;
        color: #c9d6cf; padding: 10px 22px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(52,211,153,0.16) !important;
        border-color: rgba(52,211,153,0.5) !important;
        color: #a7f3d0 !important;
    }
    /* buttons */
    .stLinkButton > a, .stButton > button {
        background: linear-gradient(120deg, #059669, #34d399) !important;
        color: #04120c !important; font-weight: 700 !important;
        border: none !important; border-radius: 12px !important;
        box-shadow: 0 10px 26px rgba(52,211,153,0.35) !important;
        transition: transform .2s ease !important;
    }
    .stLinkButton > a:hover, .stButton > button:hover { transform: translateY(-2px); }
    /* uploader */
    [data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.04) !important;
        border: 1.5px dashed rgba(52,211,153,0.5) !important;
        border-radius: 16px !important;
    }
    [data-testid="stFileUploader"] section small, [data-testid="stFileUploader"] section span { color: #9fb3a8 !important; }
    /* progress */
    .stProgress > div > div > div > div { background: linear-gradient(90deg,#34d399,#fbbf24) !important; }
    .footer { text-align:center; color:#5f7168; font-size:0.85rem;
              margin-top:34px; padding-top:16px; border-top:1px solid rgba(255,255,255,0.08); }
    #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_classes():
    """Download the trained model once, then reuse it."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_path = os.path.join(CACHE_DIR, "plant_model.keras")
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URL, model_path)
    model = keras.models.load_model(model_path)
    with open(os.path.join("results", "class_names.json")) as f:
        class_names = json.load(f)
    return model, class_names


def pretty_name(raw: str) -> str:
    """Tomato___Late_blight -> Tomato - Late blight."""
    return raw.replace("___", " - ").replace("_", " ")


def detector_tab(model, class_names):
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("1 · Upload a leaf photo")
        uploaded = st.file_uploader("Choose a leaf photo",
                                    type=["jpg", "jpeg", "png"],
                                    label_visibility="collapsed")
        if uploaded is not None:
            st.image(uploaded, caption="Your leaf photo", use_container_width=True)
        else:
            st.caption("JPG or PNG — the clearer the leaf, the better the diagnosis.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("2 · Get the diagnosis")
        if uploaded is None:
            st.info("Upload a photo on the left — the AI diagnosis appears here.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        with st.spinner("Analyzing the leaf..."):
            img = keras.utils.load_img(uploaded, target_size=(128, 128))
            x = keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            probs = model.predict(x, verbose=0)[0]
        top3 = probs.argsort()[-3:][::-1]
        st.markdown(
            f'<div class="result3d">Predicted<br><b>{pretty_name(class_names[top3[0]])}</b>'
            f'<br><span class="conf">Confidence: {probs[top3[0]] * 100:.1f}%</span></div>',
            unsafe_allow_html=True,
        )
        st.write("Top 3 predictions:")
        for i in top3:
            st.write(f"**{pretty_name(class_names[i])}** — {probs[i] * 100:.1f}%")
            st.progress(float(probs[i]))
        st.markdown("</div>", unsafe_allow_html=True)


def project_tab():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("What this project is")
    st.write(
        "A convolutional neural network that looks at a photo of a plant leaf and "
        "tells which disease it has — or that the leaf is healthy. I trained it on "
        "the public PlantVillage dataset and built this page so anyone can try it live."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, "54,303", "Leaf photos"),
        (c2, "38", "Classes: 30 diseases + 8 healthy"),
        (c3, "14", "Crops"),
        (c4, "94.57%", "Validation accuracy"),
    ]:
        col.markdown(
            f'<div class="stat3d"><b>{val}</b><span>{label}</span></div>',
            unsafe_allow_html=True,
        )
    st.write("")

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("The model")
    st.write(
        "A custom CNN I designed and trained from scratch: three convolutional blocks "
        "(32 / 64 / 128 filters), data augmentation during training (flip, rotation, zoom), "
        "and a dense classification head with dropout. Input 128×128, 10 epochs with Adam. "
        "Weighted F1 score: 0.95."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Tested honestly")
    st.write(
        "On lab photos (PlantVillage) the model reaches 94.57% accuracy. I also tested it "
        "on real field photos (PlantDoc) — there it reaches only 8.90%. That gap is the "
        "honest finding of my research: lab-trained models struggle in the field, and that "
        "is exactly what my ICECT 2027 paper studies, with Grad-CAM visualizations showing "
        "where the model looks when it decides."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.link_button("View the code on GitHub →",
                   "https://github.com/uzair-hamza/plant-disease-detection")


def about_tab():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3])
    with c1:
        st.image("https://github.com/uzair-hamza.png", width=170)
    with c2:
        st.subheader("Muhammad Uzair Hamza")
        st.write(
            "BSc Computer Systems Engineering student (5th semester), UET Peshawar. "
            "Undergraduate Research Assistant and NAVTTC Courses Coordinator at the "
            "National Center for Big Data and Cloud Computing."
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("What I do")
    st.write(
        "Machine learning, deep learning, and computer vision — plus graphic design "
        "and UI/UX. This plant disease detector is one of five ML projects on my GitHub, "
        "and the subject of my ICECT 2027 research paper."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    b1.link_button("GitHub →", "https://github.com/uzair-hamza")
    b2.link_button("ML Portfolio →", "https://ml-portfolio-uzair-hamza.vercel.app")
    b3.link_button("Design Portfolio →",
                   "https://graphics-portfolio-uzair-hamza.vercel.app")


def main() -> None:
    components.html(HERO_HTML, height=380)

    with st.spinner("Loading the trained model (first visit takes a moment)..."):
        model, class_names = load_model_and_classes()

    tab1, tab2, tab3 = st.tabs(["Detector", "About the Project", "About Me"])
    with tab1:
        detector_tab(model, class_names)
    with tab2:
        project_tab()
    with tab3:
        about_tab()

    st.markdown(
        '<div class="footer">Built by Muhammad Uzair Hamza · UET Peshawar · '
        "Custom CNN on PlantVillage · Live demo</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
