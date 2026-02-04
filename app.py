import streamlit as st
import cv2
import numpy as np
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Inspection Qualité (OpenCV)", layout="wide")

# -----------------------------
# Utils
# -----------------------------
def read_image_from_upload(upload):
    data = np.frombuffer(upload.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR
    return img

def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0

def largest_contour(contours):
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def compute_features(bgr, mask, contour):
    # Area + perimeter
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    circularity = safe_div(4.0 * np.pi * area, (peri * peri))

    # ROI pixels (product only)
    product_pixels = bgr[mask == 255]
    if product_pixels.size == 0:
        return None

    # Color in Lab (more stable than RGB)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_pixels = lab[mask == 255]
    L_mean, a_mean, b_mean = lab_pixels.mean(axis=0).tolist()

    # Texture: variance of grayscale + Laplacian variance
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_prod = gray[mask == 255]
    gray_var = float(np.var(gray_prod))

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_prod = lap[mask == 255]
    lap_var = float(np.var(lap_prod))

    return {
        "area": area,
        "perimeter": peri,
        "circularity": circularity,
        "L_mean": float(L_mean),
        "a_mean": float(a_mean),
        "b_mean": float(b_mean),
        "gray_var": gray_var,
        "lap_var": lap_var,
    }

def segment_product(bgr, hsv_s_low, hsv_s_high, hsv_v_low, hsv_v_high, morph_k):
    # Blur to reduce noise
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # "Fond clair" vs "produit coloré" heuristic: use S and V thresholds
    # Keep pixels likely belonging to product (moderate/high saturation, not too bright background)
    lower = np.array([0, hsv_s_low, hsv_v_low], dtype=np.uint8)
    upper = np.array([179, hsv_s_high, hsv_v_high], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Morphology for robustness
    k = max(1, int(morph_k))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = largest_contour(contours)

    if c is None or cv2.contourArea(c) < 50:
        return None, None, None

    # Keep only largest contour as product
    product_mask = np.zeros_like(mask)
    cv2.drawContours(product_mask, [c], -1, 255, thickness=-1)

    # Clean again
    product_mask = cv2.morphologyEx(product_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Bounding box ROI
    x, y, w, h = cv2.boundingRect(c)
    roi = bgr[y:y+h, x:x+w].copy()

    return product_mask, c, roi

def default_thresholds():
    # Seuils "démo" (tu peux les calibrer ensuite)
    return {
        "area_min": 4000,
        "area_max": 200000,
        "circularity_min": 0.60,

        # "trop sombre" -> L_mean trop bas
        "L_min": 90,
        "L_max": 240,

        # texture: laplacian variance trop faible => trop lisse/flou / pas de détails
        "lap_var_min": 20,
        "lap_var_max": 2000,

        # gray variance trop élevée => bruit/texture anormale (optionnel)
        "gray_var_max": 6000,
    }

def decide_ok_nok(feat, th):
    reasons = []

    # Checks
    if feat["area"] < th["area_min"]:
        reasons.append("surface trop petite")
    if feat["area"] > th["area_max"]:
        reasons.append("surface trop grande")
    if feat["circularity"] < th["circularity_min"]:
        reasons.append("forme irrégulière")

    if feat["L_mean"] < th["L_min"]:
        reasons.append("trop sombre")
    if feat["L_mean"] > th["L_max"]:
        reasons.append("trop claire")

    if feat["lap_var"] < th["lap_var_min"]:
        reasons.append("texture trop faible / image floue")
    if feat["lap_var"] > th["lap_var_max"]:
        reasons.append("texture anormale (trop de détails)")

    if feat["gray_var"] > th["gray_var_max"]:
        reasons.append("bruit/variation anormale")

    ok = (len(reasons) == 0)

    # Simple confidence score: start at 1.0, subtract penalties
    # (tu peux le rendre plus “ML-like” si tu veux)
    penalties = {
        "surface trop petite": 0.20,
        "surface trop grande": 0.15,
        "forme irrégulière": 0.25,
        "trop sombre": 0.20,
        "trop claire": 0.15,
        "texture trop faible / image floue": 0.20,
        "texture anormale (trop de détails)": 0.15,
        "bruit/variation anormale": 0.10,
    }
    conf = 1.0
    for r in reasons:
        conf -= penalties.get(r, 0.10)
    conf = float(np.clip(conf, 0.05, 0.99))

    return ok, conf, reasons

def calibrate_from_ok_features(ok_features, k=2.0):
    # th = mean ± k*std (clamped to sane ranges)
    df = pd.DataFrame(ok_features)
    th = default_thresholds()

    def mean_std(col):
        return float(df[col].mean()), float(df[col].std(ddof=0) + 1e-9)

    area_m, area_s = mean_std("area")
    circ_m, circ_s = mean_std("circularity")
    L_m, L_s = mean_std("L_mean")
    lap_m, lap_s = mean_std("lap_var")
    gray_m, gray_s = mean_std("gray_var")

    th["area_min"] = max(100.0, area_m - k * area_s)
    th["area_max"] = max(th["area_min"] + 100.0, area_m + k * area_s)

    th["circularity_min"] = float(np.clip(circ_m - k * circ_s, 0.10, 0.98))

    th["L_min"] = float(np.clip(L_m - k * L_s, 0.0, 255.0))
    th["L_max"] = float(np.clip(L_m + k * L_s, th["L_min"] + 1.0, 255.0))

    th["lap_var_min"] = max(0.0, lap_m - k * lap_s)
    th["lap_var_max"] = max(th["lap_var_min"] + 1.0, lap_m + k * lap_s)

    th["gray_var_max"] = max(500.0, gray_m + k * gray_s)

    return th

# -----------------------------
# UI
# -----------------------------
st.title("🍫 Inspection Qualité — OpenCV (OK / NOK)")

with st.sidebar:
    st.header("Paramètres segmentation")
    hsv_s_low = st.slider("HSV S min", 0, 255, 40)
    hsv_s_high = st.slider("HSV S max", 0, 255, 255)
    hsv_v_low = st.slider("HSV V min", 0, 255, 30)
    hsv_v_high = st.slider("HSV V max", 0, 255, 240)
    morph_k = st.slider("Morph kernel (rayon)", 1, 15, 5)

    st.divider()
    st.header("Seuils décision (manual)")
    if "thresholds" not in st.session_state:
        st.session_state["thresholds"] = default_thresholds()

    th = st.session_state["thresholds"]

    th["area_min"] = st.number_input("Aire min", value=float(th["area_min"]), step=100.0)
    th["area_max"] = st.number_input("Aire max", value=float(th["area_max"]), step=100.0)
    th["circularity_min"] = st.slider("Circularité min", 0.0, 1.0, float(th["circularity_min"]), 0.01)

    th["L_min"] = st.slider("L* min (trop sombre)", 0, 255, int(th["L_min"]))
    th["L_max"] = st.slider("L* max (trop claire)", 0, 255, int(th["L_max"]))

    th["lap_var_min"] = st.number_input("Laplacian var min", value=float(th["lap_var_min"]), step=1.0)
    th["lap_var_max"] = st.number_input("Laplacian var max", value=float(th["lap_var_max"]), step=10.0)
    th["gray_var_max"] = st.number_input("Gray var max", value=float(th["gray_var_max"]), step=50.0)

    st.session_state["thresholds"] = th

tab1, tab2 = st.tabs(["🔎 Inspection", "🎯 Calibration (images OK)"])

# -----------------------------
# TAB 1 - Inspection
# -----------------------------
with tab1:
    uploads = st.file_uploader(
        "Upload une image ou un lot d’images (png/jpg/jpeg)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    show_debug = st.checkbox("Afficher les visuels (masque/contours/ROI)", value=True)

    results = []
    if uploads:
        for up in uploads:
            bgr = read_image_from_upload(up)
            if bgr is None:
                st.error(f"Impossible de lire {up.name}")
                continue

            mask, contour, roi = segment_product(
                bgr, hsv_s_low, hsv_s_high, hsv_v_low, hsv_v_high, morph_k
            )

            colA, colB = st.columns([1, 1])
            with colA:
                st.subheader(up.name)
                st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Image originale", use_container_width=True)

            if mask is None:
                with colB:
                    st.warning("Produit non détecté → NOK (segmentation)")
                results.append({
                    "file": up.name,
                    "status": "NOK",
                    "confidence": 0.20,
                    "reason": "produit non détecté",
                })
                continue

            feat = compute_features(bgr, mask, contour)
            if feat is None:
                with colB:
                    st.warning("Features non calculables → NOK")
                results.append({
                    "file": up.name,
                    "status": "NOK",
                    "confidence": 0.20,
                    "reason": "features non calculables",
                })
                continue

            ok, conf, reasons = decide_ok_nok(feat, st.session_state["thresholds"])
            status = "OK" if ok else "NOK"

            explanation = "✅ Conforme" if ok else ("❌ " + ", ".join(reasons))

            with colB:
                if ok:
                    st.success(f"{status} — confiance: {conf:.2f}")
                else:
                    st.error(f"{status} — confiance: {conf:.2f}")
                st.write(explanation)

                st.caption("Features (debug)")
                st.json({k: round(v, 3) for k, v in feat.items()})

            if show_debug:
                dbg = bgr.copy()
                cv2.drawContours(dbg, [contour], -1, (0, 255, 0), 2)
                st.image(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB), caption="Contours", use_container_width=True)

                st.image(mask, caption="Masque produit", use_container_width=True)

                if roi is not None and roi.size > 0:
                    st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption="ROI (bounding box)", use_container_width=True)

            results.append({
                "file": up.name,
                "status": status,
                "confidence": round(conf, 3),
                "reason": "; ".join(reasons) if reasons else "conforme",
                **{k: round(v, 3) for k, v in feat.items()}
            })

        if results:
            st.divider()
            df = pd.DataFrame(results)
            st.subheader("📄 Résultats (table + export CSV)")
            st.dataframe(df, use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Télécharger CSV",
                data=csv_bytes,
                file_name="inspection_results.csv",
                mime="text/csv"
            )

# -----------------------------
# TAB 2 - Calibration
# -----------------------------
with tab2:
    st.write("Upload des images **OK** (produits conformes). L’app calcule des seuils automatiquement.")

    ok_uploads = st.file_uploader(
        "Images OK (png/jpg/jpeg)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="ok_calib"
    )

    k = st.slider("k (seuil = moyenne ± k·std)", 1.0, 3.5, 2.0, 0.1)

    if ok_uploads:
        ok_feats = []
        preview_cols = st.columns(4)

        for idx, up in enumerate(ok_uploads):
            bgr = read_image_from_upload(up)
            if bgr is None:
                continue

            mask, contour, roi = segment_product(
                bgr, hsv_s_low, hsv_s_high, hsv_v_low, hsv_v_high, morph_k
            )
            if mask is None:
                continue

            feat = compute_features(bgr, mask, contour)
            if feat is None:
                continue

            ok_feats.append(feat)

            with preview_cols[idx % 4]:
                st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption=up.name, use_container_width=True)

        if len(ok_feats) < 3:
            st.warning("Ajoute au moins 3 images OK (idéalement 10+) pour une calibration stable.")
        else:
            new_th = calibrate_from_ok_features(ok_feats, k=k)

            st.success("Seuils calibrés !")
            st.json({k: round(v, 3) if isinstance(v, float) else v for k, v in new_th.items()})

            if st.button("✅ Appliquer ces seuils à l’inspection"):
                st.session_state["thresholds"] = new_th
                st.success("Seuils appliqués (tab Inspection).")
                
    st.set_page_config(
    page_title="Inspection Qualité – OK / NOK",
    page_icon="🧪",  # emoji OU image
    layout="wide"
)