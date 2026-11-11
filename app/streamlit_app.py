import numpy as np
import streamlit as st

import requests

st.set_page_config(page_title="Digit Studio", layout="centered")
st.title("Digit Studio")
st.write("Draw a single digit from 0 to 9. Leave a little space around the edges.")

digit_pad = st.components.v2.component(
    "digit_pad",
    html="""
      <div class="studio">
        <section class="workspace">
          <h2>Drawing pad</h2>
          <div class="surface">
            <canvas class="pad" width="28" height="28" aria-label="Draw a digit with a mouse, pen or finger"></canvas>
          </div>
          <div class="tools">
            <button class="undo" disabled>↶ &nbsp; Undo</button>
            <button class="clear" disabled>Clear</button>
          </div>
        </section>
        <section class="inspect">
          <h2>Model input</h2>
          <canvas class="preview" width="28" height="28" aria-label="Live grayscale preview"></canvas>
          <p class="size">28 × 28 pixels</p>
          <p class="status" role="status">Waiting for a stroke</p>
        </section>
      </div>
    """,
    css="""
      * { box-sizing: border-box; }
      .studio {
        container-type: inline-size;
        display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
        background: #f5f6ee; color: #192d24; border: 1px solid #cdd6ca;
        border-radius: 20px; overflow: hidden;
        font-family: Inter, system-ui, sans-serif;
      }
      section { min-width: 0; padding: clamp(16px, 3vw, 28px); }
      h2 { font-size: 1.05rem; margin: 0 0 16px; font-weight: 650; }
      .surface {
        padding: 8px; background: #d5e8a8; border-radius: 10px;
        box-shadow: 0 3px 8px #20312015;
      }
      canvas { display: block; image-rendering: pixelated; background-color: #101713; }
      .pad {
        width: 100%; height: auto; border-radius: 4px; cursor: crosshair; touch-action: none;
        background-image: radial-gradient(#465349 0.8px, transparent 1px);
        background-size: calc(100% / 14) calc(100% / 14);
      }
      .tools { display: flex; gap: 8px; margin-top: 14px; }
      button {
        min-height: 44px; padding: 0 17px; border-radius: 8px; cursor: pointer;
        border: 1px solid #aebcaf; color: #21362a; background: transparent;
        font: 600 0.9rem Inter, system-ui, sans-serif;
      }
      button:hover:enabled { background: #e3eadb; }
      button:focus-visible { outline: 3px solid #39784e; outline-offset: 2px; }
      button:disabled { opacity: .45; cursor: default; }
      .inspect { background: #e7ecdf; }
      .preview { background: black; width: 84px; height: 84px; border-radius: 4px; }
      .size { font: 0.85rem ui-monospace, monospace; margin: 10px 0 20px; }
      .status { font-size: 0.9rem; margin: 0; color: #435447; }
      @container (max-width: 475px) {
        .workspace, .inspect { grid-column: 1 / -1; }
        .inspect { display: grid; grid-template-columns: 56px 1fr; gap: 3px 14px; }
        .preview { width: 56px; height: 56px; grid-column: 1; grid-row: 1 / 4; }
        .inspect h2, .size, .status { grid-column: 2; margin: 0; }
      }
    """,
    js="""
    export default function ({ parentElement, setStateValue }) {
      const get = selector => parentElement.querySelector(selector);
      const pad = get(".pad");
      // Keep stroke history and active handlers across Streamlit rerenders.
      if (pad.dataset.ready) return;
      pad.dataset.ready = "true";
      const ctx = pad.getContext("2d", { willReadFrequently: true });
      const preview = get(".preview").getContext("2d");
      const undo = get(".undo");
      const clear = get(".clear");
      const history = [];
      let pointer = null;
      let previous;
      ctx.strokeStyle = ctx.fillStyle = "white";
      ctx.lineCap = "round";
      ctx.lineWidth = 2.2;

      function snapshot() { return ctx.getImageData(0, 0, 28, 28); }
      function remember() {
        history.push(snapshot());
        if (history.length > 50) history.shift();
      }
      function refresh() {
        const frame = snapshot();
        preview.putImageData(frame, 0, 0);
        // Alpha is the grayscale intensity of white ink on a black background.
        const pixels = Array.from({ length: 784 }, (_, i) => frame.data[i * 4 + 3]);
        const hasInk = pixels.some(value => value > 0);
        undo.disabled = history.length === 0;
        clear.disabled = !hasInk;
        get(".status").textContent = hasInk ? "Drawing ready" : "Waiting for a stroke";
        setStateValue("pixels", pixels);
      }
      function position(event) {
        const bounds = pad.getBoundingClientRect();
        return [(event.clientX - bounds.left) * 28 / bounds.width,
                (event.clientY - bounds.top) * 28 / bounds.height];
      }
      function updatePreview() {
        preview.clearRect(0, 0, 28, 28);
        preview.drawImage(pad, 0, 0);
      }
      pad.onpointerdown = event => {
        if (pointer !== null || event.button !== 0 || !event.isPrimary) return;
        pointer = event.pointerId;
        pad.setPointerCapture(pointer);
        remember();
        previous = position(event);
        ctx.beginPath();
        ctx.arc(...previous, ctx.lineWidth / 2, 0, 2 * Math.PI);
        ctx.fill();
        updatePreview();
      };
      pad.onpointermove = event => {
        if (event.pointerId !== pointer) return;
        const next = position(event);
        ctx.beginPath();
        ctx.moveTo(...previous);
        ctx.lineTo(...next);
        ctx.stroke();
        previous = next;
        updatePreview();
      };
      pad.onlostpointercapture = event => {
        if (event.pointerId !== pointer) return;
        pointer = null;
        refresh();
      };
      undo.onclick = () => {
        if (pointer !== null || !history.length) return;
        ctx.putImageData(history.pop(), 0, 0);
        refresh();
      };
      clear.onclick = () => {
        if (pointer !== null) return;
        remember();
        ctx.clearRect(0, 0, 28, 28);
        refresh();
      };
    }
    """,
)

drawing = digit_pad(
    key="digit",
    default={"pixels": [0] * 784},
    on_pixels_change=lambda: None,
)
image = np.asarray(drawing.pixels, dtype=np.float32).reshape(28, 28) / 255.0

if st.button("Classify", type="primary", disabled=not image.any()):
    try:
        with st.spinner("Sending drawing…"):
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"pixels": image.tolist()},
                timeout=15,
            )
            response.raise_for_status()
            result = response.json()

        if "digit" in result:
            st.metric("Predicted digit", result["digit"])
        else:
            st.write("FastAPI received your drawing:", result)

    except requests.ConnectionError:
        st.error("Couldn’t reach FastAPI. Check that it’s running on port 8000.")
    except requests.Timeout:
        st.error("FastAPI took too long to respond.")
    except requests.HTTPError as error:
        st.error(f"FastAPI returned {error.response.status_code}: {error.response.text}")
