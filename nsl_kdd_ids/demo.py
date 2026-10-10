from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from numbers import Real
from typing import Any
from urllib.parse import urlparse

from nsl_kdd_ids.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    dataset_dir,
    predict_record,
)

MAX_REQUEST_BYTES = 16_384


def _normalize_record(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Request body must be a JSON object of NSL-KDD features.")
    missing = sorted(set(FEATURE_COLUMNS) - set(payload))
    extra = sorted(set(payload) - set(FEATURE_COLUMNS))
    if missing or extra:
        raise ValueError(
            f"Invalid feature set. Missing: {missing}; unexpected: {extra}."
        )

    record: dict[str, Any] = {}
    for feature in FEATURE_COLUMNS:
        value = payload[feature]
        if feature in CATEGORICAL_FEATURES:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{feature} must be a non-empty string.")
            record[feature] = value.strip()
            continue
        if isinstance(value, bool):
            raise ValueError(f"{feature} must be a finite number.")
        try:
            numeric_value = float(value) if isinstance(value, (str, Real)) else math.nan
        except (TypeError, ValueError, OverflowError):
            numeric_value = math.nan
        if not math.isfinite(numeric_value):
            raise ValueError(f"{feature} must be a finite number.")
        record[feature] = numeric_value
    return record


def predict_payload(payload: Any) -> dict[str, Any]:
    record = _normalize_record(payload)
    return predict_record(record, model_config="baseline")


def load_training_sample() -> dict[str, Any]:
    train_path = dataset_dir() / "KDDTrain+.txt"
    with train_path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.reader(stream), None)
    if row is None or len(row) != len(FEATURE_COLUMNS) + 2:
        raise ValueError(f"{train_path} does not contain a valid NSL-KDD training row.")

    raw_record = dict(zip(FEATURE_COLUMNS, row[: len(FEATURE_COLUMNS)], strict=True))
    return _normalize_record(raw_record)


def _feature_label(name: str) -> str:
    return name.replace("_", " ").title()


def _render_page() -> bytes:
    features_json = json.dumps(FEATURE_COLUMNS)
    page = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ThreatAhead | Intrusion Detection Demo</title>
  <style>
    :root { color-scheme: light; --ink:#152438; --muted:#63748a; --line:#dce5ee;
      --blue:#185adb; --blue-dark:#1246ae; --canvas:#f3f6fa; --green:#117a53;
      --red:#b42335; --surface:#fff; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--canvas); color:var(--ink);
      font:15px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
    header { color:#fff; background:linear-gradient(115deg,#102b52,#185adb);
      padding:38px max(24px,calc((100% - 1100px)/2)) 34px; }
    .eyebrow { color:#bcd0ff; text-transform:uppercase; letter-spacing:.12em;
      font-size:12px; font-weight:700; }
    h1 { margin:7px 0 2px; font-size:clamp(30px,5vw,42px); letter-spacing:-.04em; }
    header p { margin:0; color:#dce7ff; font-size:17px; }
    main { max-width:1100px; padding:26px 22px 54px; margin:auto; }
    .intro { display:flex; align-items:flex-start; justify-content:space-between;
      gap:20px; margin-bottom:20px; }
    .intro p { max-width:700px; margin:0; color:var(--muted); }
    .badge { flex:none; border:1px solid #b9caff; color:#1749b0; background:#eaf0ff;
      border-radius:99px; padding:7px 12px; font-size:12px; font-weight:800;
      letter-spacing:.05em; text-transform:uppercase; }
    .layout { display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:20px; }
    .card { background:var(--surface); border:1px solid var(--line); border-radius:14px;
      box-shadow:0 4px 18px #172c4610; }
    .form-card { padding:22px; }
    h2 { font-size:19px; margin:0 0 4px; }
    .hint { color:var(--muted); margin:0 0 16px; font-size:13px; }
    details { border-top:1px solid var(--line); padding:14px 0 2px; }
    summary { cursor:pointer; font-weight:700; margin-bottom:12px; }
    .fields { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
    label { color:#34465b; display:block; font-size:12px; font-weight:650; }
    input { display:block; width:100%; margin-top:5px; min-height:39px; border:1px solid #cbd6e2;
      border-radius:7px; padding:8px 9px; color:var(--ink); background:white; font:inherit; font-size:14px; }
    input:focus { outline:3px solid #185adb2b; border-color:var(--blue); }
    .actions { display:flex; flex-wrap:wrap; align-items:center; gap:10px; padding-top:18px; }
    button { cursor:pointer; border:0; border-radius:8px; padding:11px 16px;
      font:inherit; font-weight:750; }
    .primary { color:#fff; background:var(--blue); }
    .primary:hover { background:var(--blue-dark); }
    .secondary { color:#26415f; background:#edf2f7; }
    button:disabled { opacity:.6; cursor:wait; }
    .status { min-height:24px; color:var(--red); font-size:13px; margin:10px 0 0; }
    .side { padding:20px; height:max-content; }
    .side h2 { font-size:16px; }
    .side p, .side li { color:var(--muted); font-size:13px; }
    .side ul { padding-left:18px; }
    .result { margin-top:20px; padding:22px; border-radius:12px; background:#edf2f7;
      border:1px solid var(--line); }
    .result[hidden] { display:none; }
    .result.normal { border-color:#a9dfc9; background:#edfaf4; }
    .result.attack { border-color:#f1b5bc; background:#fff1f2; }
    .result-kicker { margin:0; color:var(--muted); font-size:12px; font-weight:800;
      letter-spacing:.08em; text-transform:uppercase; }
    .result h2 { margin:2px 0; font-size:34px; letter-spacing:-.03em; }
    .result.normal h2 { color:var(--green); }
    .result.attack h2 { color:var(--red); }
    .family { margin:4px 0 0; font-weight:700; }
    .disclaimer { margin:16px 0 0; padding:12px; border-left:3px solid #f0b44d;
      background:#fff9e9; color:#614b1b; font-size:12px; }
    footer { max-width:1100px; margin:auto; padding:0 22px 25px; color:var(--muted);
      font-size:12px; }
    @media(max-width:850px) { .layout { grid-template-columns:1fr; } .side { order:2; } }
    @media(max-width:620px) { .fields { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .intro { flex-direction:column; } header { padding-top:28px; } }
  </style>
</head>
<body>
  <header><div class="eyebrow">Network security · ML demo</div>
    <h1>ThreatAhead</h1>
    <p>Machine Learning-Based Network Intrusion Detection</p>
  </header>
  <main>
    <div class="intro">
      <p>This demonstration classifies one NSL-KDD-format connection record in two
        stages: normal versus attack, then attack family for detected attacks.
        It is not connected to a live network.</p>
      <span class="badge">Baseline Model</span>
    </div>
    <div class="layout">
      <section class="card form-card">
        <h2>Connection record</h2>
        <p class="hint">Enter all 41 raw NSL-KDD input features. Labels and difficulty are not inputs.</p>
        <form id="record-form">
          <details open><summary>Categorical features</summary>
            <div class="fields">__CAT_FIELDS__</div></details>
          <details open><summary>Numeric features (38)</summary>
            <div class="fields">__NUM_FIELDS__</div></details>
          <div class="actions">
            <button class="primary" id="predict" type="submit">Classify record</button>
            <button class="secondary" id="sample" type="button">Load training sample</button>
          </div>
          <p class="status" id="status" role="alert" aria-live="polite"></p>
        </form>
        <section class="result" id="result" aria-live="polite" hidden>
          <p class="result-kicker">Baseline model prediction</p>
          <h2 id="decision"></h2>
          <p class="family" id="family"></p>
          <p class="hint" id="threshold"></p>
        </section>
        <p class="disclaimer">Dataset-model output only. A prediction is not a guarantee of
          real-world network security and does not replace monitoring or expert review.</p>
      </section>
      <aside class="card side">
        <h2>About this demo</h2>
        <p>The saved baseline Random Forest models and their fitted preprocessing are
          loaded through ThreatAhead's shared <code>predict_record</code> contract.</p>
        <ul><li>Only one record is classified at a time.</li>
          <li>Features are checked before inference.</li>
          <li>No probability is presented as a risk percentage.</li>
          <li>Sample input comes from the training split only.</li></ul>
      </aside>
    </div>
  </main>
  <footer>ThreatAhead · NSL-KDD demonstration interface · not live traffic monitoring</footer>
  <script>
    const featureNames = __FEATURES__;
    const form = document.getElementById("record-form");
    const status = document.getElementById("status");
    const result = document.getElementById("result");
    const sampleButton = document.getElementById("sample");
    const predictButton = document.getElementById("predict");
    function readRecord() {
      const record = {};
      for (const name of featureNames) {
        const input = document.getElementById(name);
        const value = input.value.trim();
        if (!value) throw new Error(`${name} is required.`);
        if (["protocol_type", "service", "flag"].includes(name)) {
          record[name] = value;
        } else {
          const number = Number(value);
          if (!Number.isFinite(number)) throw new Error(`${name} must be a finite number.`);
          record[name] = number;
        }
      }
      return record;
    }
    sampleButton.addEventListener("click", async () => {
      status.textContent = "";
      sampleButton.disabled = true;
      try {
        const response = await fetch("/api/sample");
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Could not load a sample.");
        for (const name of featureNames) document.getElementById(name).value = data[name];
      } catch (error) {
        status.textContent = error.message;
      } finally {
        sampleButton.disabled = false;
      }
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      status.textContent = "";
      result.hidden = true;
      try {
        const record = readRecord();
        predictButton.disabled = true;
        const response = await fetch("/api/predict", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify(record)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Inference failed.");
        const attack = data.binary_prediction === "Attack";
        result.className = `result ${attack ? "attack" : "normal"}`;
        document.getElementById("decision").textContent = data.binary_prediction;
        document.getElementById("family").textContent =
          attack ? `Predicted family: ${data.attack_family}` : "No attack family assigned.";
        document.getElementById("threshold").textContent =
          `Configuration: ${data.model_config} · ${data.threshold_semantics}`;
        result.hidden = false;
      } catch (error) {
        status.textContent = error.message;
      } finally {
        predictButton.disabled = false;
      }
    });
  </script>
</body>
</html>"""
    categorical_fields = "\n".join(
        f'<label for="{name}">{_feature_label(name)}'
        f'<input id="{name}" name="{name}" type="text" required '
        'autocomplete="off" spellcheck="false"></label>'
        for name in CATEGORICAL_FEATURES
    )
    numeric_fields = "\n".join(
        f'<label for="{name}">{_feature_label(name)}'
        f'<input id="{name}" name="{name}" type="number" step="any" required></label>'
        for name in FEATURE_COLUMNS
        if name not in CATEGORICAL_FEATURES
    )
    page = (
        page.replace("__CAT_FIELDS__", categorical_fields)
        .replace("__NUM_FIELDS__", numeric_fields)
        .replace("__FEATURES__", features_json)
    )
    return page.encode("utf-8")


class _DemoHandler(BaseHTTPRequestHandler):
    server_version = "ThreatAheadDemo/1.0"

    def _send_json(self, status: int, data: Mapping[str, Any]) -> None:
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = _render_page()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/sample":
            try:
                self._send_json(200, load_training_sample())
            except (FileNotFoundError, ValueError) as error:
                self._send_json(503, {"error": str(error)})
            return
        self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self._send_json(404, {"error": "Not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(415, {"error": "Send a JSON request body."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length header."})
            return
        if not 0 < content_length <= MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "Request body is empty or exceeds the size limit."})
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            prediction = predict_payload(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "Request body is not valid UTF-8 JSON."})
        except (TypeError, ValueError) as error:
            self._send_json(400, {"error": str(error)})
        except FileNotFoundError:
            self._send_json(503, {"error": "Baseline model artifacts are missing."})
        else:
            self._send_json(200, prediction)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8501), _DemoHandler)
    print("ThreatAhead demo is available at http://127.0.0.1:8501")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ThreatAhead demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
