"""Flask HTTP contract shared by the web dashboard and optional Streamlit client."""
from __future__ import annotations

import logging
import uuid

from flask import Flask, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from config import MACHINE_TYPES, NOISE_LEVELS, Settings, VERSION
from inference import InferenceEngine, PredictionError, validate_upload


class UnavailableEngine:
    def health(self):
        return {"status": "degraded", "version": VERSION, "models": [],
                "preprocessing": {"verified": False},
                "classifier": {"available": False, "detail": "Choose a machine type or a labeled filename."},
                "errors": ["Model initialization failed. Check the server log and exported model files."]}

    def predict(self, **kwargs):
        raise PredictionError("MODEL_UNAVAILABLE", "Inference is unavailable. Check model readiness and the server log.", 503)


def create_app(engine=None, settings: Settings | None = None) -> Flask:
    settings = settings or Settings()
    app = Flask(__name__)
    # Allow multipart framing on top of the advertised file-size limit.
    app.config.update(MAX_CONTENT_LENGTH=settings.max_upload_bytes + 1024 * 1024,
                      MAX_FORM_MEMORY_SIZE=256 * 1024, MAX_FORM_PARTS=10)
    if engine is None:
        try:
            engine = InferenceEngine(settings=settings)
        except Exception:
            app.logger.exception("Inference initialization failed")
            engine = UnavailableEngine()
    app.extensions["inference_engine"] = engine

    @app.before_request
    def request_id():
        g.request_id = uuid.uuid4().hex

    @app.after_request
    def response_headers(response):
        response.headers["X-Request-ID"] = g.get("request_id", "")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def error_response(code, message, status):
        return jsonify(error={"code": code, "message": message}, request_id=g.get("request_id", "")), status

    @app.errorhandler(PredictionError)
    def prediction_error(error):
        return error_response(error.code, error.message, error.status)

    @app.errorhandler(RequestEntityTooLarge)
    def oversized(error):
        return error_response("FILE_TOO_LARGE", "The request exceeds the upload limit. Upload a WAV smaller than 25 MiB.", 413)

    @app.errorhandler(HTTPException)
    def http_error(error):
        return error_response(error.name.upper().replace(" ", "_"), error.description, error.code or 500)

    @app.errorhandler(Exception)
    def unexpected(error):
        app.logger.exception("Request %s failed", g.get("request_id"))
        return error_response("INTERNAL_ERROR", "The recording could not be analyzed. Retry or check the server log using the request ID.", 500)

    @app.get("/")
    def dashboard():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        body = engine.health()
        body["max_upload_bytes"] = settings.max_upload_bytes
        return jsonify(body), 200 if body["status"] == "ready" else 503

    @app.get("/api/models")
    def models():
        body = engine.health()
        return jsonify(models=body.get("models", []), preprocessing=body.get("preprocessing", {}),
                       classifier=body.get("classifier", {}))

    @app.post("/api/predict")
    def predict():
        upload = request.files.get("file")
        if upload is None:
            raise PredictionError("MISSING_FILE", "Send a WAV recording in the multipart file field named 'file'.")
        filename = upload.filename or ""
        data = upload.stream.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise PredictionError("FILE_TOO_LARGE", "The WAV recording exceeds the upload limit.", 413)
        validate_upload(data, filename)
        machine_type = request.form.get("machine_type", "auto")
        snr = request.form.get("snr", "6_dB")
        if machine_type not in (*MACHINE_TYPES, "auto"):
            raise PredictionError("INVALID_MACHINE", "Select Fan, Pump, or Auto from filename.", 422)
        if snr not in NOISE_LEVELS:
            raise PredictionError("INVALID_SNR", "Choose a noise condition of +6, 0, or -6 dB.", 422)
        body = engine.predict(data=data, filename=filename,
                              machine_type=machine_type, snr=snr)
        body["request_id"] = g.request_id
        return jsonify(body)

    return app
