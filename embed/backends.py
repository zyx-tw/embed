# The models behind the service. EmbeddingGemma 2 runs in sentence-transformers
# (the only library that runs its vision encoder); its vision tower, which is
# most of the cost of a photo, runs on the Intel GPU through OpenVINO when
# there is one. Faces use InsightFace's buffalo_l (faces.py); text in images
# uses RapidOCR's PP-OCR models through OpenVINO on the CPU.
import base64
import io
import os
import threading
import time
import types

import numpy as np
import torch
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer

from embed import otel
from embed.faces import Faces

MODEL = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
# The model card's prompts: queries and untitled documents.
QUERY = "task: search result | query: "
DOCUMENT = "title: none | text: "
# Bigger images cost more vision tokens for no gain on a photo wall.
MAX_SIDE = 896


def open_image(image_b64):
    image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    return ImageOps.exif_transpose(image).convert("RGB")


class Backends:
    def __init__(self, folder, threads):
        os.environ.setdefault("HF_HOME", folder)
        torch.set_num_threads(threads)
        self.folder = folder
        self.threads = threads
        self.model = SentenceTransformer(
            MODEL,
            revision=REVISION,
            device="cpu",
            # float16 overflows in this model; bfloat16 is slow on most CPUs.
            model_kwargs={"dtype": torch.float32},
            config_kwargs={"audio_config": None},
        )
        self.vision = "torch-cpu"
        if os.environ.get("EMBED_VISION", "auto") != "cpu":
            try:
                self.vision_on_gpu()
            except Exception as error:  # noqa: BLE001 - the CPU path still works
                otel.log("vision.gpu_unavailable", {"error": type(error).__name__, "detail": str(error)[:200]}, "WARN")
        self.face_finder = None
        self.reader = None
        # The face and OCR models download on first start; search must not wait for them.
        threading.Thread(target=self.load_faces, daemon=True).start()
        threading.Thread(target=self.load_ocr, daemon=True).start()

    def status(self):
        return {"model": MODEL, "vision": self.vision, "faces": self.face_finder is not None, "ocr": self.reader is not None}

    def vision_on_gpu(self):
        """Converts the vision tower to OpenVINO once (about a minute), then
        swaps its forward for the compiled model on the GPU. Output matches
        the CPU path to cosine 0.9999 at about 2.5 times the speed on an N150."""
        import openvino as ov
        from transformers.modeling_outputs import BaseModelOutput

        core = ov.Core()
        if "GPU" not in core.available_devices:
            raise RuntimeError("no OpenVINO GPU device")
        tower = self.model[0].auto_model.vision_tower
        path = os.path.join(self.folder, "openvino", f"vision-{REVISION[:12]}.xml")
        if not os.path.exists(path):
            started = time.monotonic()
            sample = self.model[0].tokenize([{"image": Image.new("RGB", (MAX_SIDE, MAX_SIDE), "gray")}])

            class Tower(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(self, pixel_values, pixel_position_ids):
                    out = self.inner(pixel_values=pixel_values, pixel_position_ids=pixel_position_ids)
                    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

            converted = ov.convert_model(
                Tower(tower).eval(),
                example_input={"pixel_values": sample["pixel_values"], "pixel_position_ids": sample["image_position_ids"]},
                input=[("pixel_values", [1, -1, 768]), ("pixel_position_ids", [1, -1, 2])],
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            ov.save_model(converted, path + ".tmp.xml", compress_to_fp16=True)
            os.replace(path + ".tmp.bin", path[:-4] + ".bin")
            os.replace(path + ".tmp.xml", path)
            otel.log("vision.converted", {"ms": round((time.monotonic() - started) * 1000)})
        core.set_property({"CACHE_DIR": os.path.join(self.folder, "openvino", "cache")})
        compiled = core.compile_model(path, "GPU", {"INFERENCE_PRECISION_HINT": "f16", "PERFORMANCE_HINT": "LATENCY"})

        def forward(_self, pixel_values, pixel_position_ids, **_kwargs):
            out = compiled({"pixel_values": pixel_values.numpy(), "pixel_position_ids": pixel_position_ids.numpy()})[0]
            return BaseModelOutput(last_hidden_state=torch.from_numpy(out))

        tower.forward = types.MethodType(forward, tower)
        self.vision = "openvino-gpu"

    def load_faces(self):
        """Retries every minute until the models are downloaded and checked."""
        while self.face_finder is None:
            try:
                self.face_finder = Faces(os.path.join(self.folder, "faces"), self.threads)
            except Exception as error:  # noqa: BLE001 - logged, then retried
                otel.log("faces.load_failed", {"error": str(error)[:200]}, "WARN")
                time.sleep(60)

    def load_ocr(self):
        while self.reader is None:
            try:
                from rapidocr import EngineType, RapidOCR

                engine = EngineType.OPENVINO
                self.reader = RapidOCR(params={"Det.engine_type": engine, "Cls.engine_type": engine, "Rec.engine_type": engine})
            except Exception as error:  # noqa: BLE001 - logged, then retried
                otel.log("ocr.load_failed", {"error": str(error)[:200]}, "WARN")
                time.sleep(60)

    def embed(self, kind, items):
        vectors = []
        for item in items:
            if "image_b64" in item:
                image = open_image(item["image_b64"])
                image.thumbnail((MAX_SIDE, MAX_SIDE))
                source = {"image": image}
            elif isinstance(item.get("text"), str):
                source = (QUERY if kind == "query" else DOCUMENT) + item["text"]
            else:
                raise ValueError("each item needs text or image_b64")
            with torch.inference_mode():
                vector = self.model.encode(source, prompt="", normalize_embeddings=True, show_progress_bar=False)
            vectors.append([round(float(x), 6) for x in vector])
        return vectors

    def faces(self, image_b64):
        if self.face_finder is None:
            raise RuntimeError("face models are still loading")
        image = open_image(image_b64)
        # The models take blue, green, red.
        bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        found = self.face_finder.find(bgr)
        for face in found:
            face["box"] = [round(x, 1) for x in face["box"]]
            face["score"] = round(face["score"], 4)
            face["vector"] = [round(x, 6) for x in face["vector"]]
        return {"width": image.width, "height": image.height, "faces": found}

    def ocr(self, image_b64):
        if self.reader is None:
            raise RuntimeError("OCR models are still loading")
        image = open_image(image_b64)
        result = self.reader(np.asarray(image)[:, :, ::-1])
        lines = []
        for text, score, box in zip(result.txts or (), result.scores or (), result.boxes if result.boxes is not None else ()):
            lines.append({"text": text, "score": round(float(score), 4), "box": [[round(float(x), 1), round(float(y), 1)] for x, y in box]})
        return {"text": "\n".join(line["text"] for line in lines), "lines": lines}
