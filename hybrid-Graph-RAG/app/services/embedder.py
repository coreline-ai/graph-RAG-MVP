import time

from app.settings import Settings


class BgeM3Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = None

    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._torch = torch
            self._device = torch.device(self.settings.embedding_device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.settings.embedding_model_name)
            self._model = AutoModel.from_pretrained(self.settings.embedding_model_name)
            self._model.to(self._device)
            self._model.eval()
        return self._model, self._tokenizer, self._torch, self._device

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model, tokenizer, torch, device = self._load_model()
        attempts = 3
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.settings.embedding_max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.no_grad():
                    outputs = model(**encoded)

                # The model's sentence-transformers config uses CLS pooling followed by L2 normalization.
                dense_vectors = outputs.last_hidden_state[:, 0]
                dense_vectors = torch.nn.functional.normalize(dense_vectors, p=2, dim=1)
                return dense_vectors.cpu().tolist()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < attempts:
                    time.sleep(0.5 * attempt)
        raise RuntimeError(f"embedding failed after {attempts} attempts") from last_error
