from symcascade.core.types import Query
from symcascade.neural.cloud_stage import CloudStage, GeminiClient


class FakeCloudClient:
    def __init__(self, text="cloud-answer", raise_exc=None):
        self._text = text
        self._raise = raise_exc
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        if self._raise:
            raise self._raise
        return self._text


def test_cloud_stage_returns_success_with_answer():
    client = FakeCloudClient(text="done")
    stage = CloudStage(client=client, cost=1.0, latency_ms=500.0)
    r = stage.run(Query(text="q"))
    assert r.success is True
    assert r.answer == "done"
    assert r.cost == 1.0
    assert r.confidence == 1.0
    assert client.calls == ["q"]


def test_cloud_stage_returns_failure_on_exception():
    client = FakeCloudClient(raise_exc=RuntimeError("api down"))
    stage = CloudStage(client=client)
    r = stage.run(Query(text="q"))
    assert r.success is False
    assert r.answer == ""


def test_cloud_stage_cost_latency_configurable_for_pareto_sweep():
    stage = CloudStage(client=FakeCloudClient(), cost=2.5, latency_ms=900.0)
    r = stage.run(Query(text="q"))
    assert r.cost == 2.5
    assert r.latency_ms == 900.0


class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def generate_content(self, model, contents):
        return _FakeResp(f"gen:{model}:{contents}")


class _FakeGenaiClient:
    def __init__(self):
        self.models = _FakeModels()


def test_gemini_client_uses_injected_client():
    gc = GeminiClient(model_name="gemini-3-pro", client=_FakeGenaiClient())
    assert gc.generate("hello") == "gen:gemini-3-pro:hello"


def test_gemini_client_construct_does_not_import_google_genai():
    import sys
    sys.modules.pop("google", None)
    sys.modules.pop("google.genai", None)
    GeminiClient(model_name="gemini-3-pro")  # no client -> no import yet
    # google may be partially loaded by other libs; only assert genai not imported
    assert "google.genai" not in sys.modules
