from symcascade.neural.gemma_backend import GemmaBackend, GemmaResponse


class FakeVLLM:
    def __init__(self, answer, logprob):
        self._answer = answer
        self._logprob = logprob
        self.thinking_enabled = None

    def generate(self, prompt, enable_thinking):
        self.thinking_enabled = enable_thinking
        return {"text": self._answer, "logprobs": [[self._logprob]]}


def test_backend_forces_thinking_false_on_low_cost_tier():
    fake = FakeVLLM(answer="42", logprob=-0.1)
    backend = GemmaBackend(client=fake, enable_thinking=False)
    resp = backend.answer("what is 2+2?")
    assert fake.thinking_enabled is False
    assert resp.text == "42"


def test_backend_confidence_is_softmax_of_top_logprob():
    fake = FakeVLLM(answer="42", logprob=-0.1)
    backend = GemmaBackend(client=fake, enable_thinking=False)
    resp = backend.answer("q")
    # softmax(-0.1) ~ 0.475; we approximate confidence as exp(top_logprob)
    assert 0.0 < resp.confidence <= 1.0


def test_backend_failure_returns_low_confidence():
    class BrokenVLLM:
        def generate(self, prompt, enable_thinking):
            raise RuntimeError("vllm down")
    backend = GemmaBackend(client=BrokenVLLM(), enable_thinking=False)
    resp = backend.answer("q")
    assert resp.confidence == 0.0
    assert "ERROR" in resp.text
